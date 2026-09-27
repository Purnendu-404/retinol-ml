import numpy as np
import cv2
import base64
import threading
import onnxruntime as ort

from pathlib import Path
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "Retinol_BEST.onnx"
)

IMAGE_SIZE = 384
NUM_CLASSES = 5

CLASS_NAMES = [
    "Grade 0",
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4"
]


# ============================================================
# NORMALIZATION
# ============================================================

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(3, 1, 1)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(3, 1, 1)


def normalize_input(x):
    """
    Same normalization used by the original PyTorch model.

    RGB:
        ImageNet mean/std

    Edge:
        (edge - 0.5) / 0.5
    """

    rgb = x[:3]

    edge = x[3:4]

    rgb = (
        rgb - IMAGENET_MEAN
    ) / IMAGENET_STD

    edge = (
        edge - 0.5
    ) / 0.5

    return np.concatenate(
        [rgb, edge],
        axis=0
    ).astype(np.float32)


# ============================================================
# RETINAL ROI
# ============================================================

def detect_retinal_roi(img):
    """
    Detect retinal region using Otsu thresholding
    and morphological operations.
    """

    if img is None:
        raise ValueError(
            "Image is None."
        )

    h, w = img.shape[:2]

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    thresh = cv2.bitwise_not(
        thresh
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (41, 41)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return img

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        contour
    )

    if area < (0.10 * h * w):
        return img

    x, y, cw, ch = cv2.boundingRect(
        contour
    )

    margin = int(
        0.08 * max(cw, ch)
    )

    x1 = max(
        0,
        x - margin
    )

    y1 = max(
        0,
        y - margin
    )

    x2 = min(
        w,
        x + cw + margin
    )

    y2 = min(
        h,
        y + ch + margin
    )

    crop = img[
        y1:y2,
        x1:x2
    ]

    return (
        crop
        if crop.size > 0
        else img
    )


# ============================================================
# SQUARE CROP
# ============================================================

def square_crop(img):
    """
    Extract a centered square region.
    """

    h, w = img.shape[:2]

    size = min(
        h,
        w
    )

    cx = w // 2
    cy = h // 2

    half = size // 2

    x1 = max(
        0,
        cx - half
    )

    y1 = max(
        0,
        cy - half
    )

    x2 = min(
        w,
        x1 + size
    )

    y2 = min(
        h,
        y1 + size
    )

    crop = img[
        y1:y2,
        x1:x2
    ]

    side = min(
        crop.shape[0],
        crop.shape[1]
    )

    return crop[
        :side,
        :side
    ]


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(img):
    """
    Same CLAHE preprocessing used during training.
    """

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(
        l
    )

    enhanced = cv2.merge(
        [l, a, b]
    )

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# ILLUMINATION NORMALIZATION
# ============================================================

def normalize_illumination(img):
    """
    Background illumination normalization.
    """

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    background = cv2.GaussianBlur(
        l,
        (0, 0),
        sigmaX=40
    )

    corrected = (
        l.astype(np.float32)
        - background.astype(np.float32)
        + 128.0
    )

    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(np.uint8)

    output = cv2.merge(
        [corrected, a, b]
    )

    return cv2.cvtColor(
        output,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# EDGE EXTRACTION
# ============================================================

def extract_edges(img):
    """
    Canny edge extraction used by the trained model.
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        1.0
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    edges = cv2.erode(
        edges,
        kernel,
        iterations=1
    )

    return edges


# ============================================================
# MODEL STATE
# ============================================================

_session = None
_input_name = None
_output_name = None

_state = "loading"
_error = None

_lock = threading.Lock()

# Prevent multiple background loading threads
_loading_started = False


def _load_model_sync():
    global _session
    global _input_name
    global _output_name
    global _state
    global _error

    try:

        print(
            "========== MODEL LOADING START ==========",
            flush=True
        )

        print(
            f"Loading ONNX model from: {MODEL_PATH}",
            flush=True
        )

        print(
            "Creating ONNX Runtime session...",
            flush=True
        )

        session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=[
                "CPUExecutionProvider"
            ]
        )

        print(
            "ONNX Runtime session CREATED.",
            flush=True
        )

        input_name = (
            session
            .get_inputs()[0]
            .name
        )

        output_name = (
            session
            .get_outputs()[0]
            .name
        )

        print(
            "Starting warmup inference...",
            flush=True
        )

        dummy_input = np.zeros(
            (
                1,
                4,
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            dtype=np.float32
        )

        session.run(
            [output_name],
            {
                input_name: dummy_input
            }
        )

        print(
            "WARMUP COMPLETE.",
            flush=True
        )

        with _lock:

            _session = session
            _input_name = input_name
            _output_name = output_name

            _state = "ready"

        print(
            "========== MODEL READY ==========",
            flush=True
        )

        print(
            f"Input: {input_name}",
            flush=True
        )

        print(
            f"Output: {output_name}",
            flush=True
        )

        print(
            f"Provider: {session.get_providers()}",
            flush=True
        )

    except Exception as e:

        with _lock:

            _state = "failed"
            _error = str(e)

        print(
            f"MODEL LOAD FAILED: {e}",
            flush=True
        )


def start_loading():
    """
    Start model loading exactly once.

    Loading runs on a daemon thread so Flask/Gunicorn
    can continue serving requests while the model loads.
    """

    global _loading_started

    with _lock:

        if _loading_started:
            return

        _loading_started = True

    print(
        "Starting background model loader...",
        flush=True
    )

    thread = threading.Thread(
        target=_load_model_sync,
        daemon=True
    )

    thread.start()


def get_state():
    """
    Returns (state, error) where state is one of:
        "loading"
        "ready"
        "failed"
    """

    with _lock:

        return (
            _state,
            _error
        )


def get_session():
    """
    Returns:
        session
        input_name
        output_name
    """

    with _lock:

        return (
            _session,
            _input_name,
            _output_name
        )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # Convert to NumPy RGB
    # --------------------------------------------------------

    if isinstance(
        image,
        Image.Image
    ):

        image = image.convert(
            "RGB"
        )

        image = np.asarray(
            image
        )

    else:

        image = np.asarray(
            image
        )

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    if image.ndim == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    # --------------------------------------------------------
    # RGBA
    # --------------------------------------------------------

    if (
        image.ndim == 3
        and image.shape[2] == 4
    ):

        image = image[:, :, :3]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):

        raise ValueError(
            f"Invalid image shape: {image.shape}"
        )

    # --------------------------------------------------------
    # Convert to uint8
    # --------------------------------------------------------

    if image.dtype != np.uint8:

        if np.issubdtype(
            image.dtype,
            np.floating
        ):

            if image.max() <= 1.0:

                image = image * 255.0

        image = np.clip(
            image,
            0,
            255
        ).astype(
            np.uint8
        )

    image = np.ascontiguousarray(
        image
    )

    # --------------------------------------------------------
    # RGB -> BGR
    # --------------------------------------------------------

    bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    roi = detect_retinal_roi(
        bgr
    )

    # --------------------------------------------------------
    # Square crop
    # --------------------------------------------------------

    roi = square_crop(
        roi
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    roi = cv2.resize(
        roi,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    roi = apply_clahe(
        roi
    )

    # --------------------------------------------------------
    # Illumination normalization
    # --------------------------------------------------------

    roi = normalize_illumination(
        roi
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    processed_rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

    processed_rgb = (
        np.ascontiguousarray(
            processed_rgb
        ).astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    edges = extract_edges(
        roi
    )

    edges = (
        np.ascontiguousarray(
            edges
        ).astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # Convert to 0-1
    # --------------------------------------------------------

    rgb_float = (
        processed_rgb.astype(
            np.float32
        ) / 255.0
    )

    edge_float = (
        edges.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # RGB + EDGE
    # --------------------------------------------------------

    combined = np.concatenate(
        [
            rgb_float,
            edge_float[..., None]
        ],
        axis=2
    )

    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    combined = np.transpose(
        combined,
        (2, 0, 1)
    )

    combined = (
        np.ascontiguousarray(
            combined
        ).astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    combined = normalize_input(
        combined
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    tensor = np.expand_dims(
        combined,
        axis=0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Display exactly the processed RGB image
    # --------------------------------------------------------

    display_roi = processed_rgb

    return (
        tensor,
        display_roi,
        edges
    )


# ============================================================
# IMAGE -> BASE64
# ============================================================

def image_to_base64(image):

    # Processed image is RGB.
    # OpenCV expects BGR for color encoding.

    if (
        image.ndim == 3
        and image.shape[2] == 3
    ):

        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    else:

        image_bgr = image

    success, buffer = cv2.imencode(
        ".png",
        image_bgr
    )

    if not success:

        raise ValueError(
            "Failed to encode image."
        )

    return base64.b64encode(
        buffer.tobytes()
    ).decode(
        "utf-8"
    )


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits):

    logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exp_logits = np.exp(
        logits
    )

    return (
        exp_logits
        / np.sum(
            exp_logits,
            axis=1,
            keepdims=True
        )
    )


# ============================================================
# ONNX INFERENCE
# ============================================================

def run_model(tensor):
    """
    Run one forward pass through ONNX Runtime.
    """

    session, input_name, output_name = (
        get_session()
    )

    if session is None:
        raise RuntimeError(
            "ONNX model session is not ready."
        )

    logits = session.run(
        [output_name],
        {
            input_name: tensor
        }
    )[0]

    return logits


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    tensor, roi_image, edges = (
        preprocess_image(image)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    logits1 = run_model(
        tensor
    )

    probs1 = softmax(
        logits1
    )

    # --------------------------------------------------------
    # Horizontal flip
    # --------------------------------------------------------

    tensor_h = np.flip(
        tensor,
        axis=3
    ).copy()

    logits2 = run_model(
        tensor_h
    )

    probs2 = softmax(
        logits2
    )

    # --------------------------------------------------------
    # Vertical flip
    # --------------------------------------------------------

    tensor_v = np.flip(
        tensor,
        axis=2
    ).copy()

    logits3 = run_model(
        tensor_v
    )

    probs3 = softmax(
        logits3
    )

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    tensor_r = np.rot90(
        tensor,
        k=1,
        axes=(2, 3)
    ).copy()

    logits4 = run_model(
        tensor_r
    )

    probs4 = softmax(
        logits4
    )

    # --------------------------------------------------------
    # Weighted TTA
    #
    # Original:       0.40
    # Horizontal:     0.25
    # Vertical:       0.20
    # Rotation:       0.15
    # --------------------------------------------------------

    probs = (
        0.40 * probs1
        + 0.25 * probs2
        + 0.20 * probs3
        + 0.15 * probs4
    )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    probs = probs[0]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(
            probs
        )
    )

    confidence = float(
        probs[predicted_index]
    )

    probabilities = {
        CLASS_NAMES[i]: float(
            probs[i]
        )
        for i in range(NUM_CLASSES)
    }

    # --------------------------------------------------------
    # Return API structure
    # --------------------------------------------------------

    return {
        "prediction": CLASS_NAMES[
            predicted_index
        ],

        "grade": predicted_index,

        "confidence": confidence,

        "probabilities": probabilities,

        "processed_image": image_to_base64(
            roi_image
        ),

        "edge_image": image_to_base64(
            edges
        )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ONNX INFERENCE MODULE — MANUAL TEST")
    print("=" * 60)

    start_loading()

    import time

    while True:

        state, error = get_state()

        if state == "ready":
            break

        if state == "failed":

            print(
                f"Load failed: {error}"
            )

            raise SystemExit(1)

        time.sleep(0.5)

    session, input_name, output_name = (
        get_session()
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Input: 4 x {IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"Classes: {CLASS_NAMES}"
    )

    print(
        f"Provider: {session.get_providers()}"
    )

    print("=" * 60)