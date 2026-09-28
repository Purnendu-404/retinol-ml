import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "model/Retinol_BEST.onnx"

IMAGE_SIZE = 384
NUM_CLASSES = 5

CLASS_NAMES = [
    "Grade 0",
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4",
]


# ============================================================
# MODEL STATE
# ============================================================

session = None
INPUT_NAME = None
OUTPUT_NAME = None

_model_state = "loading"
_model_error = None

_model_lock = threading.Lock()
_model_thread = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    global session
    global INPUT_NAME
    global OUTPUT_NAME

    print("[MODEL] Loading ONNX model...", flush=True)

    providers = [
        "CPUExecutionProvider"
    ]

    sess = ort.InferenceSession(
        MODEL_PATH,
        providers=providers
    )

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    print(
        f"[MODEL] Input: {input_name}",
        flush=True
    )

    print(
        f"[MODEL] Output: {output_name}",
        flush=True
    )

    print(
        f"[MODEL] Providers: {sess.get_providers()}",
        flush=True
    )

    return sess, input_name, output_name


def _load_model_background():
    global session
    global INPUT_NAME
    global OUTPUT_NAME
    global _model_state
    global _model_error

    try:
        sess, input_name, output_name = load_model()

        session = sess
        INPUT_NAME = input_name
        OUTPUT_NAME = output_name

        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------

        print("[MODEL] Running warmup...", flush=True)

        dummy = np.zeros(
            (
                1,
                4,
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            dtype=np.float32
        )

        session.run(
            [OUTPUT_NAME],
            {
                INPUT_NAME: dummy
            }
        )

        print(
            "[MODEL] Warmup complete.",
            flush=True
        )

        _model_state = "ready"

        print(
            "[MODEL] Model ready.",
            flush=True
        )

    except Exception as e:
        _model_error = str(e)
        _model_state = "failed"

        print(
            f"[MODEL] Loading failed: {_model_error}",
            flush=True
        )


def start_loading():
    global _model_thread

    with _model_lock:

        if _model_state == "ready":
            return

        if (
            _model_thread is not None
            and _model_thread.is_alive()
        ):
            return

        _model_thread = threading.Thread(
            target=_load_model_background,
            daemon=True
        )

        _model_thread.start()


def get_state():
    return _model_state, _model_error


# Start loading when this module is imported.
start_loading()


# ============================================================
# IMAGE HELPERS
# ============================================================

def pil_to_bgr(image):
    """
    Convert PIL image into OpenCV BGR format.
    """

    if image.mode != "RGB":
        image = image.convert("RGB")

    rgb = np.array(image)

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    return bgr


# ============================================================
# RETINAL ROI
# ============================================================

def find_retinal_roi(image):
    """
    Detect the approximate retinal region.

    Uses grayscale thresholding + morphology to
    remove most of the black background around the fundus.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (41, 41)
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    if w <= 0 or h <= 0:
        return image

    return image[
        y:y + h,
        x:x + w
    ]


# ============================================================
# CENTERED SQUARE CROP
# ============================================================

def center_square_crop(image):
    """
    Crop the image into a centered square.
    """

    height, width = image.shape[:2]

    size = min(
        height,
        width
    )

    start_x = max(
        0,
        (width - size) // 2
    )

    start_y = max(
        0,
        (height - size) // 2
    )

    return image[
        start_y:start_y + size,
        start_x:start_x + size
    ]


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(image):
    """
    Apply CLAHE independently to LAB lightness channel.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    lab = cv2.merge(
        (
            l_channel,
            a_channel,
            b_channel
        )
    )

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# ILLUMINATION NORMALIZATION
# ============================================================

def normalize_illumination(image):
    """
    Remove slow illumination variation using
    a large Gaussian background estimate.
    """

    background = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=40
    )

    image_float = image.astype(
        np.float32
    )

    background_float = background.astype(
        np.float32
    )

    normalized = (
        image_float
        / (background_float + 1.0)
    )

    normalized = normalized * 128.0

    normalized = np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)

    return normalized


# ============================================================
# EDGE EXTRACTION
# ============================================================

def extract_edges(image):
    """
    Generate Canny edge channel.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
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
# PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Complete preprocessing pipeline.

    Output:
        tensor       -> (1, 4, 384, 384)
        roi_image    -> processed retinal image
        edges        -> edge image
    """

    image = pil_to_bgr(
        image
    )

    roi_image = find_retinal_roi(
        image
    )

    roi_image = center_square_crop(
        roi_image
    )

    roi_image = cv2.resize(
        roi_image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    roi_image = apply_clahe(
        roi_image
    )

    roi_image = normalize_illumination(
        roi_image
    )

    edges = extract_edges(
        roi_image
    )

    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        roi_image,
        cv2.COLOR_BGR2RGB
    )

    rgb = rgb.astype(
        np.float32
    ) / 255.0

    # ImageNet normalization
    mean = np.array(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=np.float32
    )

    std = np.array(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=np.float32
    )

    rgb = (
        rgb - mean
    ) / std

    # --------------------------------------------------------
    # Edge normalization
    # --------------------------------------------------------

    edge = edges.astype(
        np.float32
    ) / 255.0

    edge = (
        edge - 0.5
    ) / 0.5

    edge = edge[
        np.newaxis,
        ...,
        np.newaxis
    ]

    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    rgb = np.transpose(
        rgb,
        (2, 0, 1)
    )

    edge = np.transpose(
        edge,
        (3, 0, 1, 2)
    )

    edge = edge[0]

    # --------------------------------------------------------
    # RGB + Edge = 4 channels
    # --------------------------------------------------------

    tensor = np.concatenate(
        [
            rgb,
            edge
        ],
        axis=0
    )

    tensor = tensor[
        np.newaxis,
        ...
    ].astype(
        np.float32
    )

    return (
        tensor,
        roi_image,
        edges
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

def run_model(tensor):

    if session is None:
        raise RuntimeError(
            "Model is not loaded yet."
        )

    outputs = session.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME: tensor
        }
    )

    return outputs[0]


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits):

    logits = logits.astype(
        np.float32
    )

    logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exp_values = np.exp(
        logits
    )

    return (
        exp_values
        / np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )
    )


# ============================================================
# IMAGE -> BASE64
# ============================================================

def image_to_base64(image):

    if len(image.shape) == 2:

        success, encoded = cv2.imencode(
            ".jpg",
            image
        )

    else:

        success, encoded = cv2.imencode(
            ".jpg",
            image
        )

    if not success:
        raise RuntimeError(
            "Failed to encode image."
        )

    return base64.b64encode(
        encoded.tobytes()
    ).decode(
        "utf-8"
    )


# ============================================================
# PARALLEL TTA
# ============================================================

def run_tta_parallel(tensor):
    """
    Run the four TTA variants concurrently.

    TTA configuration:

        Original       -> 40%
        Horizontal     -> 25%
        Vertical       -> 20%
        Rotation       -> 15%

    The transformations are performed on the already
    preprocessed tensor, so preprocessing is NOT repeated.

    Returns:
        Weighted-average logits.
    """

    # --------------------------------------------------------
    # Create TTA tensors
    # --------------------------------------------------------

    original = tensor

    horizontal_flip = np.flip(
        tensor,
        axis=3
    ).copy()

    vertical_flip = np.flip(
        tensor,
        axis=2
    ).copy()

    rotation = np.rot90(
        tensor,
        k=1,
        axes=(2, 3)
    ).copy()

    tta_tensors = [
        original,
        horizontal_flip,
        vertical_flip,
        rotation
    ]

    # --------------------------------------------------------
    # TTA weights
    # --------------------------------------------------------

    weights = np.array(
        [
            0.40,
            0.25,
            0.20,
            0.15
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Run four model passes concurrently
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:

        futures = [
            executor.submit(
                run_model,
                tta_tensor
            )
            for tta_tensor in tta_tensors
        ]

        logits_list = [
            future.result()
            for future in futures
        ]

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    print(
        f"[TIMING] parallel TTA inference: "
        f"{inference_time:.3f}s",
        flush=True
    )

    # --------------------------------------------------------
    # Convert each logits output to probabilities
    # --------------------------------------------------------

    probabilities = [
        softmax(logits)[0]
        for logits in logits_list
    ]

    # --------------------------------------------------------
    # Weighted probability average
    # --------------------------------------------------------

    combined_probabilities = np.zeros(
        NUM_CLASSES,
        dtype=np.float32
    )

    for probability, weight in zip(
        probabilities,
        weights
    ):
        combined_probabilities += (
            probability * weight
        )

    # --------------------------------------------------------
    # Convert probabilities back to logits-like format
    #
    # The caller only needs probabilities, so return the
    # combined probabilities directly.
    # --------------------------------------------------------

    return combined_probabilities


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    preprocessing_start = time.perf_counter()

    tensor, roi_image, edges = preprocess_image(
        image
    )

    print(
        f"[TIMING] preprocessing: "
        f"{time.perf_counter() - preprocessing_start:.3f}s",
        flush=True
    )

    # --------------------------------------------------------
    # PARALLEL TTA INFERENCE
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    probs = run_tta_parallel(
        tensor
    )

    print(
        f"[TIMING] TTA total: "
        f"{time.perf_counter() - inference_start:.3f}s",
        flush=True
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probs)
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
    # BASE64 OUTPUTS
    # --------------------------------------------------------

    base64_start = time.perf_counter()

    processed_image_base64 = image_to_base64(
        roi_image
    )

    edge_image_base64 = image_to_base64(
        edges
    )

    print(
        f"[TIMING] base64 encoding: "
        f"{time.perf_counter() - base64_start:.3f}s",
        flush=True
    )

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    print(
        f"[TIMING] predict() TOTAL: "
        f"{time.perf_counter() - total_start:.3f}s",
        flush=True
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "grade": predicted_index,
        "confidence": confidence,
        "probabilities": probabilities,
        "processed_image": processed_image_base64,
        "edge_image": edge_image_base64
    }