import numpy as np
import cv2
import torch
import torch.nn as nn
import base64

from pathlib import Path
from PIL import Image
from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# MODEL_PATH = (
#     BASE_DIR
#     / "model"
#     / "EfficientNetV2S_IDRiD_BEST_FIXED.pt"
# )

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "Retinol_BEST.pt"
)

# MODEL_PATH = (
#     BASE_DIR
#     / "model"
#     / "effnet_v2_s_seed_999_best.pt"
# )

IMAGE_SIZE = 384
NUM_CLASSES = 5
DROPOUT = 0.4

CLASS_NAMES = [
    "Grade 0",
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# NORMALIZATION
# ============================================================

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


def normalize_input(x):
    """
    Same normalization used during training.

    RGB:
        ImageNet mean/std

    Edge:
        (edge - 0.5) / 0.5
    """

    rgb = x[:3]
    edge = x[3:4]

    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    edge = (edge - 0.5) / 0.5

    return torch.cat(
        [rgb, edge],
        dim=0
    ).float()


# ============================================================
# RETINAL ROI
# ============================================================

def detect_retinal_roi(img):
    """
    Detect retinal region using Otsu thresholding
    and morphological operations.
    """

    if img is None:
        raise ValueError("Image is None.")

    h, w = img.shape[:2]

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thresh = cv2.bitwise_not(thresh)

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

    area = cv2.contourArea(contour)

    if area < (0.10 * h * w):
        return img

    x, y, cw, ch = cv2.boundingRect(contour)

    margin = int(
        0.08 * max(cw, ch)
    )

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)

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

    size = min(h, w)

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

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

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

    l, a, b = cv2.split(lab)

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
# MODEL ARCHITECTURE
# ============================================================

def build_model():

    weights = (
        EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )

    model = efficientnet_v2_s(
        weights=weights
    )

    # Original EfficientNet first convolution
    old_conv = model.features[0][0]

    # Change RGB input -> 4 channels
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None)
    )

    with torch.no_grad():

        # Copy RGB weights
        new_conv.weight[:, :3] = (
            old_conv.weight
        )

        # Edge-channel initialization
        new_conv.weight[:, 3:4] = (
            old_conv.weight.mean(
                dim=1,
                keepdim=True
            ) * 0.5
        )

        if old_conv.bias is not None:
            new_conv.bias.copy_(
                old_conv.bias
            )

    model.features[0][0] = new_conv

    # Same classifier used during training
    in_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Sequential(
        nn.Dropout(
            p=DROPOUT
        ),

        nn.Linear(
            in_features,
            256
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            p=DROPOUT * 0.8
        ),

        nn.Linear(
            256,
            NUM_CLASSES
        )
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        f"Loading model from: {MODEL_PATH}"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = build_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print(
        f"Model loaded on: {DEVICE}"
    )

    return model


model = load_model()


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
        ).astype(np.uint8)

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

    display_roi = roi.copy()

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
        ).astype(np.uint8)
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
        ).astype(np.uint8)
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

    # HWC -> CHW
    combined = np.transpose(
        combined,
        (2, 0, 1)
    )

    combined = np.ascontiguousarray(
        combined
    )

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        combined
    ).float()

    tensor = normalize_input(
        tensor
    )

        # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # Display exactly the processed RGB image from the notebook.
    display_roi = processed_rgb

    return tensor, display_roi, edges


def image_to_base64(image):
    # The notebook/Gradio image is RGB.
    # OpenCV expects BGR when encoding a color image.
    if image.ndim == 3 and image.shape[2] == 3:
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
        raise ValueError("Failed to encode image.")

    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(image):

    tensor, roi_image, edges = preprocess_image(image)

    tensor = tensor.to(DEVICE)

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    logits1 = model(
        tensor
    )

    probs1 = torch.softmax(
        logits1,
        dim=1
    )

    # --------------------------------------------------------
    # Horizontal flip
    # --------------------------------------------------------

    tensor_h = torch.flip(
        tensor,
        dims=[3]
    )

    logits2 = model(
        tensor_h
    )

    probs2 = torch.softmax(
        logits2,
        dim=1
    )

    # --------------------------------------------------------
    # Vertical flip
    # --------------------------------------------------------

    tensor_v = torch.flip(
        tensor,
        dims=[2]
    )

    logits3 = model(
        tensor_v
    )

    probs3 = torch.softmax(
        logits3,
        dim=1
    )

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    tensor_r = torch.rot90(
        tensor,
        k=1,
        dims=[2, 3]
    )

    logits4 = model(
        tensor_r
    )

    probs4 = torch.softmax(
        logits4,
        dim=1
    )

    # print("\n========== TTA DEBUG ==========")

    # print("Original :", probs1[0].cpu().numpy())
    # print("H-Flip   :", probs2[0].cpu().numpy())
    # print("V-Flip   :", probs3[0].cpu().numpy())
    # print("Rotation :", probs4[0].cpu().numpy())

    # print("\nTensor:")
    # print("shape:", tensor.shape)
    # print("min:", tensor.min().item())
    # print("max:", tensor.max().item())
    # print("mean:", tensor.mean().item())
    # print("std:", tensor.std().item())

    # print("===============================\n")

    # --------------------------------------------------------
    # Weighted TTA
    # Same weights as retinol.py
    # --------------------------------------------------------

    probs = (
        0.40 * probs1
        + 0.25 * probs2
        + 0.20 * probs3
        + 0.15 * probs4
    )

    probs = probs[0].cpu().numpy()

    predicted_index = int(
        np.argmax(probs)
    )

    confidence = float(
        probs[predicted_index]
    )

    probabilities = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(NUM_CLASSES)
    }

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "grade": predicted_index,
        "confidence": confidence,
        "probabilities": probabilities,
        "processed_image": image_to_base64(roi_image),
        "edge_image": image_to_base64(edges)
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("INFERENCE MODULE READY")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Input: 4 x {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Classes: {CLASS_NAMES}")
    print("=" * 60)