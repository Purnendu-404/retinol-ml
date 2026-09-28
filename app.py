from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import time

import inference_onnx

app = Flask(__name__)

@app.post("/predict-url")
def prediction_from_url():

    total_start = time.perf_counter()

    state, error = inference_onnx.get_state()

    if state == "loading":
        return jsonify({
            "status": "model_loading",
            "error": "Model is still starting up. Retry shortly."
        }), 503

    if state == "failed":
        return jsonify({
            "status": "model_failed",
            "error": error
        }), 500

    data = request.get_json()

    if not data or "image_url" not in data:
        return jsonify({
            "error": "image_url is required"
        }), 400

    # -------------------------
    # Download image
    # -------------------------
    start = time.perf_counter()

    try:
        with urlopen(
            data["image_url"],
            timeout=30
        ) as response:
            image_data = response.read()

    except (
        URLError,
        HTTPError,
        TimeoutError
    ) as e:
        return jsonify({
            "status": "image_fetch_failed",
            "error": str(e)
        }), 502

    print(
        f"[TIMING] image download: "
        f"{time.perf_counter() - start:.3f}s",
        flush=True
    )

    # -------------------------
    # Decode image
    # -------------------------
    start = time.perf_counter()

    try:
        image = Image.open(BytesIO(image_data))
        image.load()

    except Exception as e:
        return jsonify({
            "status": "invalid_image",
            "error": str(e)
        }), 422

    print(
        f"[TIMING] image decode: "
        f"{time.perf_counter() - start:.3f}s",
        flush=True
    )

    # -------------------------
    # ML inference
    # -------------------------
    start = time.perf_counter()

    try:
        result = inference_onnx.predict(image)

    except Exception as e:
        return jsonify({
            "status": "inference_error",
            "error": str(e)
        }), 500

    print(
        f"[TIMING] inference + preprocessing + base64: "
        f"{time.perf_counter() - start:.3f}s",
        flush=True
    )

    print(
        f"[TIMING] TOTAL: "
        f"{time.perf_counter() - total_start:.3f}s",
        flush=True
    )

    return jsonify(result), 200