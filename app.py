from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import inference_onnx

app = Flask(__name__)

# Kick off model loading in the background — does NOT block
# Gunicorn from binding the port or answering /health.
inference_onnx.start_loading()


# ============================================================
# HEALTH CHECK — process is alive, nothing more
# ============================================================

@app.get("/health")
def health():
    return jsonify({"status": "alive", "service": "retinal-ml"}), 200


# ============================================================
# READINESS CHECK — model is loaded and warmed up
# ============================================================

@app.get("/ready")
def ready():
    state, error = inference_onnx.get_state()

    if state == "ready":
        return jsonify({"status": "ready"}), 200

    if state == "failed":
        return jsonify({"status": "failed", "error": error}), 503

    return jsonify({"status": "loading"}), 503


# ============================================================
# PREDICT FROM CLOUDINARY URL
# ============================================================

@app.post("/predict-url")
def prediction_from_url():

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
        return jsonify({"error": "image_url is required"}), 400

    image_url = data["image_url"]

    # ----------------------------------------------------
    # Download image — failures here are about the URL/
    # network, not the model. Don't conflate with 500.
    # ----------------------------------------------------

    try:
        with urlopen(image_url, timeout=30) as response:
            image_data = response.read()
    except (URLError, HTTPError, TimeoutError) as e:
        return jsonify({
            "status": "image_fetch_failed",
            "error": str(e)
        }), 502

    # ----------------------------------------------------
    # Open image — a bad/corrupt file, not a model problem
    # ----------------------------------------------------

    try:
        image = Image.open(BytesIO(image_data))
        image.load()  # forces decode now, surfaces corrupt files here
    except Exception as e:
        return jsonify({
            "status": "invalid_image",
            "error": str(e)
        }), 422

    # ----------------------------------------------------
    # ML prediction — genuine inference-time failure
    # ----------------------------------------------------

    try:
        result = inference_onnx.predict(image)
    except Exception as e:
        return jsonify({
            "status": "inference_error",
            "error": str(e)
        }), 500

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)