from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import inference_onnx

app = Flask(__name__)

inference_onnx.start_loading()


@app.get("/health")
def health():
    state, error = inference_onnx.get_state()

    if state == "ready":
        return jsonify({"status": "ready"}), 200

    if state == "failed":
        return jsonify({
            "status": "failed",
            "error": error
        }), 503

    return jsonify({"status": "loading"}), 503


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

    try:
        with urlopen(data["image_url"], timeout=30) as response:
            image_data = response.read()
    except (URLError, HTTPError, TimeoutError) as e:
        return jsonify({
            "status": "image_fetch_failed",
            "error": str(e)
        }), 502

    try:
        image = Image.open(BytesIO(image_data))
        image.load()
    except Exception as e:
        return jsonify({
            "status": "invalid_image",
            "error": str(e)
        }), 422

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