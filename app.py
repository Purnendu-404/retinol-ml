from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from urllib.request import urlopen

from inference import predict

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "retinal-ml"
    })


@app.post("/predict-url")
def prediction_from_url():
    try:
        data = request.get_json()

        if not data or "image_url" not in data:
            return jsonify({
                "error": "image_url is required"
            }), 400

        image_url = data["image_url"]

        with urlopen(image_url, timeout=30) as response:
            image_data = response.read()

        image = Image.open(BytesIO(image_data))

        result = predict(image)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    import os

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=False
    )