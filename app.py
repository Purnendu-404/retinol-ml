from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from urllib.request import urlopen

from inference_onnx import predict


app = Flask(__name__)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return jsonify({
        "status": "ok",
        "service": "retinal-ml"
    })


# ============================================================
# PREDICT FROM CLOUDINARY URL
# ============================================================

@app.post("/predict-url")
def prediction_from_url():

    try:

        data = request.get_json()

        if not data or "image_url" not in data:

            return jsonify({
                "error": "image_url is required"
            }), 400

        image_url = data["image_url"]

        # ----------------------------------------------------
        # Download image
        # ----------------------------------------------------

        with urlopen(
            image_url,
            timeout=30
        ) as response:

            image_data = response.read()

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            BytesIO(image_data)
        )

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        result = predict(
            image
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify(
            result
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )