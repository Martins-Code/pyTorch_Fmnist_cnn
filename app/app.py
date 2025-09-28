import base64
from flask import Flask, render_template, request, redirect, url_for
from scripts.predict import predict_from_bytes

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # limit uploads to 4 MB (optional)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        # Read bytes once (FileStorage stream)
        image_bytes = file.read()

        # 1) Run prediction using same bytes
        idx, label, confidence = predict_from_bytes(image_bytes)

        # 2) Build a data URI for preview (use the uploaded MIME type)
        mime = file.mimetype or "image/png"  # fallback if missing
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data = f"data:{mime};base64,{image_b64}"

        # 3) Prepare confidence as percent for nicer display
        confidence_pct = round(confidence * 100, 2)

        # 4) Render result template and pass image_data
        return render_template(
            "result.html",
            label=label,
            confidence=confidence_pct,
            image_data=image_data
        )

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
