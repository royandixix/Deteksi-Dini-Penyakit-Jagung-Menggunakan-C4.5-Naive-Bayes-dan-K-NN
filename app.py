from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from utils.features import extract_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "ganti-secret-key-ini"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MODEL_FILES = {
    "c45": "c45_model.pkl",
    "naive_bayes": "naive_bayes_model.pkl",
    "knn": "knn_model.pkl",
}

MODEL_NAMES = {
    "c45": "C4.5",
    "naive_bayes": "Naive Bayes",
    "knn": "K-Nearest Neighbor",
}

DIAGNOSIS_INFO = {
    "Bulai": {
        "title": "Terdeteksi Penyakit Bulai",
        "description": "Model mengenali pola gejala yang mengarah pada penyakit bulai pada tanaman jagung.",
        "recommendation": "Pisahkan tanaman yang terindikasi parah, gunakan benih sehat, perbaiki drainase, dan konsultasikan pengendalian penyakit dengan penyuluh pertanian.",
        "badge": "warning",
    },
    "Hawar_Daun": {
        "title": "Terdeteksi Penyakit Hawar Daun",
        "description": "Model mengenali pola kerusakan daun yang mengarah pada hawar daun.",
        "recommendation": "Kurangi kelembapan berlebih, bersihkan sisa tanaman terinfeksi, dan lakukan pengendalian sesuai rekomendasi pertanian setempat.",
        "badge": "danger",
    },
    "Bercak_Daun": {
        "title": "Terdeteksi Penyakit Bercak Daun",
        "description": "Model mengenali pola bercak pada daun jagung yang sesuai dengan kelas bercak daun.",
        "recommendation": "Lakukan sanitasi lahan, pantau penyebaran bercak, dan gunakan pengendalian penyakit bila gejala semakin meluas.",
        "badge": "danger",
    },
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_metrics() -> Dict:
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def models_ready() -> bool:
    if not (MODEL_DIR / "label_encoder.pkl").exists():
        return False
    return all((MODEL_DIR / filename).exists() for filename in MODEL_FILES.values())


def load_models() -> Dict:
    models = {}
    for key, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Model belum ada: {filename}. Jalankan python train_model.py terlebih dahulu.")
        models[key] = joblib.load(path)
    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    return {"models": models, "encoder": encoder}


def display_label(label: str) -> str:
    return label.replace("_", " ")


def predict_with_all_models(image_path: Path) -> List[Dict]:
    bundle = load_models()
    models = bundle["models"]
    encoder = bundle["encoder"]
    features = extract_features(image_path).reshape(1, -1)

    results = []
    for key, model in models.items():
        pred_encoded = int(model.predict(features)[0])
        pred_label = str(encoder.inverse_transform([pred_encoded])[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = float(np.max(probabilities) * 100)

        results.append(
            {
                "key": key,
                "model_name": MODEL_NAMES[key],
                "prediction": pred_label,
                "prediction_display": display_label(pred_label),
                "confidence": confidence,
            }
        )
    return results


@app.route("/")
def index():
    metrics = load_metrics()
    return render_template("index.html", metrics=metrics, ready=models_ready())


@app.route("/predict", methods=["POST"])
def predict():
    if not models_ready():
        flash("Model belum dilatih. Masukkan dataset lalu jalankan: python train_model.py", "warning")
        return redirect(url_for("index"))

    if "image" not in request.files:
        flash("File gambar belum dipilih.", "danger")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("File gambar belum dipilih.", "danger")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Format file tidak didukung. Gunakan JPG, JPEG, PNG, BMP, atau WEBP.", "danger")
        return redirect(url_for("index"))

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    try:
        predictions = predict_with_all_models(save_path)
    except Exception as exc:
        flash(f"Gagal memproses gambar: {exc}", "danger")
        return redirect(url_for("index"))

    metrics = load_metrics()
    best_model_key = metrics.get("best_model")
    if best_model_key:
        final_prediction = next((item for item in predictions if item["key"] == best_model_key), predictions[0])
    else:
        final_prediction = predictions[0]

    final_label = final_prediction["prediction"]
    diagnosis = DIAGNOSIS_INFO.get(
        final_label,
        {
            "title": f"Terdeteksi {display_label(final_label)}",
            "description": "Kelas penyakit mengikuti label dataset yang digunakan saat training.",
            "recommendation": "Lakukan validasi lapangan dan bandingkan dengan gejala tanaman secara langsung.",
            "badge": "secondary",
        },
    )

    return render_template(
        "result.html",
        image_url=url_for("static", filename=f"uploads/{filename}"),
        predictions=predictions,
        final_prediction=final_prediction,
        diagnosis=diagnosis,
        metrics=metrics,
    )


@app.route("/metrics")
def metrics():
    return render_template("metrics.html", metrics=load_metrics(), ready=models_ready())


@app.errorhandler(413)
def too_large(_error):
    flash("Ukuran file terlalu besar. Maksimal 8 MB.", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
