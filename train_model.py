from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils.features import extract_features

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
GENERATED_DIR = BASE_DIR / "static" / "generated"

CLASS_FOLDERS = ["Bulai", "Hawar_Daun", "Bercak_Daun"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42
TEST_SIZE = 0.2


def collect_images() -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []

    for class_name in CLASS_FOLDERS:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            class_dir.mkdir(parents=True, exist_ok=True)

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, class_name))

    return items


def build_dataset(items: List[Tuple[Path, str]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    features = []
    labels = []
    class_counts: Dict[str, int] = {name: 0 for name in CLASS_FOLDERS}

    for image_path, label in items:
        try:
            features.append(extract_features(image_path))
            labels.append(label)
            class_counts[label] += 1
            print(f"OK  : {image_path}")
        except Exception as exc:
            print(f"SKIP: {image_path} -> {exc}")

    if not features:
        raise RuntimeError("Dataset kosong. Masukkan gambar ke folder dataset terlebih dahulu.")

    return np.array(features, dtype=np.float32), np.array(labels), class_counts


def make_models(n_train: int) -> Dict[str, Pipeline]:
    # K harus lebih kecil/sama dari jumlah data training.
    k = min(5, n_train)
    if k % 2 == 0 and k > 1:
        k -= 1
    k = max(k, 1)

    return {
        "c45": Pipeline(
            steps=[
                (
                    "model",
                    DecisionTreeClassifier(
                        criterion="entropy",
                        random_state=RANDOM_STATE,
                        min_samples_leaf=1,
                    ),
                )
            ]
        ),
        "naive_bayes": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=k, metric="euclidean")),
            ]
        ),
    }


def model_display_name(key: str) -> str:
    names = {
        "c45": "C4.5",
        "naive_bayes": "Naive Bayes",
        "knn": "K-Nearest Neighbor",
    }
    return names.get(key, key)


def save_confusion_matrix_plot(cm: np.ndarray, labels: List[str], model_key: str) -> str:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"confusion_{model_key}.png"
    output_path = GENERATED_DIR / filename

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {model_display_name(model_key)}")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")

    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return f"generated/{filename}"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    items = collect_images()
    if not items:
        print("Dataset belum ditemukan.")
        print("Masukkan gambar ke folder:")
        for folder in CLASS_FOLDERS:
            print(f"- dataset/{folder}/")
        raise SystemExit(1)

    X, y_text, class_counts = build_dataset(items)

    available_classes = sorted(set(y_text.tolist()))
    if len(available_classes) < 2:
        raise RuntimeError("Minimal harus ada 2 kelas gambar untuk training.")

    # Stratify hanya aman jika setiap kelas minimal punya 2 data.
    min_count = min(class_counts[label] for label in available_classes)
    stratify = y_text if min_count >= 2 else None

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    class_labels = encoder.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    models = make_models(n_train=len(X_train))
    metrics = {
        "class_labels": class_labels,
        "class_counts": class_counts,
        "total_images": int(len(X)),
        "train_images": int(len(X_train)),
        "test_images": int(len(X_test)),
        "test_size": TEST_SIZE,
        "models": {},
    }

    for key, model in models.items():
        print(f"\nTraining {model_display_name(key)}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_labels))))
        cm_image = save_confusion_matrix_plot(cm, class_labels, key)

        model_metrics = {
            "display_name": model_display_name(key),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_image": cm_image,
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=class_labels,
                output_dict=True,
                zero_division=0,
            ),
        }
        metrics["models"][key] = model_metrics

        model_filename = {
            "c45": "c45_model.pkl",
            "naive_bayes": "naive_bayes_model.pkl",
            "knn": "knn_model.pkl",
        }[key]
        joblib.dump(model, MODEL_DIR / model_filename)

        print(
            f"{model_display_name(key)} -> "
            f"Accuracy: {model_metrics['accuracy']:.4f}, "
            f"Precision: {model_metrics['precision']:.4f}, "
            f"Recall: {model_metrics['recall']:.4f}, "
            f"F1: {model_metrics['f1_score']:.4f}"
        )

    best_key = max(metrics["models"], key=lambda m: metrics["models"][m]["accuracy"])
    metrics["best_model"] = best_key
    metrics["best_model_name"] = model_display_name(best_key)

    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nTraining selesai.")
    print(f"Algoritma terbaik: {metrics['best_model_name']}")
    print(f"Model tersimpan di: {MODEL_DIR}")


if __name__ == "__main__":
    main()
