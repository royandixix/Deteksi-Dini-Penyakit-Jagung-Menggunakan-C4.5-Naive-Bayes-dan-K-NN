from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

IMAGE_SIZE = (224, 224)


def _safe_entropy(values: np.ndarray) -> float:
    hist, _ = np.histogram(values.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-np.sum(hist * np.log2(hist)))


def _lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """Ekstraksi tekstur sederhana memakai Local Binary Pattern 8-neighbor."""
    gray = gray.astype(np.uint8)
    center = gray[1:-1, 1:-1]

    lbp = np.zeros_like(center, dtype=np.uint8)
    lbp |= ((gray[:-2, :-2] >= center) << 7).astype(np.uint8)
    lbp |= ((gray[:-2, 1:-1] >= center) << 6).astype(np.uint8)
    lbp |= ((gray[:-2, 2:] >= center) << 5).astype(np.uint8)
    lbp |= ((gray[1:-1, 2:] >= center) << 4).astype(np.uint8)
    lbp |= ((gray[2:, 2:] >= center) << 3).astype(np.uint8)
    lbp |= ((gray[2:, 1:-1] >= center) << 2).astype(np.uint8)
    lbp |= ((gray[2:, :-2] >= center) << 1).astype(np.uint8)
    lbp |= (gray[1:-1, :-2] >= center).astype(np.uint8)

    hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256), density=True)
    return hist.astype(np.float32)


def extract_features(image_path: str | Path) -> np.ndarray:
    """
    Mengubah citra daun jagung menjadi vektor fitur numerik.

    Fitur yang digunakan:
    - statistik warna RGB dan HSV
    - histogram HSV
    - histogram grayscale
    - fitur tekstur LBP
    - fitur tepi/laplacian
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")

    img = cv2.resize(img, IMAGE_SIZE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Statistik warna
    rgb_mean = rgb.reshape(-1, 3).mean(axis=0)
    rgb_std = rgb.reshape(-1, 3).std(axis=0)
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0)
    hsv_std = hsv.reshape(-1, 3).std(axis=0)

    # Histogram warna HSV
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    # Normalisasi histogram agar tidak bergantung pada ukuran gambar
    h_hist = h_hist / (h_hist.sum() + 1e-8)
    s_hist = s_hist / (s_hist.sum() + 1e-8)
    v_hist = v_hist / (v_hist.sum() + 1e-8)

    gray_hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    gray_hist = gray_hist / (gray_hist.sum() + 1e-8)

    # Fitur tekstur dan tepi
    lbp_hist = _lbp_histogram(gray)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    texture_features = np.array(
        [
            gray.mean(),
            gray.std(),
            _safe_entropy(gray),
            magnitude.mean(),
            magnitude.std(),
            laplacian.var(),
        ],
        dtype=np.float32,
    )

    features = np.concatenate(
        [
            rgb_mean,
            rgb_std,
            hsv_mean,
            hsv_std,
            h_hist,
            s_hist,
            v_hist,
            gray_hist,
            lbp_hist,
            texture_features,
        ]
    ).astype(np.float32)

    return features
