Markdown# Deteksi Dini Penyakit Tanaman Jagung

Aplikasi berbasis **Flask + Python** untuk membandingkan performa algoritma **C4.5 (Decision Tree)**, **Naive Bayes**, dan **K-Nearest Neighbor (K-NN)** dalam klasifikasi penyakit pada daun jagung.

Aplikasi ini dirancang fokus pada fungsionalitas inti:
* Training model langsung dari dataset citra daun jagung.
* Komparasi nilai akurasi antara algoritma C4.5, Naive Bayes, dan K-NN.
* Fitur upload citra daun jagung untuk prediksi instan.
* Menampilkan algoritma terbaik berdasarkan hasil evaluasi metrik.

---

## 📂 Struktur Dataset

Masukkan gambar daun jagung ke dalam folder dengan struktur berikut:

```text
dataset/
├── Bulai/
├── Hawar_Daun/
└── Bercak_Daun/
Catatan Format: Gunakan ekstensi gambar .jpg, .jpeg, .png, atau .webp.⚙️ Cara Menjalankan Aplikasi1. Buat & Aktifkan Virtual EnvironmentBash# Membuat venv
python -m venv venv

# Mengaktifkan venv (Mac/Linux)
source venv/bin/activate

# Mengaktifkan venv (Windows)
venv\Scripts\activate
2. Install DependenciesBashpip install -r requirements.txt
3. Masukkan Dataset ke FolderPastikan gambar dimasukkan sesuai dengan kelas penyakitnya masing-masing:Plaintextdataset/Bulai/bulai_001.jpg
dataset/Hawar_Daun/hawar_001.jpg
dataset/Bercak_Daun/bercak_001.jpg
4. Training ModelJalankan script untuk melatih ketiga algoritma:Bashpython train_model.py
Setelah proses training selesai, model dan metrik evaluasi akan otomatis tersimpan di folder model/:Plaintextmodel/
├── c45_model.pkl
├── naive_bayes_model.pkl
├── knn_model.pkl
├── label_encoder.pkl
└── metrics.json
5. Jalankan Server FlaskBashpython app.py
Buka browser dan akses URL berikut:Plaintext[http://127.0.0.1:5000](http://127.0.0.1:5000)
🛠️ Tool Bantuan: Download & Penyusunan DatasetBagian ini digunakan jika Anda ingin mengunduh dan menyusun dataset otomatis dari sumber publik (Kaggle) menggunakan script pembantu yang tersedia di folder tools/.Representasi Kelas DatasetHawar Daun: Menggunakan kelas Blight dari Corn or Maize Leaf Disease Dataset.Bercak Daun: Menggunakan kelas Gray Leaf Spot dari Corn or Maize Leaf Disease Dataset.Bulai: Menggunakan dataset Corn Downy Mildew dari The DoctorP Project (sebagai representasi penyakit Bulai).Langkah-Langkah:Install Package Pembantu:Bashpython3 -m pip install kagglehub
Download & Susun Hawar Daun dan Bercak Daun:Bashpython3 tools/susun_dataset.py --download-kaggle
Download & Susun Kelas Bulai (Downy Mildew):Bashpython3 - <<'PY'
import shutil
from pathlib import Path
import kagglehub

print("Download dataset Bulai dari Kaggle DoctorP...")
root = Path(kagglehub.dataset_download("alexanderuzhinskiy/the-doctorp-project-dataset"))
print("Dataset tersimpan di:", root)

target = Path("dataset/Bulai")
target.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".webp"}

candidates = []
for folder in root.rglob("*"):
    if folder.is_dir():
        name = folder.name.lower()
        if "corn" in name and "downy" in name:
            candidates.append(folder)

if not candidates:
    print("Folder Corn downy mildew tidak ditemukan.")
    print("Coba cek folder dataset ini:", root)
    raise SystemExit

source = candidates[0]
print("Folder Bulai ditemukan:", source)

count = 0
for img in source.rglob("*"):
    if img.suffix.lower() in exts:
        count += 1
        shutil.copy2(img, target / f"bulai_{count:03d}{img.suffix.lower()}")

print(f"Berhasil salin {count} gambar ke {target.resolve()}")
PY
Verifikasi Jumlah Dataset:Bashls dataset/Bulai | wc -l
ls dataset/Hawar_Daun | wc -l
ls dataset/Bercak_Daun | wc -l
📋 Rekomendasi Distribusi Dataset SkripsiUntuk memenuhi target standar pengujian minimal (~250 citra) dengan distribusi yang seimbang, berikut acuan komposisi data:Kelas PenyakitJumlah GambarKeterangan SumberBulai80Kaggle (Downy Mildew / Tambahan)Hawar Daun85Kaggle (Blight)Bercak Daun85Kaggle (Gray Leaf Spot)Total250Seimbang (Recommended)📌 Catatan PentingAlur Eksekusi: Aplikasi Flask tidak akan menampilkan hasil prediksi atau komparasi sebelum script train_model.py dijalankan.Pendekatan C4.5: Implementasi algoritma C4.5 menggunakan pendekatan Decision Tree berbasis Entropy (menggunakan kustomisasi scikit-learn), dikarenakan library tersebut tidak menyediakan algoritma C4.5 murni atau J48 secara bawaan.Keseimbangan Data: Hasil akurasi sangat dipengaruhi oleh kuantitas dan keseimbangan dataset pada tiap kelas. Pastikan jumlah data seimbang agar kesimpulan penelitian valid.Manajemen Repository (.gitignore): Hindari mengunggah folder dataset berukuran besar ke GitHub. Pastikan folder dataset/ telah didaftarkan ke dalam file .gitignore.Plaintext# Contoh isi .gitignore
dataset/
model/
__pycache__/
*.pyc
.DS_Store
venv/# Deteksi-Dini-Penyakit-Jagung-Menggunakan-C4.5-Naive-Bayes-dan-K-NN
