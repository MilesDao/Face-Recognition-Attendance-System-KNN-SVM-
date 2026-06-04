# Face Recognition Attendance System

A real-time face recognition attendance system using **FaceNet embeddings** + **SVM (RBF)** and **KNN (cosine)** classifiers, deployed via **Streamlit** with browser-based WebRTC camera streaming.

## Project Structure

```
face-class-cosine/
├── svm/                          # SVM classifier
│   ├── train.py                  #   training script
│   └── svm_face_classifier.pkl   #   trained model (100% accuracy)
├── knn/                          # KNN classifier
│   ├── train.py                  #   training script
│   └── knn_face_classifier.pkl   #   trained model (100% accuracy)
├── extract_faces.py              # Face extraction from videos using MTCNN
├── augment_data.py               # Data augmentation + distribution plots
├── analyze_distribution.py       # Regenerate PCA / t-SNE / distribution figures
├── train_comparison.py           # Side-by-side SVM vs KNN training + figures
├── app.py                        # Streamlit WebRTC real-time attendance app
├── train_embeddings.npz          # Cached FaceNet embeddings (1505 x 512)
├── train.py                      # (deprecated, use svm/train.py)
├── dataset/
│   ├── train_raw/                # Cleaned raw extracted faces (301 images)
│   └── train_augmented/          # Augmented dataset (1505 images)
├── figures/                      # All generated figures (PNG)
│   ├── distribution_before.png
│   ├── distribution_after.png
│   ├── embedding_pca.png
│   ├── embedding_tsne.png
│   ├── svm_vs_knn_accuracy.png
│   ├── svm_vs_knn_timing.png
│   ├── knn_parameter_search.png
│   ├── svm_parameter_search.png
│   ├── metrics_comparison.png
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_knn.png
│   └── per_class_accuracy.png
├── report/
│   └── report_en.tex             # English LaTeX report
├── Video_/                       # Original input videos (22 .mov files)
├── requirements.txt
└── .gitignore
```

## Pipeline

1. **Face Extraction** — MTCNN detects and crops faces from 22 classroom videos, sampled at 1 fps with confidence > 0.95. Manual noise removal: **320 → 301 clean images**.
2. **Data Augmentation** — 4 transformations per image (horizontal flip, ±10° rotation, brightness/contrast adjustment, Gaussian blur) → **1,505 images**.
3. **Feature Extraction** — FaceNet InceptionResNetV1 (VGGFace2 pretrained) outputs 512-dimensional embeddings.
4. **Classification** — SVM (RBF, C=10, gamma=scale) and KNN (cosine distance, k=1 via 5-fold CV) both achieve **100% accuracy** on the test set.
5. **Real-time Attendance** — Streamlit web app with WebRTC browser camera, model selector (SVM/KNN), confidence threshold slider, attendance log with CSV export.

## Results

| Metric | SVM (RBF) | KNN (k=1, cosine) |
|---|---|---|
| Test Accuracy | 100.00% | 100.00% |
| Weighted Precision / Recall / F1 | 100.00% | 100.00% |
| Top-2 Accuracy | 100.00% | 100.00% |
| Mean Confidence | 80.78% | 100.00% |
| Training Time | 0.53s | 0.00s |
| Inference per sample | 0.20ms | 0.05ms |

Both models classify perfectly. The key trade-off: SVM provides calibrated confidence scores (80.78%), while KNN is simpler, trains instantly, and has faster inference.

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
# Extract faces from videos (if not already done)
python extract_faces.py

# Augment dataset (if not already done)
python augment_data.py

# Train SVM only
python svm\train.py

# Train KNN only
python knn\train.py

# Train both + generate all comparison figures
python train_comparison.py

# Regenerate analysis figures (PCA, t-SNE, distributions)
python analyze_distribution.py
```

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Select SVM or KNN in the sidebar, adjust confidence threshold (default 0.5), click **Start**, and point the webcam at faces. Attendance is logged once per session per person and can be downloaded as CSV.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA (recommended) or CPU
- Webcam
- Windows / Linux / macOS

Key libraries: `facenet-pytorch`, `streamlit`, `streamlit-webrtc`, `scikit-learn`, `opencv-python-headless`, `av`.
