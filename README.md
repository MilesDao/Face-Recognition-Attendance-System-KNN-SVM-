# Face Recognition Attendance System

A real-time face recognition attendance system built with **FaceNet embeddings** + **SVM** and **KNN** classifiers, deployed via **Streamlit**.

## Project Structure

```
face-class-cosine/
├── svm/                          # SVM classifier
│   ├── train.py                  #   training script
│   └── svm_face_classifier.pkl   #   trained model
├── knn/                          # KNN classifier
│   ├── train.py                  #   training script
│   └── knn_face_classifier.pkl   #   trained model
├── extract_faces.py              # Face extraction from videos
├── augment_data.py               # Data augmentation + distribution plots
├── analyze_distribution.py       # Regenerate analysis figures
├── app.py                        # Streamlit real-time attendance app
├── train_comparison.py           # Side-by-side SVM vs KNN training
├── requirements.txt
├── .gitignore
└── report/
    ├── report_en.tex             # English report
    └── report.tex                # Vietnamese report
```

## Pipeline

1. **Face Extraction** — MTCNN detects faces from 22 classroom videos → 320 images
2. **Data Augmentation** — horizontal flip, rotation, brightness/contrast, Gaussian blur → 1,600 images
3. **Feature Extraction** — FaceNet InceptionResNetV1 (VGGFace2 pretrained) → 512-d embeddings
4. **Classification** — SVM (RBF, C=10) and KNN (cosine, k=1) both achieve **99.69% accuracy**
5. **Real-time Attendance** — Streamlit web app with live webcam feed, model selection, CSV export

## Results

| Metric | SVM (RBF) | KNN (k=1, cosine) |
|---|---|---|
| Test Accuracy | 99.69% | 99.69% |
| Mean Confidence | 85.04% | 100.00% |
| Training Time | 2.38s | 0.00s |
| Inference per sample | 0.77ms | 0.21ms |

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
# Train SVM
python svm\train.py

# Train KNN
python knn\train.py

# Train both + comparison figure
python train_comparison.py
```

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Select SVM or KNN in the sidebar, click **Start**, and point the webcam at faces.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA (recommended) or CPU
- Webcam
- Windows (for DirectShow capture; adjust camera backend on Linux/macOS)
