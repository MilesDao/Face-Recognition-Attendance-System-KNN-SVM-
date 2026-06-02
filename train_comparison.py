import os
import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("dataset/train_augmented")
MODEL_SVM_DIR = Path("svm")
MODEL_KNN_DIR = Path("knn")
FIGURES_DIR = Path("figures")
EMBEDDING_DIM = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
logger.info("Loaded FaceNet (InceptionResnetV1) pretrained on VGGFace2")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def compute_embeddings(image_paths):
    embeddings = []
    valid_paths = []

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                emb = resnet(img_tensor).cpu().numpy().flatten()
                embeddings.append(emb)
                valid_paths.append(img_path)
            except Exception as e:
                logger.warning(f"  Error processing {img_path}: {e}")

            if (i + 1) % 100 == 0:
                logger.info(f"  Embeddings computed: {i + 1}/{len(image_paths)}")

    return np.array(embeddings), valid_paths


def main():
    MODEL_SVM_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_KNN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = []
    labels = []

    person_dirs = sorted(DATA_DIR.iterdir())
    logger.info(f"Loading images from {len(person_dirs)} classes...")

    for person_dir in person_dirs:
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        imgs = sorted(person_dir.glob("*.jpg"))
        for img_path in imgs:
            image_paths.append(img_path)
            labels.append(person_name)

    logger.info(f"Total images: {len(image_paths)}, classes: {len(set(labels))}")

    logger.info("Computing FaceNet embeddings...")
    embeddings, valid_paths = compute_embeddings(image_paths)
    valid_labels = [labels[image_paths.index(p)] for p in valid_paths]

    le = LabelEncoder()
    y = le.fit_transform(valid_labels)

    logger.info(f"Embedding shape: {embeddings.shape}")
    logger.info(f"Number of classes: {len(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # ─── SVM ───────────────────────────────────────────────────────────────
    logger.info("\n=== Training SVM (RBF kernel, C=10, gamma=scale) ===")
    t0 = time.time()
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_train_time = time.time() - t0

    t0 = time.time()
    svm_pred = svm.predict(X_test)
    svm_infer_time = (time.time() - t0) / len(X_test)

    svm_acc = accuracy_score(y_test, svm_pred)
    svm_proba = svm.predict_proba(X_test)
    svm_conf = svm_proba.max(axis=1).mean()

    logger.info(f"  Train time: {svm_train_time:.2f}s")
    logger.info(f"  Inference time per sample: {svm_infer_time*1000:.2f}ms")
    logger.info(f"  Test accuracy: {svm_acc:.4f}")
    logger.info(f"  Mean confidence: {svm_conf:.4f}")
    logger.info(f"  Classification report:\n{classification_report(y_test, svm_pred, target_names=le.classes_)}")

    # ─── KNN ───────────────────────────────────────────────────────────────
    logger.info("\n=== Training KNN ===")

    knn_candidates = [1, 3, 5, 7, 9, 11, 15]
    knn_scores = {}
    knn_results = []

    logger.info("  Searching for best k (5-fold CV)...")
    best_k = 1
    best_cv = 0.0

    for k in knn_candidates:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
        mean_score = scores.mean()
        knn_scores[k] = mean_score
        logger.info(f"    k={k}: CV accuracy = {mean_score:.4f} (+/- {scores.std()*2:.4f})")
        if mean_score > best_cv:
            best_cv = mean_score
            best_k = k

    logger.info(f"  Best k={best_k} (CV accuracy={best_cv:.4f})")

    t0 = time.time()
    best_knn = KNeighborsClassifier(n_neighbors=best_k, metric="cosine", weights="distance")
    best_knn.fit(X_train, y_train)
    knn_train_time = time.time() - t0

    t0 = time.time()
    knn_pred = best_knn.predict(X_test)
    knn_infer_time = (time.time() - t0) / len(X_test)

    knn_proba = best_knn.predict_proba(X_test)
    knn_conf = knn_proba.max(axis=1).mean()
    knn_acc = accuracy_score(y_test, knn_pred)

    logger.info(f"  Train time: {knn_train_time:.2f}s")
    logger.info(f"  Inference time per sample: {knn_infer_time*1000:.2f}ms")
    logger.info(f"  Test accuracy: {knn_acc:.4f}")
    logger.info(f"  Mean confidence: {knn_conf:.4f}")
    logger.info(f"  Classification report:\n{classification_report(y_test, knn_pred, target_names=le.classes_)}")

    # ─── Save models ──────────────────────────────────────────────────────
    svm_path = MODEL_SVM_DIR / "svm_face_classifier.pkl"
    knn_path = MODEL_KNN_DIR / "knn_face_classifier.pkl"

    with open(svm_path, "wb") as f:
        pickle.dump({"classifier": svm, "label_encoder": le, "type": "svm"}, f)
    logger.info(f"SVM model saved to {svm_path}")

    with open(knn_path, "wb") as f:
        pickle.dump({"classifier": best_knn, "label_encoder": le, "type": "knn", "best_k": best_k}, f)
    logger.info(f"KNN model saved to {knn_path}")

    # ─── Comparison figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 1. Accuracy + Confidence bar chart
    ax = axes[0]
    metrics = ["Accuracy", "Mean\nConfidence"]
    svm_vals = [svm_acc * 100, svm_conf * 100]
    knn_vals = [knn_acc * 100, knn_conf * 100]

    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w / 2, svm_vals, w, label="SVM (RBF)", color="#2196F3")
    bars2 = ax.bar(x + w / 2, knn_vals, w, label=f"KNN (k={best_k})", color="#FF9800")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 105)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Accuracy & Confidence")
    ax.legend(fontsize=9)

    # 2. Timing bar chart
    ax = axes[1]
    timing_labels = ["Train (s)", "Inference\n(ms/sample)"]
    svm_timing = [svm_train_time, svm_infer_time * 1000]
    knn_timing = [knn_train_time, knn_infer_time * 1000]

    x = np.arange(len(timing_labels))
    bars1 = ax.bar(x - w / 2, svm_timing, w, label="SVM (RBF)", color="#2196F3")
    bars2 = ax.bar(x + w / 2, knn_timing, w, label=f"KNN (k={best_k})", color="#FF9800")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(svm_timing) * 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(knn_timing) * 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Time")
    ax.set_title("Training & Inference Time")
    ax.legend(fontsize=9)

    # 3. KNN k-value search
    ax = axes[2]
    ks = list(knn_scores.keys())
    scores = [knn_scores[k] * 100 for k in ks]
    ax.plot(ks, scores, "o-", color="#FF9800", linewidth=2, markersize=6)
    ax.axvline(x=best_k, color="red", linestyle="--", alpha=0.5, label=f"Best k={best_k}")
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("5-fold CV Accuracy (%)")
    ax.set_title("KNN Parameter Search")
    ax.legend(fontsize=9)
    ax.set_xticks(ks)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "svm_vs_knn_comparison.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    logger.info(f"Comparison figure saved to {fig_path}")

    # ─── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  COMPARISON SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  {'Metric':<25} {'SVM (RBF)':<15} {'KNN (k='+str(best_k)+')':<15}")
    logger.info("-" * 55)
    logger.info(f"  {'Test Accuracy (%)':<25} {svm_acc*100:<15.2f} {knn_acc*100:<15.2f}")
    logger.info(f"  {'Mean Confidence (%)':<25} {svm_conf*100:<15.2f} {knn_conf*100:<15.2f}")
    logger.info(f"  {'Training Time (s)':<25} {svm_train_time:<15.2f} {knn_train_time:<15.2f}")
    logger.info(f"  {'Inference/1000 samples (ms)':<25} {svm_infer_time*1000:<15.4f} {knn_infer_time*1000:<15.4f}")
    logger.info(f"  {'Parameters':<25} {'C=10, gamma=scale':<15} {f'k={best_k}, cosine':<15}")
    logger.info("=" * 55)
    
    # Save the embeddings to train_embeddings.npz for analyze_distribution.py to use!
    np.savez("train_embeddings.npz", embeddings=embeddings, labels=valid_labels)
    logger.info("Saved embeddings to train_embeddings.npz")


if __name__ == "__main__":
    main()
