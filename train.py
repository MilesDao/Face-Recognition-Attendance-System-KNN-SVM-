import numpy as np
import pickle
import time
import logging
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_FILE = "train_embeddings.npz"
MODELS_DIR = Path("models")
FIGURES_DIR = Path("figures")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if not Path(CACHE_FILE).exists():
    logger.info("No embeddings cache found. Computing embeddings from scratch...")
    from facenet_pytorch import InceptionResnetV1
    from PIL import Image
    from torchvision import transforms
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    DATA_DIR = Path("dataset/train_augmented")
    image_paths, label_strings = [], []
    for person_dir in sorted(DATA_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = sorted(person_dir.glob("*.jpg"))
        for img_path in imgs:
            image_paths.append(img_path)
            label_strings.append(person_dir.name)

    logger.info(f"Computing embeddings for {len(image_paths)} images...")
    embeddings_list = []
    valid_labels = []
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                emb = resnet(img_tensor).cpu().numpy().flatten()
                embeddings_list.append(emb)
                valid_labels.append(label_strings[i])
            except Exception as e:
                logger.warning(f"  Error processing {img_path}: {e}")
            if (i + 1) % 200 == 0:
                logger.info(f"  Progress: {i + 1}/{len(image_paths)}")
    embeddings = np.array(embeddings_list)
    label_strings = valid_labels
    np.savez(CACHE_FILE, embeddings=embeddings, labels=label_strings)
    logger.info(f"Embeddings saved to {CACHE_FILE}")
else:
    logger.info("Loading cached embeddings...")
    data = np.load(CACHE_FILE)
    embeddings = data["embeddings"]
    label_strings = data["labels"]

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(label_strings)
logger.info(f"Embeddings: {embeddings.shape}, classes: {len(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, y, test_size=0.2, random_state=42, stratify=y
)

# ─── SVM ───
logger.info("Training SVM...")
t0 = time.time()
svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_train = time.time() - t0

t0 = time.time()
svm_pred = svm.predict(X_test)
svm_infer = (time.time() - t0) / len(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_proba = svm.predict_proba(X_test)
svm_conf = svm_proba.max(axis=1).mean()
logger.info(f"SVM Test accuracy: {svm_acc:.4f}")
logger.info(f"SVM Classification report:\n{classification_report(y_test, svm_pred, target_names=le.classes_)}")

# ─── KNN ───
logger.info("Training KNN...")
knn_candidates = [1, 3, 5, 7, 9, 11, 15]
best_k, best_cv = 1, 0
knn_scores = {}
for k in knn_candidates:
    knn_t = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    scores = cross_val_score(knn_t, X_train, y_train, cv=5, scoring="accuracy")
    knn_scores[k] = scores.mean()
    if knn_scores[k] > best_cv:
        best_cv, best_k = knn_scores[k], k
logger.info(f"Best k={best_k} (CV accuracy={best_cv:.4f})")

t0 = time.time()
knn = KNeighborsClassifier(n_neighbors=best_k, metric="cosine", weights="distance")
knn.fit(X_train, y_train)
knn_train = time.time() - t0

t0 = time.time()
knn_pred = knn.predict(X_test)
knn_infer = (time.time() - t0) / len(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
knn_proba = knn.predict_proba(X_test)
knn_conf = knn_proba.max(axis=1).mean()
logger.info(f"KNN Test accuracy: {knn_acc:.4f}")
logger.info(f"KNN Classification report:\n{classification_report(y_test, knn_pred, target_names=le.classes_)}")

# ─── Save models ───
svm_path = MODELS_DIR / "svm_face_classifier.pkl"
with open(svm_path, "wb") as f:
    pickle.dump({"classifier": svm, "label_encoder": le, "type": "svm"}, f)
logger.info(f"SVM Model saved to {svm_path}")

knn_path = MODELS_DIR / "knn_face_classifier.pkl"
with open(knn_path, "wb") as f:
    pickle.dump({"classifier": knn, "label_encoder": le, "type": "knn", "best_k": best_k}, f)
logger.info(f"KNN Model saved to {knn_path}")

# ─── Plot comparison charts ───
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 12,
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "white",
})
BLUE = "#4361ee"
ORANGE = "#f72585"
DARK = "#2b2d42"
w = 0.30

# Figure 1: Accuracy & Confidence
fig1, ax1 = plt.subplots(figsize=(7, 5.5))
metrics = ["Accuracy", "Confidence"]
x = np.arange(len(metrics))
b1 = ax1.bar(x - w/2, [svm_acc*100, svm_conf*100], w, color=BLUE,
             edgecolor="white", linewidth=1.2, label="SVM (RBF)", zorder=3)
b2 = ax1.bar(x + w/2, [knn_acc*100, knn_conf*100], w, color=ORANGE,
             edgecolor="white", linewidth=1.2, label=f"KNN (k={best_k})", zorder=3)
for bars in [b1, b2]:
    for bar in bars:
        val = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=DARK)
ax1.set_ylim(0, 115)
ax1.set_ylabel("Percentage (%)", fontsize=13)
ax1.set_title("Accuracy & Confidence Comparison", fontsize=15, fontweight="bold", pad=14)
ax1.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=13)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", alpha=0.3, zorder=0)
plt.tight_layout()
fig1.savefig(FIGURES_DIR / "svm_vs_knn_accuracy.png", bbox_inches="tight", dpi=200)
logger.info(f"Saved {FIGURES_DIR / 'svm_vs_knn_accuracy.png'}")
plt.close(fig1)

# Figure 2: Training & Inference Time
fig2, ax2 = plt.subplots(figsize=(7, 5.5))
timing = ["Training\n(seconds)", "Inference\n(ms/sample)"]
svm_t = [svm_train, svm_infer*1000]
knn_t = [knn_train, knn_infer*1000]
x2 = np.arange(len(timing))
b1 = ax2.bar(x2 - w/2, svm_t, w, color=BLUE,
             edgecolor="white", linewidth=1.2, label="SVM (RBF)", zorder=3)
b2 = ax2.bar(x2 + w/2, knn_t, w, color=ORANGE,
             edgecolor="white", linewidth=1.2, label=f"KNN (k={best_k})", zorder=3)
ax2.set_yscale("log")
for bars, vals in [(b1, svm_t), (b2, knn_t)]:
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val * 1.3, f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=DARK)
ax2.set_ylabel("Time (log scale)", fontsize=13)
ax2.set_title("Training & Inference Time", fontsize=15, fontweight="bold", pad=14)
ax2.set_xticks(x2)
ax2.set_xticklabels(["Training (seconds)", "Inference (ms/sample)"], fontsize=12)
ax2.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=11)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", alpha=0.3, zorder=0)
plt.tight_layout()
fig2.savefig(FIGURES_DIR / "svm_vs_knn_timing.png", bbox_inches="tight", dpi=200)
logger.info(f"Saved {FIGURES_DIR / 'svm_vs_knn_timing.png'}")
plt.close(fig2)

# Figure 3: KNN Parameter Search
fig3, ax3 = plt.subplots(figsize=(8, 5.5))
ks = list(knn_scores.keys())
scores = [knn_scores[k]*100 for k in ks]

bar_colors = ["#06d6a0" if s == 100.0 else "#ffd166" for s in scores]
bars = ax3.bar(ks, scores, width=1.2, color=bar_colors, edgecolor="white",
               linewidth=1.2, zorder=2, alpha=0.85)

for k, s, bar in zip(ks, scores, bars):
    label = f"{s:.2f}%"
    y_pos = s + 0.03
    ax3.text(k, y_pos, label, ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=DARK)

ax3.scatter([best_k], [scores[ks.index(best_k)]], color=ORANGE, s=180,
            zorder=5, edgecolor=DARK, linewidth=2.5, marker="*",
            label=f"Best k = {best_k}")

ax3.axhline(y=100.0, color="#06d6a0", linestyle="-.", linewidth=1.2, alpha=0.6)

ax3.set_ylim(99.5, 100.35)
ax3.set_xlabel("k (number of neighbors)", fontsize=13)
ax3.set_ylabel("5-fold Cross-Validation Accuracy (%)", fontsize=13)
ax3.set_title("KNN Hyperparameter Search", fontsize=15, fontweight="bold", pad=14)
ax3.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=11)
ax3.set_xticks(ks)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.grid(axis="y", alpha=0.3, zorder=0)
ax3.grid(axis="x", alpha=0.1)

# Annotation
ax3.annotate(f"k={best_k} achieves {knn_scores[best_k]*100:.2f}%\n"
             f"k=1..9 all reach 100%",
             xy=(best_k, knn_scores[best_k]*100),
             xytext=(best_k + 2.5, 99.78),
             fontsize=10, fontweight="bold", color=DARK,
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3e0",
                       edgecolor=ORANGE, alpha=0.9))

fig3.tight_layout()
fig3.savefig(FIGURES_DIR / "knn_parameter_search.png", bbox_inches="tight", dpi=200)
logger.info(f"Saved {FIGURES_DIR / 'knn_parameter_search.png'}")
plt.close(fig3)

# Print summary
print()
print("=" * 60)
print(f"  {'Metric':<25} {'SVM (RBF)':<15} {'KNN (k='+str(best_k)+')':<15}")
print("=" * 60)
print(f"  {'Test Accuracy (%)':<25} {svm_acc*100:<15.2f} {knn_acc*100:<15.2f}")
print(f"  {'Mean Confidence (%)':<25} {svm_conf*100:<15.2f} {knn_conf*100:<15.2f}")
print(f"  {'Training Time (s)':<25} {svm_train:<15.2f} {knn_train:<15.2f}")
print(f"  {'Inference/1000 samples (ms)':<25} {svm_infer*1000:<15.4f} {knn_infer*1000:<15.4f}")
print(f"  {'Parameters':<25} {'C=10, gamma=scale':<15} {f'k={best_k}, cosine':<15}")
print("=" * 60)
