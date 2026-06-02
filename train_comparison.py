import numpy as np
import pickle
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger.info("Loading cached embeddings...")
data = np.load("train_embeddings.npz")
embeddings = data["embeddings"]
label_strings = data["labels"]

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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
logger.info(f"SVM: acc={svm_acc:.4f}, conf={svm_conf:.4f}, train={svm_train:.2f}s, infer={svm_infer*1000:.2f}ms")

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
logger.info(f"Best k={best_k} (CV={best_cv:.4f})")

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
logger.info(f"KNN: acc={knn_acc:.4f}, conf={knn_conf:.4f}, train={knn_train:.2f}s, infer={knn_infer*1000:.2f}ms")

# ─── Save models ───
with open("svm/svm_face_classifier.pkl", "wb") as f:
    pickle.dump({"classifier": svm, "label_encoder": le, "type": "svm"}, f)
with open("knn/knn_face_classifier.pkl", "wb") as f:
    pickle.dump({"classifier": knn, "label_encoder": le, "type": "knn", "best_k": best_k}, f)
logger.info("Models saved")

# ─── Beautiful comparison figure ───
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "white",
})
CB_blue = "#4361ee"
CB_orange = "#f72585"

fig = plt.figure(figsize=(18, 7))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.3, 1], hspace=0.35, wspace=0.3)

# ─── Panel 1: Accuracy & Confidence ───
ax1 = fig.add_subplot(gs[:, 0])
metrics = ["Accuracy", "Confidence"]
x = np.arange(len(metrics))
w = 0.30
bars1 = ax1.bar(x - w/2, [svm_acc*100, svm_conf*100], w, color=CB_blue,
                edgecolor="white", linewidth=1.2, label="SVM (RBF)", zorder=3)
bars2 = ax1.bar(x + w/2, [knn_acc*100, knn_conf*100], w, color=CB_orange,
                edgecolor="white", linewidth=1.2, label=f"KNN (k={best_k})", zorder=3)

for bars in [bars1, bars2]:
    for bar in bars:
        val = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333")

ax1.set_ylim(0, 112)
ax1.set_ylabel("Percentage (%)", fontsize=12)
ax1.set_title("Accuracy & Confidence", fontsize=14, fontweight="bold", pad=12)
ax1.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=12)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ─── Panel 2: Training & Inference Time ───
ax2 = fig.add_subplot(gs[:, 1])
timing = ["Training\n(seconds)", "Inference\n(ms/sample)"]
svm_t = [svm_train, svm_infer*1000]
knn_t = [knn_train, knn_infer*1000]
x2 = np.arange(len(timing))
bars1 = ax2.bar(x2 - w/2, svm_t, w, color=CB_blue,
                edgecolor="white", linewidth=1.2, label="SVM (RBF)", zorder=3)
bars2 = ax2.bar(x2 + w/2, knn_t, w, color=CB_orange,
                edgecolor="white", linewidth=1.2, label=f"KNN (k={best_k})", zorder=3)

ax2.set_yscale("log")
for bars, vals in [(bars1, svm_t), (bars2, knn_t)]:
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val * 1.2, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333")

ax2.set_ylabel("Time (log scale)", fontsize=12)
ax2.set_title("Training & Inference Time", fontsize=14, fontweight="bold", pad=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(timing, fontsize=11)
ax2.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=10)
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ─── Panel 3: KNN k-value search ───
ax3 = fig.add_subplot(gs[:, 2])
ks = list(knn_scores.keys())
scores = [knn_scores[k]*100 for k in ks]
ax3.plot(ks, scores, "o-", color=CB_orange, linewidth=2.5, markersize=10,
         zorder=3, markerfacecolor="white", markeredgewidth=2)
ax3.axvline(x=best_k, color="#2b2d42", linestyle="--", linewidth=1.5, alpha=0.7,
            label=f"Best k = {best_k}")
ax3.scatter([best_k], [knn_scores[best_k]*100], color=CB_orange, s=150, zorder=4,
            edgecolor="#2b2d42", linewidth=2)
ax3.set_xlabel("k (number of neighbors)", fontsize=12)
ax3.set_ylabel("5-fold CV Accuracy (%)", fontsize=12)
ax3.set_title("KNN Parameter Search", fontsize=14, fontweight="bold", pad=12)
ax3.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=10)
ax3.set_xticks(ks)
ax3.grid(alpha=0.3, zorder=0)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

ax3.annotate(f"Best k = {best_k}\nCV = {knn_scores[best_k]*100:.2f}%",
             xy=(best_k, knn_scores[best_k]*100),
             xytext=(best_k + 1.5, knn_scores[best_k]*100 - 0.5),
             fontsize=10, fontweight="bold", color="#2b2d42",
             arrowprops=dict(arrowstyle="->", color="#2b2d42", lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ddd"))

# ─── Panel 4: Summary table ───
ax4 = fig.add_subplot(gs[:, 3])
ax4.axis("off")

table_data = [
    ["Metric", "SVM (RBF)", f"KNN (k={best_k})"],
    ["Test Accuracy", f"{svm_acc*100:.2f}%", f"{knn_acc*100:.2f}%"],
    ["Mean Confidence", f"{svm_conf*100:.2f}%", f"{knn_conf*100:.2f}%"],
    ["Training Time", f"{svm_train:.2f}s", f"{knn_train:.2f}s"],
    ["Inference / sample", f"{svm_infer*1000:.4f}ms", f"{knn_infer*1000:.4f}ms"],
    ["Parameters", "C=10, RBF", f"k={best_k}, cosine"],
]

table = ax4.table(cellText=table_data, loc="center", cellLoc="center",
                  colWidths=[0.2, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for j in range(3):
    cell = table[0, j]
    cell.set_facecolor("#2b2d42")
    cell.set_text_props(color="white", fontweight="bold", fontsize=11)
    cell.set_edgecolor("#2b2d42")

colors_row = ["white", "#f0f1f3"]
for i in range(1, len(table_data)):
    bg = colors_row[i % 2]
    for j in range(3):
        cell = table[i, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor("#ddd")
        cell.set_text_props(fontsize=10)

table[1, 1].set_facecolor("#d4edda")
table[1, 2].set_facecolor("#d4edda")
table[4, 2].set_facecolor("#d4edda")

ax4.set_title("Comparison Summary", fontsize=14, fontweight="bold", pad=12)

fig.suptitle("SVM vs KNN - Face Recognition Performance Comparison",
             fontsize=18, fontweight="bold", y=1.02, color="#2b2d42")

plt.savefig("figures/svm_vs_knn_comparison.pdf", bbox_inches="tight", dpi=200,
            facecolor="white", edgecolor="none")
plt.savefig("figures/svm_vs_knn_comparison.png", bbox_inches="tight", dpi=200,
            facecolor="white", edgecolor="none")
logger.info("Figure saved to figures/")

# ─── Print summary ───
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
