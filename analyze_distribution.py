import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("dataset/train_raw")
AUG_DIR = Path("dataset/train_augmented")
FIGURES_DIR = Path("figures")
EMBEDDING_FILE = Path("train_embeddings.npz")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 12,
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "white",
})


def count_images(data_dir: Path) -> dict:
    counts = {}
    if not data_dir.exists():
        return counts
    for person_dir in sorted(data_dir.iterdir()):
        if person_dir.is_dir():
            n = len(list(person_dir.glob("*.jpg")))
            counts[person_dir.name] = n
    return counts


def plot_distribution(counts, title, filename, color="steelblue"):
    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(counts.keys())
    values = list(counts.values())
    bars = ax.bar(range(len(names)), values, color=color, edgecolor="white",
                  linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of images", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"  Saved {FIGURES_DIR / filename.replace('.pdf', '.png')}")


def plot_embedding_pca(embeddings, labels, filename="embedding_pca.pdf"):
    if len(embeddings) < 3:
        logger.warning("Need >= 3 embeddings for PCA")
        return

    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(np.array(embeddings))

    fig, ax = plt.subplots(figsize=(12, 9))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("husl", len(unique_labels))
    label_color_map = dict(zip(unique_labels, palette))

    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=[label_color_map[lbl]], label=lbl.split("_")[0].title(),
            alpha=0.75, s=35, edgecolors="white", linewidth=0.4,
            zorder=3,
        )

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"Principal Component 1 ({var1:.1f}%)", fontsize=12)
    ax.set_ylabel(f"Principal Component 2 ({var2:.1f}%)", fontsize=12)
    ax.set_title("PCA Projection of FaceNet Embeddings (512-d → 2-d)",
                 fontsize=15, fontweight="bold", pad=14)
    ax.legend(loc="best", fontsize=7, markerscale=0.8,
              frameon=True, facecolor="white", edgecolor="#ddd",
              ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, zorder=0)

    # inset: variance explained
    textstr = f"Total variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4",
            facecolor="white", edgecolor="#ddd"))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"  Saved {FIGURES_DIR / filename.replace('.pdf', '.png')}")


def main():
    logger.info("=== Data Distribution Analysis ===\n")

    before = count_images(RAW_DIR)
    after = count_images(AUG_DIR)

    logger.info(f"Before augmentation ({len(before)} classes): {sum(before.values())} total")
    for k, v in sorted(before.items()):
        logger.info(f"  {k}: {v}")

    logger.info("")
    logger.info(f"After augmentation ({len(after)} classes): {sum(after.values())} total")
    for k, v in sorted(after.items()):
        logger.info(f"  {k}: {v}")

    logger.info("\n--- Generating Figure 1: Distribution Before ---")
    plot_distribution(before, "Data Distribution Before Augmentation",
                      "distribution_before.pdf", color="#4361ee")

    logger.info("\n--- Generating Figure 2: Distribution After ---")
    plot_distribution(after, "Data Distribution After Augmentation",
                      "distribution_after.pdf", color="#f72585")

    logger.info("\n--- Generating Figure 3: Embedding PCA ---")
    if EMBEDDING_FILE.exists():
        data = np.load(EMBEDDING_FILE)
        embeddings = data["embeddings"]
        labels = data["labels"]
        logger.info(f"Loaded {len(embeddings)} embeddings from {EMBEDDING_FILE}")
        plot_embedding_pca(embeddings, labels)
    else:
        logger.warning(f"{EMBEDDING_FILE} not found. Skipping PCA plot.")

    logger.info("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
