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
EMBEDDING_FILE = Path("vid2_embeddings.npz")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


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
    bars = ax.bar(range(len(names)), values, color=color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of images")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {FIGURES_DIR / filename}")


def plot_embedding_pca(embeddings, labels, filename="embedding_pca.pdf"):
    if len(embeddings) < 3:
        logger.warning("Need >= 3 embeddings for PCA")
        return

    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(np.array(embeddings))

    fig, ax = plt.subplots(figsize=(12, 8))
    unique_labels = sorted(set(labels))
    colors = sns.color_palette("husl", len(unique_labels))
    label_color_map = dict(zip(unique_labels, colors))

    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=[label_color_map[lbl]],
            label=lbl, alpha=0.7, s=30, edgecolors="black", linewidth=0.3,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax.set_title("PCA Projection of Face Embedding Vectors")
    ax.legend(loc="best", fontsize=7, markerscale=0.8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {FIGURES_DIR / filename} ({pca.explained_variance_ratio_.sum():.2%} variance explained)")


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
                      "distribution_before.pdf", color="steelblue")

    logger.info("\n--- Generating Figure 2: Distribution After ---")
    plot_distribution(after, "Data Distribution After Augmentation",
                      "distribution_after.pdf", color="coral")

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
