import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = Path("dataset/train_raw")
OUTPUT_DIR = Path("dataset/train_augmented")
FIGURES_DIR = Path("figures")
AUGMENTATIONS_PER_IMAGE = 4
TARGET_PER_CLASS = 60

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


def random_brightness_contrast(img):
    alpha = 1.0 + random.uniform(-0.2, 0.2)
    beta = random.randint(-30, 30)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def random_gaussian_blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)


def random_rotation(img):
    h, w = img.shape[:2]
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def augment_image(img):
    augs = {
        "flip": cv2.flip(img, 1),
        "brightness": random_brightness_contrast(img),
        "blur": random_gaussian_blur(img),
        "rotate": random_rotation(img),
    }
    return augs


def count_images_per_class(data_dir: Path) -> dict:
    counts = {}
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
    logger.info(f"  Saved {filename}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    before_counts = count_images_per_class(INPUT_DIR)
    logger.info(f"Distribution BEFORE augmentation ({len(before_counts)} classes):")
    for name, count in sorted(before_counts.items()):
        logger.info(f"  {name}: {count}")

    plot_distribution(
        before_counts,
        "Data Distribution Before Augmentation",
        "distribution_before.pdf",
        color="steelblue",
    )

    for person_dir in sorted(INPUT_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        out_person = OUTPUT_DIR / person_dir.name
        out_person.mkdir(parents=True, exist_ok=True)

        images = sorted(person_dir.glob("*.jpg"))
        if not images:
            continue

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            img_center = cv2.resize(img, (160, 160))

            cv2.imwrite(str(out_person / img_path.name), img_center)

            augs = augment_image(img_center)
            for aug_name, aug_img in augs.items():
                stem = img_path.stem
                aug_filename = f"{stem}_{aug_name}.jpg"
                cv2.imwrite(str(out_person / aug_filename), aug_img)

        n_before = len(images)
        n_after = len(list(out_person.glob("*.jpg")))
        logger.info(f"  {person_dir.name}: {n_before} -> {n_after} images")

    after_counts = count_images_per_class(OUTPUT_DIR)
    logger.info(f"\nDistribution AFTER augmentation ({len(after_counts)} classes):")
    for name, count in sorted(after_counts.items()):
        logger.info(f"  {name}: {count}")

    plot_distribution(
        after_counts,
        "Data Distribution After Augmentation",
        "distribution_after.pdf",
        color="coral",
    )

    logger.info(f"\nAugmentation complete!")
    logger.info(f"  Before: {sum(before_counts.values())} total images")
    logger.info(f"  After:  {sum(after_counts.values())} total images")


if __name__ == "__main__":
    main()
