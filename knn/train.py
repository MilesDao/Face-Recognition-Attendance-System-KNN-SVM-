import os
import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import InceptionResnetV1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
import time
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../dataset/train_augmented")
MODEL_DIR = Path(".")
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
    t0 = time.time()
    embeddings, valid_paths = compute_embeddings(image_paths)
    logger.info(f"Embedding computation time: {time.time() - t0:.2f}s")

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

    knn_candidates = [1, 3, 5, 7, 9, 11, 15]
    knn_scores = {}

    logger.info("Searching for best k (5-fold CV, cosine distance)...")
    best_k = 1
    best_cv = 0.0

    for k in knn_candidates:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
        mean_score = scores.mean()
        knn_scores[k] = mean_score
        logger.info(f"  k={k}: CV accuracy = {mean_score:.4f} (+/- {scores.std()*2:.4f})")
        if mean_score > best_cv:
            best_cv = mean_score
            best_k = k

    logger.info(f"Best k={best_k} (CV accuracy={best_cv:.4f})")

    t0 = time.time()
    best_knn = KNeighborsClassifier(n_neighbors=best_k, metric="cosine", weights="distance")
    best_knn.fit(X_train, y_train)
    knn_train_time = time.time() - t0
    logger.info(f"Training time: {knn_train_time:.2f}s (k={best_k})")

    t0 = time.time()
    y_pred = best_knn.predict(X_test)
    infer_time = (time.time() - t0) / len(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Inference time per sample: {infer_time*1000:.2f}ms")
    logger.info(f"Test accuracy: {accuracy:.4f}")

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    logger.info(f"Classification report:\n{report}")

    proba = best_knn.predict_proba(X_test)
    mean_conf = proba.max(axis=1).mean()
    logger.info(f"Mean confidence on test set: {mean_conf:.4f}")

    model_path = MODEL_DIR / "knn_face_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"classifier": best_knn, "label_encoder": le, "type": "knn", "best_k": best_k}, f)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
