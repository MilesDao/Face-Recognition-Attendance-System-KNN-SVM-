import os
import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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

    logger.info("Training SVM classifier (RBF kernel, C=10, gamma=scale)...")
    t0 = time.time()
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_train_time = time.time() - t0
    logger.info(f"Training time: {svm_train_time:.2f}s")

    t0 = time.time()
    y_pred = svm.predict(X_test)
    infer_time = (time.time() - t0) / len(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Inference time per sample: {infer_time*1000:.2f}ms")
    logger.info(f"Test accuracy: {accuracy:.4f}")

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    logger.info(f"Classification report:\n{report}")

    proba = svm.predict_proba(X_test)
    mean_conf = proba.max(axis=1).mean()
    logger.info(f"Mean confidence on test set: {mean_conf:.4f}")

    model_path = MODEL_DIR / "svm_face_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"classifier": svm, "label_encoder": le, "type": "svm"}, f)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
