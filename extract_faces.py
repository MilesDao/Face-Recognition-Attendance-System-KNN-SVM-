import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIDEO_DIR = Path("Video_")
OUTPUT_DIR = Path("dataset/train_raw")
SAMPLE_RATE = 30
FACE_SIZE = 160
CONFIDENCE_THRESHOLD = 0.95

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

mtcnn = MTCNN(
    image_size=FACE_SIZE,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
)


def parse_person_name(video_path: Path) -> str:
    stem = video_path.stem
    stem = stem.replace(" ", "_")
    return stem


def extract_faces_from_video(video_path: Path):
    person_name = parse_person_name(video_path)
    out_dir = OUTPUT_DIR / person_name

    sample_rates_to_try = [30, 15, 10, 5, 2, 1]
    saved_count = 0

    for sample_rate in sample_rates_to_try:
        # Clear old extracted faces to guarantee clean, noise-free starts
        if out_dir.exists():
            import shutil
            try:
                shutil.rmtree(out_dir)
            except Exception as e:
                logger.warning(f"Could not clear {out_dir}: {e}")
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"  Cannot open {video_path}, skipping")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            boxes, probs = mtcnn.detect(rgb)

            if boxes is not None:
                # Filter and measure all detected faces
                valid_faces = []
                for i, box in enumerate(boxes):
                    if probs is not None and probs[i] < CONFIDENCE_THRESHOLD:
                        continue

                    box = box.astype(int)
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                    w, h = x2 - x1, y2 - y1
                    if w < 20 or h < 20:
                        continue
                    
                    # We save face area to select the largest one
                    area = w * h
                    valid_faces.append((area, x1, y1, x2, y2))

                if valid_faces:
                    # Select only the single largest face (the target student)
                    _, x1, y1, x2, y2 = max(valid_faces, key=lambda x: x[0])

                    face_crop = rgb[y1:y2, x1:x2]
                    face_resized = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
                    face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)

                    fname = f"face_{saved_count:04d}.jpg"
                    cv2.imwrite(str(out_dir / fname), face_bgr)
                    saved_count += 1

            frame_idx += 1

        cap.release()

        # Stop retrying if we reached our target of at least 10 clean images
        if saved_count >= 10:
            logger.info(
                f"  {person_name}: successfully saved {saved_count} clean faces using sample_rate={sample_rate}"
            )
            break
        else:
            logger.info(
                f"  {person_name}: only obtained {saved_count} faces with sample_rate={sample_rate}. Retrying with higher frequency..."
            )
    return saved_count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mov", ".MOV", ".mp4", ".MP4", ".avi", ".AVI"}
    video_files = sorted(
        [f for f in VIDEO_DIR.iterdir() if f.suffix in video_extensions]
    )

    logger.info(f"Found {len(video_files)} videos in {VIDEO_DIR}")

    total_faces = 0
    for vf in video_files:
        n = extract_faces_from_video(vf)
        total_faces += n

    logger.info(f"Done! Extracted {total_faces} faces total from {len(video_files)} videos")

    logger.info("\nSummary per person:")
    for person_dir in sorted(OUTPUT_DIR.iterdir()):
        if person_dir.is_dir():
            count = len(list(person_dir.glob("*.jpg")))
            logger.info(f"  {person_dir.name}: {count} images")


if __name__ == "__main__":
    main()
