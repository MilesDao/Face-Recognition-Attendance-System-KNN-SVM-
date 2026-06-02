import streamlit as st
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

st.set_page_config(page_title="Face Attendance", layout="wide")

MODEL_SVM_PATH = Path("svm/svm_face_classifier.pkl")
MODEL_KNN_PATH = Path("knn/knn_face_classifier.pkl")
PROCESS_EVERY_N = 8
REFRESH_MS = 200

import torch
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEVICE_STR = f"{'GPU' if torch.cuda.is_available() else 'CPU'} ({_DEVICE})"


class CameraProcessor:
    def __init__(self):
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        
        self._raw_frame = None
        self._raw_frame_lock = threading.Lock()
        
        self._jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        
        self._pending = []
        self._pending_lock = threading.Lock()
        
        self._error = None
        self._ready = False
        self.threshold = 0.5
        self.model_type = "svm"
        self._heavy_initialized = False
        
        # Start preloading heavy models in a background thread immediately!
        self._preload_thread = threading.Thread(target=self._preload_heavy, daemon=True)
        self._preload_thread.start()

    def _preload_heavy(self):
        try:
            self._init_heavy()
        except Exception as e:
            self._error = f"Preload error: {str(e)}"

    def _init_heavy(self):
        if self._heavy_initialized:
            return
        import cv2
        import numpy as np
        from facenet_pytorch import MTCNN, InceptionResnetV1
        from PIL import Image
        from torchvision import transforms
        import torch
        
        # Limit CPU threads to prevent scheduling thrashing on multi-core CPUs
        if not torch.cuda.is_available():
            torch.set_num_threads(4)
            
        self.cv2 = cv2
        self.np = np
        self.Image = Image
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Highly optimized MTCNN parameters for blazing fast CPU/GPU inference
        self._mtcnn = MTCNN(image_size=160, margin=20, min_face_size=80,
                            thresholds=[0.8, 0.8, 0.8], factor=0.8,
                            post_process=True, device=_DEVICE)
        self._resnet = InceptionResnetV1(pretrained="vggface2").eval().to(_DEVICE)
        self._heavy_initialized = True

    def _capture_loop(self):
        import time
        cv2 = self.cv2
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self._error = "Cannot open webcam"
                self._running = False
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                with self._raw_frame_lock:
                    self._raw_frame = frame.copy()
        except Exception as e:
            self._error = f"Capture error: {str(e)}"
        finally:
            if cap is not None:
                cap.release()

    def _processing_loop(self):
        import pickle
        import time
        try:
            cv2 = self.cv2
            np = self.np

            model_path = MODEL_KNN_PATH if self.model_type == "knn" else MODEL_SVM_PATH
            with open(model_path, "rb") as f:
                d = pickle.load(f)
            clf = d.get("classifier") or d.get("svm") or d.get("knn")
            le = d["label_encoder"]

            self._ready = True
            fc = 0

            while self._running:
                frame = None
                with self._raw_frame_lock:
                    if self._raw_frame is not None:
                        frame = self._raw_frame.copy()

                if frame is None:
                    time.sleep(0.01)
                    continue

                fc += 1
                do_detect = (fc % PROCESS_EVERY_N == 0)

                if do_detect:
                    h, w = frame.shape[:2]
                    scale_x = w / 320.0
                    scale_y = h / 240.0

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    small_detect = cv2.resize(rgb, (320, 240))
                    try:
                        boxes, probs = self._mtcnn.detect(small_detect)
                    except Exception:
                        boxes = None

                    self._last_embeddings = []
                    if boxes is not None:
                        for i, b in enumerate(boxes):
                            if probs is not None and probs[i] < 0.8:
                                continue
                            x1 = int(max(0.0, b[0] * scale_x))
                            y1 = int(max(0.0, b[1] * scale_y))
                            x2 = int(min(w, b[2] * scale_x))
                            y2 = int(min(h, b[3] * scale_y))
                            if x2 - x1 < 30 or y2 - y1 < 30:
                                continue
                            try:
                                face_crop = cv2.resize(frame[y1:y2, x1:x2], (160, 160))
                            except cv2.error:
                                continue
                            try:
                                rgb2 = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                pil = self.Image.fromarray(rgb2)
                                t = self._transform(pil).unsqueeze(0).to(_DEVICE)
                                with torch.no_grad():
                                    emb = self._resnet(t).cpu().numpy().flatten()
                            except Exception:
                                continue
                            if emb is None:
                                continue
                            ps = clf.predict_proba([emb])[0]
                            pi = np.argmax(ps)
                            conf = ps[pi]
                            name = le.classes_[pi]
                            self._last_embeddings.append((x1, y1, x2, y2, name, conf))

                if hasattr(self, "_last_embeddings") and self._last_embeddings:
                    for x1, y1, x2, y2, name, conf in self._last_embeddings:
                        if conf >= self.threshold:
                            color = (0, 255, 0)
                            label = f"{name} ({conf:.0%})"
                            with self._pending_lock:
                                if name not in self._pending:
                                    self._pending.append(name)
                        else:
                            color = (0, 255, 255)
                            label = f"{name}? ({conf:.0%})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        tw, th2 = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - th2 - 8), (x1 + tw + 8, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 4, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                small = cv2.resize(frame, (320, 240))
                _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                with self._frame_lock:
                    self._jpeg = jpg.tobytes()

                time.sleep(0.01)

        except Exception as e:
            self._error = str(e)
            import traceback
            self._error += "\n" + traceback.format_exc()
        finally:
            self._ready = False

    def start(self, thresh=0.5, model_type="svm"):
        self.threshold = thresh
        self.model_type = model_type
        if self._running:
            return
        if hasattr(self, "_preload_thread") and self._preload_thread.is_alive():
            self._preload_thread.join()
        self._init_heavy()
        self._running = True
        self._error = None
        self._ready = False
        self._jpeg = None
        self._raw_frame = None
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._capture_thread.start()
        self._process_thread.start()

    def stop(self):
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=3)
            self._capture_thread = None
        if self._process_thread is not None:
            self._process_thread.join(timeout=3)
            self._process_thread = None

    @property
    def is_running(self):
        return self._running

    @property
    def is_ready(self):
        return self._ready

    @property
    def error(self):
        return self._error

    def get_jpeg(self):
        with self._frame_lock:
            return self._jpeg

    def pop_pending(self):
        with self._pending_lock:
            items = list(self._pending)
            self._pending.clear()
            return items


@st.cache_resource
def get_cam():
    return CameraProcessor()


@st.cache_data(ttl=60)
def get_all_names():
    return sorted(set(
        p.stem for p in Path("dataset/train_augmented").iterdir() if p.is_dir()
    ))


def split_name_id(folder_name):
    if "_" in folder_name:
        parts = folder_name.rsplit("_", 1)
        return parts[0].replace("_", " ").title(), parts[1]
    
    # Dynamically split letters (name) and numbers (student ID) if no underscore exists
    import re
    match = re.match(r"^([a-zA-Z_]+)(\d+)$", folder_name)
    if match:
        name_part = match.group(1).replace("_", " ").title()
        id_part = match.group(2)
        return name_part, id_part
        
    return folder_name, "—"


def make_attendance_html(log, all_names):
    attended = {r["Name"] for r in log}
    rows = ["<tr><th>Status</th><th>Name</th><th>StudentID</th><th>Time</th></tr>"]
    for name in all_names:
        status = "<span style='color:green'>Present</span>" if name in attended else "<span style='color:gray'>Absent</span>"
        time_str = next((r["Time"] for r in log if r["Name"] == name), "—")
        student_name, student_id = split_name_id(name)
        rows.append(f"<tr><td>{status}</td><td>{student_name}</td><td>{student_id}</td><td>{time_str}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def main():
    for k, v in [
        ("attended", {}),
        ("log", []),
        ("camera_running", False),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    cam = get_cam()

    sidebar = st.sidebar
    sidebar.title("Attendance")
    
    # Dynamic real-time model loading status in sidebar
    if hasattr(cam, "_heavy_initialized") and cam._heavy_initialized:
        status_str = f"🟢 Ready ({_DEVICE_STR})"
    else:
        status_str = f"🟡 Loading models in background... ({_DEVICE_STR})"
    sidebar.markdown(f"**Device status:** {status_str}")

    model_type = sidebar.radio("Classifier", ["SVM (RBF)", "KNN"], index=0,
                               disabled=st.session_state.camera_running,
                               horizontal=True)
    model_key = "knn" if model_type.startswith("KNN") else "svm"
    cam.model_type = model_key

    c1, c2 = sidebar.columns(2)
    start_btn = c1.button("Start", width="stretch", disabled=st.session_state.camera_running)
    stop_btn = c2.button("Stop", width="stretch", disabled=not st.session_state.camera_running)
    threshold = sidebar.slider("Confidence", 0.0, 1.0, 0.5, 0.05)
    cam.threshold = threshold

    if sidebar.button("Reset", width="stretch"):
        cam.stop()
        st.session_state.attended = {}
        st.session_state.log = []
        st.session_state.camera_running = False
        st.rerun()

    sidebar.markdown("---")
    sidebar.markdown(
        f"{'🟢' if st.session_state.camera_running else '🔴'} "
        f"{'Running' if st.session_state.camera_running else 'Stopped'}"
    )
    sidebar.markdown(f"**Model:** {'SVM (RBF)' if model_key == 'svm' else f'KNN (cosine)'}")
    sidebar.metric("Present", f"{len(st.session_state.log)} / 22")

    if "all_names" not in st.session_state:
        st.session_state.all_names = get_all_names()
    all_names = st.session_state.all_names

    if start_btn and not st.session_state.camera_running:
        cam.start(threshold, model_key)
        st.session_state.camera_running = True
        st.rerun()

    if stop_btn and st.session_state.camera_running:
        cam.stop()
        st.session_state.camera_running = False
        st.rerun()

    if st.session_state.camera_running:
        err = cam.error
        if err:
            st.error(f"Camera error: {err}")
            st.session_state.camera_running = False
            st.rerun()

    main_col, table_col = st.columns([3, 2])

    with main_col:
        st.subheader("Webcam Feed")
        feed_placeholder = st.empty()

    with table_col:
        st.subheader("Attendance")
        table_placeholder = st.empty()

    # Active streaming loop inside a single script execution
    if st.session_state.camera_running:
        while st.session_state.camera_running:
            if not cam.is_running:
                feed_placeholder.error("Camera stopped unexpectedly")
                st.session_state.camera_running = False
                break

            if cam.is_ready:
                jpg = cam.get_jpeg()
                if jpg is not None:
                    feed_placeholder.image(jpg, channels="BGR", width="stretch")
            else:
                feed_placeholder.info("Starting camera & loading models...")

            new_faces = cam.pop_pending()
            if new_faces:
                now = datetime.now().strftime("%H:%M:%S")
                log_changed = False
                for name in new_faces:
                    if name not in st.session_state.attended:
                        st.session_state.attended[name] = True
                        st.session_state.log.append({"Name": name, "Time": now, "Confidence": "—"})
                        log_changed = True
                
                if log_changed:
                    if "csv" in st.session_state:
                        del st.session_state["csv"]

            table_placeholder.markdown(
                make_attendance_html(st.session_state.log, all_names),
                unsafe_allow_html=True
            )

            time.sleep(0.05)
    else:
        # Static view when camera is stopped
        feed_placeholder.info("Click 'Start' to activate the webcam feed.")
        table_placeholder.markdown(
            make_attendance_html(st.session_state.log, all_names),
            unsafe_allow_html=True
        )

    # Render CSV download button
    if "csv" not in st.session_state or "last_log_len" not in st.session_state or st.session_state.last_log_len != len(st.session_state.log):
        csv_lines = ["Status,Name,StudentID,Time,Confidence"]
        for name in all_names:
            r = [r for r in st.session_state.log if r["Name"] == name]
            if r:
                status, t = "Present", r[0]["Time"]
            else:
                status, t = "Absent", "—"
            student_name, student_id = split_name_id(name)
            csv_lines.append(f"{status},{student_name},{student_id},{t},—")
        st.session_state.csv = "\n".join(csv_lines).encode("utf-8")
        st.session_state.last_log_len = len(st.session_state.log)
    csv = st.session_state.csv

    sidebar.download_button("Download CSV", csv,
        f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv", width="stretch")


if __name__ == "__main__":
    main()
