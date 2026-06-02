import streamlit as st
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Face Attendance", layout="wide")

MODEL_SVM_PATH = Path("svm/svm_face_classifier.pkl")
MODEL_KNN_PATH = Path("knn/knn_face_classifier.pkl")
PROCESS_EVERY_N = 8
REFRESH_MS = 200

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

import torch
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEVICE_STR = f"{'GPU' if torch.cuda.is_available() else 'CPU'} ({_DEVICE})"


class FaceRecognizerProcessor(VideoProcessorBase):
    def __init__(self):
        self.threshold = 0.5
        self.model_type = "svm"
        self._error = None
        self._heavy_initialized = False
        
        self._pending = []
        self._pending_lock = threading.Lock()
        
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

    def pop_pending(self):
        with self._pending_lock:
            items = list(self._pending)
            self._pending.clear()
            return items

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not self._heavy_initialized:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Load/reload classifiers if needed
        if not hasattr(self, "_current_model_type") or self._current_model_type != self.model_type or not hasattr(self, "_clf"):
            try:
                import pickle
                model_path = MODEL_KNN_PATH if self.model_type == "knn" else MODEL_SVM_PATH
                with open(model_path, "rb") as f:
                    d = pickle.load(f)
                self._clf = d.get("classifier") or d.get("svm") or d.get("knn")
                self._le = d["label_encoder"]
                self._current_model_type = self.model_type
            except Exception as e:
                self._error = f"Model load error: {str(e)}"
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        if not hasattr(self, "_frame_count"):
            self._frame_count = 0
            self._last_embeddings = []

        self._frame_count += 1
        do_detect = (self._frame_count % PROCESS_EVERY_N == 0)

        cv2 = self.cv2
        np = self.np

        if do_detect:
            h, w = img.shape[:2]
            scale_x = w / 320.0
            scale_y = h / 240.0

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                        face_crop = cv2.resize(img[y1:y2, x1:x2], (160, 160))
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
                    ps = self._clf.predict_proba([emb])[0]
                    pi = np.argmax(ps)
                    conf = ps[pi]
                    name = self._le.classes_[pi]
                    self._last_embeddings.append((x1, y1, x2, y2, name, conf))

        if self._last_embeddings:
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
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                tw, th2 = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(img, (x1, y1 - th2 - 8), (x1 + tw + 8, y1), color, -1)
                cv2.putText(img, label, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


FALLBACK_NAMES = [
    "daochitrung_23ba14295",
    "dohoanglong_2410559",
    "dohunganh_23ba14011",
    "duongtunganh_2410024",
    "kieuminhduc_2411138",
    "lehuuduc2411139",
    "lethanhtra_2410980",
    "letrongtien_23ba14280",
    "luuminhhoang_23BA14119",
    "nguyenbinhduong_22BA13091",
    "nguyendinhgiang_23ba14089",
    "nguyendinhminhquang_23BA14244",
    "nguyenhong_truong_23BA14300",
    "nguyenkhaiminh_2410607",
    "nguyenmy_2410678",
    "nguyenngochieu_23BA14109",
    "nguyennguyennhat_23BA14222",
    "nguyentathoangviet_23BA14320",
    "phanduyhoang_23BA14117",
    "phanminhtrang_23BA14290",
    "tranmanhhung_23BA14127",
    "transonbach_2410136"
]


@st.cache_data(ttl=60)
def get_all_names():
    path = Path("dataset/train_augmented")
    if not path.exists():
        return FALLBACK_NAMES
    names = sorted(set(
        p.stem for p in path.iterdir() if p.is_dir()
    ))
    return names if names else FALLBACK_NAMES



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
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    sidebar = st.sidebar
    sidebar.title("Attendance")
    
    sidebar.markdown(f"**Device status:** {_DEVICE_STR}")

    model_type = sidebar.radio("Classifier", ["SVM (RBF)", "KNN"], index=0,
                               horizontal=True)
    model_key = "knn" if model_type.startswith("KNN") else "svm"

    threshold = sidebar.slider("Confidence", 0.0, 1.0, 0.5, 0.05)

    if sidebar.button("Reset", width="stretch"):
        st.session_state.attended = {}
        st.session_state.log = []
        st.rerun()

    sidebar.markdown("---")
    sidebar.markdown(f"**Model:** {'SVM (RBF)' if model_key == 'svm' else f'KNN (cosine)'}")
    sidebar.metric("Present", f"{len(st.session_state.log)} / 22")

    if "all_names" not in st.session_state:
        st.session_state.all_names = get_all_names()
    all_names = st.session_state.all_names

    main_col, table_col = st.columns([3, 2])

    with main_col:
        st.subheader("Webcam Feed")
        
        webrtc_ctx = webrtc_streamer(
            key="face-attendance",
            video_processor_factory=FaceRecognizerProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.threshold = threshold
            webrtc_ctx.video_processor.model_type = model_key
            
            new_faces = webrtc_ctx.video_processor.pop_pending()
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

        if webrtc_ctx.state.playing:
            st_autorefresh(interval=1000, limit=10000, key="attendance-refresh")

    with table_col:
        st.subheader("Attendance")
        st.markdown(
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
