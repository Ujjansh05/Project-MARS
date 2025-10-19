# from flask import Flask, request, jsonify, send_file
# from werkzeug.utils import secure_filename
# from pathlib import Path
# import subprocess
# import uuid
# import os

# app = Flask(__name__)

# # Configure upload/output paths
# UPLOAD_DIR = Path("uploads")
# RESULTS_DIR = Path("results")
# UPLOAD_DIR.mkdir(exist_ok=True)
# RESULTS_DIR.mkdir(exist_ok=True)

# # Entry endpoint for file upload
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     filename = secure_filename(file.filename)
#     uid = str(uuid.uuid4())[:8]
#     input_path = UPLOAD_DIR / f"{uid}_{filename}"
#     file.save(input_path)

#     # Call tracking model
#     try:
#         cmd = [
#             "python", "track_script.py",  # rename your model script to this
#             "--video", str(input_path),
#             "--uid", uid
#         ]
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         return jsonify({"error": f"Model execution failed: {e}"}), 500

#     # Output expected files
#     video_out = RESULTS_DIR / uid / "video_tracking" / input_path.stem / "labels.mp4"
#     csv_out = RESULTS_DIR / uid / "vessel_tracks.csv"

#     return jsonify({
#         "video_url": f"/download/{uid}/video",
#         "csv_url": f"/download/{uid}/csv"
#     })


# @app.route('/download/<uid>/<filetype>', methods=['GET'])
# def download(uid, filetype):
#     video_path = RESULTS_DIR / uid / "video_tracking" / f"{uid}_vid" / "labels.mp4"
#     csv_path = RESULTS_DIR / uid / "vessel_tracks.csv"

#     if filetype == "video" and video_path.exists():
#         return send_file(video_path, as_attachment=True)

#     if filetype == "csv" and csv_path.exists():
#         return send_file(csv_path, as_attachment=True)

#     return jsonify({"error": "File not found"}), 404


# if __name__ == '__main__':
#     app.run(debug=True)


# tracker_logic.py

# tracker_logic.py

# tracker.py

# from pathlib import Path
# import torch
# import cv2
# import csv
# import numpy as np
# from ultralytics import YOLO
# import os
# import sys
# import csv
# import json
# import glob
# import math
# from pathlib import Path
# from datetime import datetime

# import cv2
# import torch
# import numpy as np

# # Configurable params
# IOU_THRESHOLD = 0.45
# TRACKER = "bytetrack.yaml"
# IMGSZ_CANDIDATES = [960, 640, 512]
# CONF_CANDIDATES = [0.05, 0.15, 0.25]
# ALLOW_COCO_FALLBACK = True
# COCO_MODEL = "yolov8n.pt"


# def run_vessel_tracking(video_path: str, model_path: str, output_dir: str):
#     device = 0 if torch.cuda.is_available() else "cpu"
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     debug_dir = output_dir / "debug"
#     debug_dir.mkdir(exist_ok=True)

#     def sample_frames(video_path, max_samples=5):
#         cap = cv2.VideoCapture(str(video_path))
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         step = max(total // (max_samples + 1), 1)
#         idxs = [(i + 1) * step for i in range(max_samples)]
#         frames = []
#         for i, idx in enumerate(idxs):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ok, frame = cap.read()
#             if ok:
#                 frames.append((idx, frame))
#         cap.release()
#         print("sampling den.")
#         return frames

#     def try_infer(model, frames, conf, imgsz):
#         total = 0
#         for idx, img in frames:
#             results = model.predict(source=img, conf=conf, imgsz=imgsz, device=device, verbose=False)
#             total += len(results[0].boxes) if results[0].boxes else 0

#         print("infer done.")
#         return total

#     # Load model
#     try:
#         model = YOLO(str(model_path))
#         print("model yolo")
#     except Exception:
#         if not ALLOW_COCO_FALLBACK:
#             raise
#         model = YOLO(COCO_MODEL)

#     # Sample frames
#     frames = sample_frames(video_path)

#     # Auto-tune params
#     for imgsz in IMGSZ_CANDIDATES:
#         for conf in CONF_CANDIDATES:
#             try:
#                 total_dets = try_infer(model, frames, conf, imgsz)
#                 if total_dets > 0:
#                     break
#             except torch.cuda.OutOfMemoryError:
#                 torch.cuda.empty_cache()
#                 continue
#         else:
#             continue
#         break
#     else:
#         raise RuntimeError("No valid detection config found.")

#     # Run tracking
#     csv_path = output_dir / "vessel_tracks.csv"
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["frame", "id", "cls", "conf", "x1", "y1", "x2", "y2"])
#         results = model.track(
#             source=str(video_path),
#             conf=conf,
#             iou=IOU_THRESHOLD,
#             imgsz=imgsz,
#             device=device,
#             tracker=TRACKER,
#             persist=True,
#             save=True,
#             project=str(output_dir),
#             name="video_tracking",
#             exist_ok=True,
#             stream=True,
#         )
#         for frame_idx, r in enumerate(results):
#             if r.boxes is None:
#                 continue
#             boxes = r.boxes.xyxy.cpu().numpy()
#             ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else np.full(len(boxes), -1)
#             confs = r.boxes.conf.cpu().numpy()
#             clses = r.boxes.cls.cpu().numpy()
#             for box, track_id, conf, cls in zip(boxes, ids, confs, clses):
#                 writer.writerow([frame_idx, int(track_id), int(cls), float(conf), *map(float, box)])

#     return str(csv_path), str(output_dir / "video_tracking")




# import os
# import sys
# import csv
# import json
# import math
# import torch
# import cv2
# import numpy as np

# from pathlib import Path
# from datetime import datetime

# # Constants for tuning and fallback
# CONF_CANDIDATES = [0.05, 0.15, 0.25]
# IMGSZ_CANDIDATES = [960, 640, 512]
# IOU_THRESHOLD = 0.45
# TRACKER = "bytetrack.yaml"
# COCO_MODEL = "yolov8n.pt"
# ALLOW_COCO_FALLBACK = True


# def log(msg: str):
#     print(f"[Track] {msg}")


# def get_device():
#     if torch.cuda.is_available():
#         name = torch.cuda.get_device_name(0)
#         vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
#         log(f"Using GPU: {name} ({vram_gb:.1f} GB)")
#         return 0
#     log("CUDA not available. Running on CPU.")
#     return "cpu"


# def load_model(weights_path_or_name):
#     from ultralytics import YOLO
#     log(f"Loading model: {weights_path_or_name}")
#     return YOLO(str(weights_path_or_name))


# def sample_frames(video_path: Path, debug_dir: Path, max_samples=5):
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise FileNotFoundError(f"Cannot open video: {video_path}")

#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     step = max(total // (max_samples + 1), 1) if total > 0 else 1
#     idxs = [(i + 1) * step for i in range(max_samples)]

#     frames = []
#     for i, idx in enumerate(idxs):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ok, frame = cap.read()
#         if not ok or frame is None:
#             continue
#         frames.append((idx, frame))
#         debug_dir.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(debug_dir / f"sample_{i:02d}_idx{idx}.jpg"), frame)
#     cap.release()
#     return frames


# def try_infer_on_frames(model, frames, conf, imgsz, device, debug_dir):
#     total = 0
#     for i, (idx, img) in enumerate(frames):
#         try:
#             results = model.predict(
#                 source=img,
#                 conf=conf,
#                 imgsz=imgsz,
#                 device=device,
#                 verbose=False
#             )
#         except torch.cuda.OutOfMemoryError:
#             torch.cuda.empty_cache()
#             raise

#         r = results[0]
#         n = 0 if r.boxes is None else len(r.boxes)
#         total += n

#         annotated = r.plot()
#         cv2.imwrite(str(debug_dir / f"probe_conf{conf}_imgsz{imgsz}_frame{i:02d}_dets{n}.jpg"), annotated)

#     return total


# def auto_tune_params(model, frames, device, debug_dir):
#     for imgsz in IMGSZ_CANDIDATES:
#         for conf in CONF_CANDIDATES:
#             log(f"Probing: conf={conf}, imgsz={imgsz}")
#             try:
#                 total_dets = try_infer_on_frames(model, frames, conf, imgsz, device, debug_dir)
#             except torch.cuda.OutOfMemoryError:
#                 log("OOM during probe -> lowering imgsz...")
#                 break
#             if total_dets > 0:
#                 log(f"Detections found with conf={conf}, imgsz={imgsz} (total={total_dets})")
#                 return conf, imgsz
#     return None, None


# def track_video(model, video_path, device, conf, imgsz, output_dir, tracker_config):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     csv_path = output_dir / "vessel_tracks.csv"
#     tracking_dir = output_dir / "video_tracking"

#     frame_idx = -1
#     with open(csv_path, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["frame", "id", "cls", "conf", "x1", "y1", "x2", "y2"])

#         gen = model.track(
#             source=str(video_path),
#             conf=conf,
#             iou=IOU_THRESHOLD,
#             imgsz=imgsz,
#             device=device,
#             tracker=tracker_config,
#             persist=True,
#             save=True,
#             project=str(output_dir),
#             name="video_tracking",
#             exist_ok=True,
#             vid_stride=1,
#             stream=True,
#             verbose=False
#         )

#         for r in gen:
#             frame_idx += 1
#             if r is None or r.boxes is None:
#                 continue

#             boxes = r.boxes
#             xyxy = boxes.xyxy.cpu().numpy()
#             confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
#             clses = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
#             ids = boxes.id.cpu().numpy() if boxes.id is not None else np.full(len(xyxy), -1)

#             for (x1, y1, x2, y2), tid, c, cls in zip(xyxy, ids, confs, clses):
#                 w.writerow([frame_idx, int(tid), int(cls), float(c), float(x1), float(y1), float(x2), float(y2)])

#     video_files = list(tracking_dir.glob("*.mp4"))
#     return csv_path, video_files[0] if video_files else None


# def run_tracking(video_path, model_path, output_dir):
#     """
#     Backend-friendly tracking function.

#     Args:
#         video_path (str | Path): Path to input video
#         model_path (str | Path): Path to YOLOv8 model weights
#         output_dir (str | Path): Output directory for results

#     Returns:
#         tuple: (csv_path, video_path, annotated_video_path)
#     """
#     video_path = Path(video_path)
#     model_path = Path(model_path)
#     output_dir = Path(output_dir)
#     debug_dir = output_dir / "debug"

#     if not video_path.exists():
#         raise FileNotFoundError(f"Video not found: {video_path}")
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model weights not found: {model_path}")

#     device = get_device()
#     model = None

#     try:
#         model = load_model(model_path)
#     except Exception as e:
#         log(f"Failed to load trained model: {e}")
#         if not ALLOW_COCO_FALLBACK:
#             raise RuntimeError("Model loading failed.")
#         log("Falling back to COCO model.")
#         model = load_model(COCO_MODEL)

#     frames = sample_frames(video_path, debug_dir)
#     if not frames:
#         raise RuntimeError("Could not extract frames from video.")

#     conf, imgsz = auto_tune_params(model, frames, device, debug_dir)
#     if (conf is None or imgsz is None) and ALLOW_COCO_FALLBACK:
#         log("Fallback to COCO model for tuning.")
#         model = load_model(COCO_MODEL)
#         conf, imgsz = auto_tune_params(model, frames, device, debug_dir)

#     if conf is None or imgsz is None:
#         raise RuntimeError("Auto-tuning failed to find valid conf/imgsz.")

#     tried_imgsz = [imgsz] + [s for s in IMGSZ_CANDIDATES if s < imgsz]
#     last_err = None

#     for s in tried_imgsz:
#         try:
#             log(f"Tracking with conf={conf}, imgsz={s}")
#             csv_path, output_video_path = track_video(
#                 model, video_path, device, conf, s, output_dir, tracker_config=TRACKER
#             )
#             return output_video_path, csv_path #, video_path, 
#         except torch.cuda.OutOfMemoryError as e:
#             last_err = e
#             torch.cuda.empty_cache()
#             log("OOM during tracking. Retrying with smaller imgsz...")
#         except Exception as e:
#             last_err = e
#             log(f"Tracking error: {e}")
#             break

#     raise RuntimeError(f"Tracking failed. Last error: {last_err}")




# from pathlib import Path
# import torch
# import cv2
# import csv
# import numpy as np
# from ultralytics import YOLO
# import os
# import sys
# import csv
# import json
# import glob
# import math
# from pathlib import Path
# from datetime import datetime

# import cv2
# import torch
# import numpy as np

# # Configurable params
# IOU_THRESHOLD = 0.45
# TRACKER = "bytetrack.yaml"
# IMGSZ_CANDIDATES = [960, 640, 512]
# CONF_CANDIDATES = [0.05, 0.15, 0.25]
# ALLOW_COCO_FALLBACK = True
# COCO_MODEL = "yolov8n.pt"


# def run_vessel_tracking(video_path: str, model_path: str, output_dir: str):
#     device = 0 if torch.cuda.is_available() else "cpu"
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     debug_dir = output_dir / "debug"
#     debug_dir.mkdir(exist_ok=True)

#     def sample_frames(video_path, max_samples=5):
#         cap = cv2.VideoCapture(str(video_path))
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         step = max(total // (max_samples + 1), 1)
#         idxs = [(i + 1) * step for i in range(max_samples)]
#         frames = []
#         for i, idx in enumerate(idxs):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ok, frame = cap.read()
#             if ok:
#                 frames.append((idx, frame))
#         cap.release()
#         print("sampling den.")
#         return frames

#     def try_infer(model, frames, conf, imgsz):
#         total = 0
#         for idx, img in frames:
#             results = model.predict(source=img, conf=conf, imgsz=imgsz, device=device, verbose=False)
#             total += len(results[0].boxes) if results[0].boxes else 0

#         print("infer done.")
#         return total

#     # Load model
#     try:
#         model = YOLO(str(model_path))
#         print("model yolo")
#     except Exception:
#         if not ALLOW_COCO_FALLBACK:
#             raise
#         model = YOLO(COCO_MODEL)

#     # Sample frames
#     frames = sample_frames(video_path)

#     # Auto-tune params
#     for imgsz in IMGSZ_CANDIDATES:
#         for conf in CONF_CANDIDATES:
#             try:
#                 total_dets = try_infer(model, frames, conf, imgsz)
#                 if total_dets > 0:
#                     break
#             except torch.cuda.OutOfMemoryError:
#                 torch.cuda.empty_cache()
#                 continue
#         else:
#             continue
#         break
#     else:
#         raise RuntimeError("No valid detection config found.")

#     # Run tracking
#     csv_path = output_dir / "vessel_tracks.csv"
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["frame", "id", "cls", "conf", "x1", "y1", "x2", "y2"])
#         results = model.track(
#             source=str(video_path),
#             conf=conf,
#             iou=IOU_THRESHOLD,
#             imgsz=imgsz,
#             device=device,
#             tracker=TRACKER,
#             persist=True,
#             save=True,
#             project=str(output_dir),
#             name="video_tracking",
#             exist_ok=True,
#             stream=True,
#         )
#         for frame_idx, r in enumerate(results):
#             if r.boxes is None:
#                 continue
#             boxes = r.boxes.xyxy.cpu().numpy()
#             ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else np.full(len(boxes), -1)
#             confs = r.boxes.conf.cpu().numpy()
#             clses = r.boxes.cls.cpu().numpy()
#             for box, track_id, conf, cls in zip(boxes, ids, confs, clses):
#                 writer.writerow([frame_idx, int(track_id), int(cls), float(conf), *map(float, box)])

#     return str(csv_path), str(output_dir / "video_tracking")




# import os
# import sys
# import csv
# import json
# import math
# import torch
# import cv2
# import numpy as np

# from pathlib import Path
# from datetime import datetime

# # Constants for tuning and fallback
# CONF_CANDIDATES = [0.05, 0.15, 0.25]
# IMGSZ_CANDIDATES = [960, 640, 512]
# IOU_THRESHOLD = 0.45
# TRACKER = "bytetrack.yaml"
# COCO_MODEL = "yolov8n.pt"
# ALLOW_COCO_FALLBACK = True


# def log(msg: str):
#     print(f"[Track] {msg}")


# def get_device():
#     if torch.cuda.is_available():
#         name = torch.cuda.get_device_name(0)
#         vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
#         log(f"Using GPU: {name} ({vram_gb:.1f} GB)")
#         return 0
#     log("CUDA not available. Running on CPU.")
#     return "cpu"


# def load_model(weights_path_or_name):
#     from ultralytics import YOLO
#     log(f"Loading model: {weights_path_or_name}")
#     return YOLO(str(weights_path_or_name))


# def sample_frames(video_path: Path, debug_dir: Path, max_samples=5):
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise FileNotFoundError(f"Cannot open video: {video_path}")

#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     step = max(total // (max_samples + 1), 1) if total > 0 else 1
#     idxs = [(i + 1) * step for i in range(max_samples)]

#     frames = []
#     for i, idx in enumerate(idxs):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ok, frame = cap.read()
#         if not ok or frame is None:
#             continue
#         frames.append((idx, frame))
#         debug_dir.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(debug_dir / f"sample_{i:02d}_idx{idx}.jpg"), frame)
#     cap.release()
#     return frames


# def try_infer_on_frames(model, frames, conf, imgsz, device, debug_dir):
#     total = 0
#     for i, (idx, img) in enumerate(frames):
#         try:
#             results = model.predict(
#                 source=img,
#                 conf=conf,
#                 imgsz=imgsz,
#                 device=device,
#                 verbose=False
#             )
#         except torch.cuda.OutOfMemoryError:
#             torch.cuda.empty_cache()
#             raise

#         r = results[0]
#         n = 0 if r.boxes is None else len(r.boxes)
#         total += n

#         annotated = r.plot()
#         cv2.imwrite(str(debug_dir / f"probe_conf{conf}_imgsz{imgsz}_frame{i:02d}_dets{n}.jpg"), annotated)

#     return total


# def auto_tune_params(model, frames, device, debug_dir):
#     for imgsz in IMGSZ_CANDIDATES:
#         for conf in CONF_CANDIDATES:
#             log(f"Probing: conf={conf}, imgsz={imgsz}")
#             try:
#                 total_dets = try_infer_on_frames(model, frames, conf, imgsz, device, debug_dir)
#             except torch.cuda.OutOfMemoryError:
#                 log("OOM during probe -> lowering imgsz...")
#                 break
#             if total_dets > 0:
#                 log(f"Detections found with conf={conf}, imgsz={imgsz} (total={total_dets})")
#                 return conf, imgsz
#     return None, None


# def track_video(model, video_path, device, conf, imgsz, output_dir, tracker_config):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     csv_path = output_dir / "vessel_tracks.csv"
#     tracking_dir = output_dir / "video_tracking"

#     frame_idx = -1
#     with open(csv_path, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["frame", "id", "cls", "conf", "x1", "y1", "x2", "y2"])

#         gen = model.track(
#             source=str(video_path),
#             conf=conf,
#             iou=IOU_THRESHOLD,
#             imgsz=imgsz,
#             device=device,
#             tracker=tracker_config,
#             persist=True,
#             save=True,
#             project=str(output_dir),
#             name="video_tracking",
#             exist_ok=True,
#             vid_stride=1,
#             stream=True,
#             verbose=False
#         )

#         for r in gen:
#             frame_idx += 1
#             if r is None or r.boxes is None:
#                 continue

#             boxes = r.boxes
#             xyxy = boxes.xyxy.cpu().numpy()
#             confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
#             clses = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
#             ids = boxes.id.cpu().numpy() if boxes.id is not None else np.full(len(xyxy), -1)

#             for (x1, y1, x2, y2), tid, c, cls in zip(xyxy, ids, confs, clses):
#                 w.writerow([frame_idx, int(tid), int(cls), float(c), float(x1), float(y1), float(x2), float(y2)])

#     video_files = list(tracking_dir.glob("*.mp4"))
#     return csv_path, video_files[0] if video_files else None


# def run_tracking(video_path, model_path, output_dir):
#     """
#     Backend-friendly tracking function.

#     Args:
#         video_path (str | Path): Path to input video
#         model_path (str | Path): Path to YOLOv8 model weights
#         output_dir (str | Path): Output directory for results

#     Returns:
#         tuple: (csv_path, video_path, annotated_video_path)
#     """
#     video_path = Path(video_path)
#     model_path = Path(model_path)
#     output_dir = Path(output_dir)
#     debug_dir = output_dir / "debug"

#     if not video_path.exists():
#         raise FileNotFoundError(f"Video not found: {video_path}")
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model weights not found: {model_path}")

#     device = get_device()
#     model = None

#     try:
#         model = load_model(model_path)
#     except Exception as e:
#         log(f"Failed to load trained model: {e}")
#         if not ALLOW_COCO_FALLBACK:
#             raise RuntimeError("Model loading failed.")
#         log("Falling back to COCO model.")
#         model = load_model(COCO_MODEL)

#     frames = sample_frames(video_path, debug_dir)
#     if not frames:
#         raise RuntimeError("Could not extract frames from video.")

#     conf, imgsz = auto_tune_params(model, frames, device, debug_dir)
#     if (conf is None or imgsz is None) and ALLOW_COCO_FALLBACK:
#         log("Fallback to COCO model for tuning.")
#         model = load_model(COCO_MODEL)
#         conf, imgsz = auto_tune_params(model, frames, device, debug_dir)

#     if conf is None or imgsz is None:
#         raise RuntimeError("Auto-tuning failed to find valid conf/imgsz.")

#     tried_imgsz = [imgsz] + [s for s in IMGSZ_CANDIDATES if s < imgsz]
#     last_err = None

#     for s in tried_imgsz:
#         try:
#             log(f"Tracking with conf={conf}, imgsz={s}")
#             csv_path, output_video_path = track_video(
#                 model, video_path, device, conf, s, output_dir, tracker_config=TRACKER
#             )
#             return output_video_path, csv_path #, video_path, 
#         except torch.cuda.OutOfMemoryError as e:
#             last_err = e
#             torch.cuda.empty_cache()
#             log("OOM during tracking. Retrying with smaller imgsz...")
#         except Exception as e:
#             last_err = e
#             log(f"Tracking error: {e}")
#             break
#     raise RuntimeError(f"Tracking failed. Last error: {last_err}")


# tracker.py
# Backend/tracker_logic.py
#!/usr/bin/env python3
"""
YOLOv8 + ByteTrack tracking logic.
"""

import csv
import glob
from pathlib import Path
import cv2
import torch
import numpy as np

# ---------- Config ----------
WORK_DIR = Path(__file__).resolve().parents[1]  # .../project
# Preferred trained model, if you have that exact path:
PREFERRED_MODEL = WORK_DIR / "models" / "detection" / "training" / "weights" / "best.pt"

# Search space + defaults
CONF_CANDIDATES = [0.05, 0.15, 0.25]
IMGSZ_CANDIDATES = [960, 640, 512]
IOU_THRESHOLD = 0.45
TRACKER = "bytetrack.yaml"

# Fallback to COCO boat if your custom model can’t detect on sample frames
ALLOW_COCO_FALLBACK = True
COCO_MODEL = "yolov8n.pt"

def log(msg): print(f"[Track] {msg}")

def get_device():
    if torch.cuda.is_available():
        log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return 0
    log("CUDA not available – using CPU")
    return "cpu"

def resolve_model_path():
    # 1) If you placed your weights at the preferred path:
    if PREFERRED_MODEL.exists():
        return PREFERRED_MODEL
    # 2) Otherwise, use the newest *.pt anywhere under models/
    candidates = list(WORK_DIR.glob("models/**/*.pt"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    raise FileNotFoundError("No model weights found. Put your best.pt under project/models/...")

def sample_frames(video_path, max_samples=5, dbg_dir=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // (max_samples + 1), 1)
    idxs = [(i + 1) * step for i in range(max_samples)]
    frames = []
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
            if dbg_dir:
                dbg_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dbg_dir / f"raw_{i:02d}.jpg"), frame)
    cap.release()
    if not frames:
        raise RuntimeError("Could not read frames from the video.")
    return frames

def load_model(weights):
    from ultralytics import YOLO
    log(f"Loading model: {weights}")
    return YOLO(str(weights))

def try_infer_on_frames(model, frames, conf, imgsz, device, dbg_dir=None):
    total = 0
    for i, (_, img) in enumerate(frames):
        results = model.predict(source=img, conf=conf, imgsz=imgsz, device=device, verbose=False)
        r = results[0]
        n = 0 if r.boxes is None else len(r.boxes)
        total += n
        if dbg_dir is not None:
            plot = r.plot()
            cv2.imwrite(str(dbg_dir / f"probe_c{conf}_s{imgsz}_{i:02d}_d{n}.jpg"), plot)
    return total

def auto_tune_params(model, frames, device, dbg_dir=None):
    for imgsz in IMGSZ_CANDIDATES:
        for conf in CONF_CANDIDATES:
            log(f"Probe: conf={conf}, imgsz={imgsz}")
            total = try_infer_on_frames(model, frames, conf, imgsz, device, dbg_dir)
            if total > 0:
                log(f"✓ Probes found detections with conf={conf}, imgsz={imgsz}")
                return conf, imgsz
    return None, None

def track_video(model, video_path, device, conf, imgsz, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vessel_tracks.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "id", "cls", "conf", "x1", "y1", "x2", "y2"])

        gen = model.track(
            source=str(video_path),
            conf=conf,
            iou=IOU_THRESHOLD,
            imgsz=imgsz,
            device=device,
            tracker=TRACKER,
            persist=True,
            save=True,                   # save annotated video
            project=str(out_dir),        # Ultralytics will create out_dir/name/
            name="video_tracking",
            exist_ok=True,
            stream=True,
            verbose=False
        )

        frame_idx = -1
        for r in gen:
            frame_idx += 1
            if r is None or r.boxes is None:
                continue
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy() if b.conf is not None else np.zeros((len(xyxy),))
            clses = b.cls.cpu().numpy() if b.cls is not None else np.zeros((len(xyxy),))
            ids = b.id.cpu().numpy() if b.id is not None else np.full((len(xyxy),), -1)

            for (x1, y1, x2, y2), tid, c, cls in zip(xyxy, ids, confs, clses):
                w.writerow([frame_idx, int(tid), int(cls), float(c), float(x1), float(y1), float(x2), float(y2)])

    return csv_path

def run_tracking_pipeline(video_path: Path, output_dir: Path):
    """
    Returns: (video_path: Path, csv_path: Path)
    The returned paths are on disk under Backend/results/<job_id>/...
    """
    device = get_device()
    dbg_dir = output_dir / "debug"
    dbg_dir.mkdir(parents=True, exist_ok=True)

    # Load trained model, or fallback if needed
    try:
        model_path = resolve_model_path()
        model = load_model(model_path)
    except Exception as e:
        log(f"Model load error: {e}")
        if not ALLOW_COCO_FALLBACK:
            raise
        log("Falling back to COCO yolov8n.pt")
        model = load_model(COCO_MODEL)

    # Probe frames to find a working conf/imgsz
    frames = sample_frames(video_path, max_samples=5, dbg_dir=dbg_dir)
    conf, imgsz = auto_tune_params(model, frames, device, dbg_dir)
    if conf is None or imgsz is None and ALLOW_COCO_FALLBACK:
        coco = load_model(COCO_MODEL)
        conf, imgsz = auto_tune_params(coco, frames, device, dbg_dir)
        if conf is not None and imgsz is not None:
            model = coco

    if conf is None or imgsz is None:
        raise RuntimeError("No detections found during probes. Check your video/model.")

    # Track the full video
    csv_path = track_video(model, video_path, device, conf, imgsz, output_dir)

    # Find the annotated video saved by Ultralytics in output_dir/video_tracking/
    vid_dir = output_dir / "video_tracking"
    vids = []
    for ext in ("*.mp4", "*.avi", "*.mkv"):
        vids.extend(glob.glob(str(vid_dir / ext)))
    vids.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if not vids:
        raise RuntimeError("Annotated video was not produced by Ultralytics.")
    return Path(vids[0]), Path(csv_path)
