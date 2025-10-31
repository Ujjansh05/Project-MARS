# #!/usr/bin/env python3
# """
# Step 3: Track vessels across frames or in video using YOLOv8 + ByteTrack
# """

# import sys
# import json
# import csv
# from pathlib import Path
# import torch

# # ==== CONFIG ====
# WORK_DIR = Path(__file__).resolve().parents[1]
# MODEL_PATH = WORK_DIR / "models" / "detection" / "training" / "weights" / "best.pt"

# # Input video or directory of images
# VIDEO_PATH = Path(r"E:\mars\data\videos\vid2.mp4")
# OUTPUT_DIR = WORK_DIR / "results" / "tracking"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CONF_THRESHOLD = 0.25
# IOU_THRESHOLD = 0.45

# def check_gpu():
#     if torch.cuda.is_available():
#         print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
#     else:
#         print("[WARNING] CUDA not available. Running on CPU.")

# def load_model(model_path):
#     from ultralytics import YOLO
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model not found at {model_path}")
#     print(f"[INFO] Loading model from {model_path}")
#     return YOLO(str(model_path))

# def main():
#     check_gpu()
#     model = load_model(MODEL_PATH)

#     print(f"[STEP] Starting vessel tracking on: {VIDEO_PATH}")

#     # Run YOLOv8 tracking (ByteTrack)
#     results = model.track(
#         source=str(VIDEO_PATH),
#         conf=CONF_THRESHOLD,
#         iou=IOU_THRESHOLD,
#         device=0 if torch.cuda.is_available() else "cpu",
#         tracker="bytetrack.yaml",  # Built-in tracker
#         save=True,
#         project=str(OUTPUT_DIR),
#         name="video_tracking",
#         exist_ok=True
#     )

#     # Export CSV with tracked object data
#     csv_path = OUTPUT_DIR / "vessel_tracks.csv"
#     with open(csv_path, mode="w", newline="") as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(["frame", "id", "class", "conf", "x1", "y1", "x2", "y2"])

#         for r in results:
#             frame_num = r.path  # Frame index
#             if hasattr(r, 'boxes') and r.boxes is not None:
#                 boxes = r.boxes.xyxy.cpu().numpy()
#                 ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [-1] * len(boxes)
#                 confs = r.boxes.conf.cpu().numpy()
#                 classes = r.boxes.cls.cpu().numpy()

#                 for box, track_id, conf, cls in zip(boxes, ids, confs, classes):
#                     csv_writer.writerow([
#                         frame_num,
#                         int(track_id),
#                         int(cls),
#                         float(conf),
#                         float(box[0]), float(box[1]), float(box[2]), float(box[3])
#                     ])

#     print(f"[SUCCESS] Tracking complete. Video & data saved in {OUTPUT_DIR}")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Step 3: Robust vessel tracking with YOLOv8 + ByteTrack
- Sanity-checks a few frames first and auto-selects conf/imgsz that actually detect.
- Handles GPU OOM by retrying with smaller imgsz.
- Falls back to COCO yolov8n.pt (boat class) if trained model fails on the sampled frames.
- Saves debug images and a CSV log (frame,id,conf,bbox).
"""

import os
import sys
import csv
import json
import glob
import math
from pathlib import Path
from datetime import datetime

import cv2
import torch
import numpy as np

# ========= CONFIG (edit as needed) =========
WORK_DIR = Path(__file__).resolve().parents[1]

# Your input video:
VIDEO_PATH = Path(r"E:\mars\data\videos\vid2.mp4")

# Output folders:
OUT_DIR = WORK_DIR / "results" / "tracking"
DBG_DIR = OUT_DIR / "debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DBG_DIR.mkdir(parents=True, exist_ok=True)

# Preferred trained model:
PREFERRED_MODEL = WORK_DIR / "models" / "detection" / "training" / "weights" / "best.pt"

# Auto-tune search spaces:
CONF_CANDIDATES = [0.05, 0.15, 0.25]
IMGSZ_CANDIDATES = [960, 640, 512]  # tries larger first for better recall, then smaller if OOM or no dets

# Tracking params:
IOU_THRESHOLD = 0.45
TRACKER = "bytetrack.yaml"

# If trained model finds nothing in sanity check, try COCO base as last resort:
ALLOW_COCO_FALLBACK = True
COCO_MODEL = "yolov8n.pt"

# ===========================================


def log(msg: str):
    print(f"[Track] {msg}")


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log(f"Using GPU: {name} ({vram_gb:.1f} GB)")
        return 0
    log("CUDA not available. Running on CPU.")
    return "cpu"


def resolve_model_path():
    # 1) Use preferred path if it exists
    if PREFERRED_MODEL.exists():
        return PREFERRED_MODEL

    # 2) Try model_info.json
    mi = WORK_DIR / "model_info.json"
    if mi.exists():
        try:
            with open(mi, "r") as f:
                p = Path(json.load(f).get("model_path", ""))
            if p.exists():
                return p
        except Exception:
            pass

    # 3) Find most recent best.pt under models/detection
    candidates = list(WORK_DIR.glob("models/detection/**/best.pt"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    raise FileNotFoundError("Could not locate trained model weights (best.pt).")


def sample_frames(video_path: Path, max_samples=5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Try to read first few frames anyway
        idxs = list(range(max_samples))
    else:
        step = max(total // (max_samples + 1), 1)
        idxs = [(i + 1) * step for i in range(max_samples)]
        idxs = [min(i, max(0, total - 1)) for i in idxs]

    frames = []
    for i, target_idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append((target_idx, frame))
        # Save raw debug frame
        cv2.imwrite(str(DBG_DIR / f"raw_frame_{i:02d}_idx{target_idx}.jpg"), frame)
    cap.release()
    return frames


def try_infer_on_frames(model, frames, conf, imgsz, device):
    """
    Runs prediction on sampled frames with given conf/imgsz.
    Returns: total_dets, last_annotated_paths
    """
    total = 0
    saved = []
    for i, (idx, img) in enumerate(frames):
        try:
            results = model.predict(
                source=img,  # numpy array
                conf=conf,
                imgsz=imgsz,
                device=device,
                verbose=False
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise

        # results is a list of Results
        r = results[0]
        n = 0 if r.boxes is None else len(r.boxes)
        total += n

        # Save annotated preview
        plot = r.plot()  # BGR numpy image
        out_path = DBG_DIR / f"probe_conf{conf}_imgsz{imgsz}_frame{i:02d}_idx{idx}_dets{n}.jpg"
        cv2.imwrite(str(out_path), plot)
        saved.append(out_path)
    return total, saved


def auto_tune_params(model, frames, device):
    """
    Try combos to find one that yields detections on sampled frames.
    Also handles GPU OOM by retrying with smaller imgsz.
    """
    for imgsz in IMGSZ_CANDIDATES:
        for conf in CONF_CANDIDATES:
            log(f"Probe: conf={conf}, imgsz={imgsz}")
            try:
                total_dets, _ = try_infer_on_frames(model, frames, conf, imgsz, device)
            except torch.cuda.OutOfMemoryError:
                log("OOM during probe -> lowering imgsz...")
                break  # try next smaller imgsz
            if total_dets > 0:
                log(f"Found detections on probes with conf={conf}, imgsz={imgsz} (total={total_dets})")
                return conf, imgsz
    return None, None


def load_model(weights_path_or_name):
    from ultralytics import YOLO
    log(f"Loading model: {weights_path_or_name}")
    return YOLO(str(weights_path_or_name))


def track_video(model, video_path, device, conf, imgsz, out_dir, iou=0.45, tracker="bytetrack.yaml"):
    """
    Runs ByteTrack and logs per-frame boxes to CSV. Uses stream=True to collect IDs.
    Also asks Ultralytics to save an annotated video to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vessel_tracks.csv"

    frame_idx = -1
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "id", "cls", "conf", "x1", "y1", "x2", "y2"])

        gen = model.track(
            source=str(video_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            tracker=tracker,
            persist=True,
            save=True,
            project=str(out_dir),
            name="video_tracking",
            exist_ok=True,
            vid_stride=1,
            stream=True,        # yield results per frame
            verbose=False
        )

        for r in gen:
            frame_idx += 1
            if r is None or r.boxes is None:
                continue

            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),))
            clses = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros((len(xyxy),))
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy()
            else:
                ids = np.full((len(xyxy),), -1)

            for (x1, y1, x2, y2), tid, c, cls in zip(xyxy, ids, confs, clses):
                w.writerow([frame_idx, int(tid), int(cls), float(c), float(x1), float(y1), float(x2), float(y2)])

    return csv_path


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    device = get_device()
    DBG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load trained model or fall back
    try:
        trained_model_path = resolve_model_path()
        model = load_model(trained_model_path)
    except Exception as e:
        log(f"Could not load trained model: {e}")
        if not ALLOW_COCO_FALLBACK:
            sys.exit(1)
        log("Falling back to COCO model.")
        model = load_model(COCO_MODEL)

    # 2) Sample frames & probe detections
    frames = sample_frames(VIDEO_PATH, max_samples=5)
    if not frames:
        raise RuntimeError("Could not read any frames from video.")

    conf, imgsz = None, None
    try:
        conf, imgsz = auto_tune_params(model, frames, device)
    except Exception as e:
        log(f"Probe error: {e}")

    # If trained model fails to detect on probes, optionally try COCO fallback
    if (conf is None or imgsz is None) and ALLOW_COCO_FALLBACK:
        log("No detections on probes with trained model. Trying COCO fallback...")
        coco_model = load_model(COCO_MODEL)
        conf, imgsz = auto_tune_params(coco_model, frames, device)
        if conf is None or imgsz is None:
            log("Still no detections on probes with COCO either. Check video content/domain.")
            sys.exit(1)
        model = coco_model  # use fallback

    if conf is None or imgsz is None:
        log("No working conf/imgsz combo found. Exiting.")
        sys.exit(1)

    # 3) Tracking with chosen params; handle OOM by resizing attempts
    tried_imgsz = [imgsz] + [s for s in IMGSZ_CANDIDATES if s < imgsz]
    last_err = None
    for s in tried_imgsz:
        try:
            log(f"Starting tracking with conf={conf}, imgsz={s}")
            csv_path = track_video(model, VIDEO_PATH, device, conf, s, OUT_DIR, iou=IOU_THRESHOLD, tracker=TRACKER)
            log(f"SUCCESS. CSV saved at: {csv_path}")
            log(f"Annotated video is in: {OUT_DIR / 'video_tracking'}")
            return
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            torch.cuda.empty_cache()
            log("OOM during tracking -> trying smaller imgsz...")
        except Exception as e:
            last_err = e
            log(f"Tracking error: {e}")
            break

    log(f"Failed to complete tracking. Last error: {last_err}")
    sys.exit(1)


if __name__ == "__main__":
    main()
