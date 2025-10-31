# 🌊 Project MARS – Maritime Activity Recognition System

Automatically detect, track, and analyze maritime vessels from satellite videos using YOLOv8 + ByteTrack.

---

## 🧠 About

**Project MARS** is an AI-powered maritime monitoring system that processes satellite video feeds to:

- Detect vessels using YOLOv8
- Track them across frames with ByteTrack
- Output an annotated video
- Generate CSV logs of detections

Built with:

- Flask backend
- Clean HTML/CSS/JS frontend
- Deep learning + computer vision

---

## ✨ Features

| Feature                   | Description                                                            |
| ------------------------- | ---------------------------------------------------------------------- |
| 📅 Video Upload           | Upload any `.mp4` satellite video via the web interface                |
| 🛰️ Vessel Detection      | YOLOv8 detects maritime vessels per frame                              |
| 🎯 ByteTrack Tracking     | Assigns unique IDs to vessels over time                                |
| 🎥 Annotated Output Video | Bounding boxes + track IDs overlaid in output `.mp4`                   |
| 📃 CSV Export             | Frame-by-frame detection logs as `vessel_tracks.csv`                   |
| ⚡ Fallback Logic          | Falls back to YOLOv8 COCO model if custom weights don’t detect vessels |

---

## 📂 Project Structure

```
project-mars/
├── app.py              # Flask API server
├── tracker.py          # Inference & tracking logic
├── models/
│   └── best.pt         # Custom YOLOv8 model
├── jobs/               # Per-job outputs (videos, CSVs)
├── uploads/            # Uploaded videos
├── index.html          # Web UI
├── script.js           # Frontend logic
├── style.css           # CSS styling
└── requirements.txt
```

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.12+
- pip
- GPU (optional, for faster inference)

### ➜ Clone & Install

```bash
git clone https://github.com/your-username/project-mars.git
cd project-mars
pip install -r requirements.txt
```

### 🌐 Run Flask API

```bash
python app.py
```

Runs on: `http://127.0.0.1:5000`

### 🌐 Open Frontend

Just open `index.html` in your browser to access the control panel.

---


### 📄 CSV Format

```
frame,id,class,conf,x1,y1,x2,y2
3,1,0,0.92,100,50,140,90
```

---

## 🚧 API Endpoints

| Route                   | Method | Description                   |
| ----------------------- | ------ | ----------------------------- |
| `/track`                | POST   | Upload video + start analysis |
| `/status/<job_id>`      | GET    | Poll job status               |
| `/jobs/<job_id>/<file>` | GET    | Download CSV or video result  |

---

## 💼 Deployment

- ✅ Works locally with Python 3
- 🚧 Docker version planned
- ☁️ Ideal for deployment on AWS EC2, Azure, or Render

---

## 📆 Roadmap

- [ ] Add Docker support
- [ ] Real-time map visualization
- [ ] Geo-tagged detection overlay

---

## 📚 License

MIT License — free to use and modify with attribution.

---


Enjoy tracking ships like a satellite intelligence analyst! 🚀🚢
