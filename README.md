# =============================================
# README.md
# =============================================
# FaceRecognition Attendance (PySide6 + ArcFace ONNX + MediaPipe)


A modern desktop app for university attendance:
- **Face login** for admin/staff
- **Live camera** attendance marking for students
- **Admin enrollment** (upload/capture student photo → stored embeddings)
- **CSV logs** saved per run (file name = timestamp)
- **Log viewer** inside the app
- **Error messaging** when a face is detected but not matched (shows "NO matching ID")


> Stack: PySide6 UI, OpenCV camera, InsightFace (ArcFace) embeddings via onnxruntime, SQLite for metadata.


## Install
```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```


## Run
```bash
python -m app.main
```


## Default PIN
- Default PIN: `1234`
- Change it later from the **Admin → Settings** area.


## Folders
- `app/data/students/` – uploaded/captured student photos
- `app/data/attendance/` – CSV logs per run (e.g., `attendance_2025-08-30_001530.csv`)
- `app/data/embeddings.db` – SQLite database


## High Precision Matching
- Uses ArcFace embeddings (InsightFace) with cosine similarity.
- Default threshold = 0.35. If match < threshold → **“NO matching ID”**.
- Adjust threshold in Settings.


---