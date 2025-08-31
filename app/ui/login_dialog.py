# app/ui/login_dialog.py
import os
import glob
import time
import cv2
from typing import List, Tuple
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox,
    QSizePolicy,
)
from services import db, face_engine

# gallery item: (id, name, matric, embedding)
GalleryItem = Tuple[str, str, str, list]

class LoginDialog(QDialog):
    def __init__(self, db_path: str, admin_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Admin Login")
        self.resize(720, 520)
        self.setMinimumSize(700, 480)

        self.db_path = db_path
        self.admin_dir = admin_dir
        self.threshold = float(db.get_setting('cosine_threshold', '0.40'))
        self.consecutive = max(1, int(db.get_setting('consecutive_frames', '2')))

        # UI
        self.video_label = QLabel("Initializing camera…")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111; border-radius:12px; padding:8px;")
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setScaledContents(False)

        self.status_label = QLabel("Look at the camera to login with Face ID, or enter PIN.")
        self.status_label.setWordWrap(True)

        self.pin_edit = QLineEdit()
        self.pin_edit.setEchoMode(QLineEdit.Password)
        self.pin_edit.setPlaceholderText("Admin PIN")

        self.pin_btn = QPushButton("Login with PIN")
        self.pin_btn.clicked.connect(self._check_pin)

        row = QHBoxLayout()
        row.addWidget(self.pin_edit, 1)
        row.addWidget(self.pin_btn)

        root = QVBoxLayout(self)
        root.addWidget(self.video_label, 1)
        root.addWidget(self.status_label)
        root.addLayout(row)

        # State for face login
        self._stable_id = None
        self._stable_count = 0
        self._verified = False
        self._verify_start = None
        self._verify_name = ""
        self._verify_bbox = None

        self.gallery: List[GalleryItem] = self._build_admin_gallery(self.admin_dir)

        # Camera (prefer DirectShow on Windows)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Camera error: could not open device.")
        else:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

        if not self.gallery:
            self.status_label.setText(
                "No admin photos found.\n\n"
                "Put one or more clear, frontal photos in:\n"
                f"{os.path.abspath(self.admin_dir)}\n\n"
                "Then try again or use the PIN."
            )

    def _check_pin(self):
        pin = self.pin_edit.text().strip()
        if not pin:
            QMessageBox.information(self, "Missing", "Enter your admin PIN.")
            return
        expected = db.get_setting('admin_pin', '1234')
        if pin == expected:
            self._cleanup_camera()
            self.accept()
        else:
            QMessageBox.critical(self, "Invalid PIN", "Incorrect PIN.")

    def _tick(self):
        if not self.cap or not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if not ok:
            return

        # If we've already verified, keep showing overlay and count down 2s then accept
        if self._verified:
            self._draw_verified_overlay(frame, self._verify_bbox, self._verify_name)
            self._display_frame(frame)
            if (time.monotonic() - self._verify_start) >= 2.0:
                self._cleanup_camera()
                self.accept()
            return

        # Normal flow: detect+match
        emb, bbox, size = face_engine.get_face_embedding_bgr(frame)
        if emb is None:
            self.status_label.setText("No face detected. Center your face with good lighting.")
            self._stable_id = None
            self._stable_count = 0
            self._display_frame(frame)
            return

        if not self.gallery:
            # Just draw the box for UX if we have it
            if bbox:
                self._draw_box(frame, bbox, (0, 165, 255))  # orange
            self.status_label.setText("No admin photos found. Use PIN or add images to data/admins.")
            self._display_frame(frame)
            return

        match = face_engine.best_match(emb, self.gallery)
        if not match:
            if bbox:
                self._draw_box(frame, bbox, (0, 0, 255))  # red
            self.status_label.setText("No matching admin found. Add your photos to data/admins or use PIN.")
            self._stable_id = None
            self._stable_count = 0
            self._display_frame(frame)
            return

        sid, name, matric, sim = match
        if sim >= self.threshold:
            # stability gating
            if self._stable_id != sid:
                self._stable_id = sid
                self._stable_count = 1
            else:
                self._stable_count += 1

            self.status_label.setText(f"Recognizing {name}… ({self._stable_count}/{self.consecutive})  sim={sim:.2f}")

            if self._stable_count >= self.consecutive:
                # Mark as verified, capture bbox for overlay, start 2s timer
                self._verified = True
                self._verify_start = time.monotonic()
                self._verify_name = name
                self._verify_bbox = bbox
                # Draw overlay immediately on the current frame
                self._draw_verified_overlay(frame, bbox, name)
            else:
                if bbox:
                    self._draw_box(frame, bbox, (0, 255, 255))  # yellow while verifying
        else:
            if bbox:
                self._draw_box(frame, bbox, (0, 0, 255))  # red
            self.status_label.setText(
                f"Similarity {sim:.2f} below threshold {self.threshold:.2f}. Use PIN or add better photos."
            )
            self._stable_id = None
            self._stable_count = 0

        self._display_frame(frame)

    def _draw_box(self, frame_bgr, bbox, color=(0, 255, 0)):
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

    def _draw_verified_overlay(self, frame_bgr, bbox, name: str):
        # Green box + "Identity verified" centered above the box
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"Identity verified: {name}"
            # Place label slightly above the top-left corner of face box
            org = (x1, max(25, y1 - 10))
            cv2.putText(frame_bgr, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
        else:
            # fallback: center text if no bbox
            h, w = frame_bgr.shape[:2]
            org = (int(0.1 * w), int(0.1 * h))
            cv2.putText(frame_bgr, f"Identity verified: {name}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
        # Also reflect in the status label
        self.status_label.setText("Identity verified. Logging you in…")

    def _display_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        from PySide6.QtGui import QImage, QPixmap
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def _cleanup_camera(self):
        # Stop timer and release camera before accept()/reject()
        try:
            if self.timer.isActive():
                self.timer.stop()
        except Exception:
            pass
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

    def closeEvent(self, e):
        self._cleanup_camera()
        return super().closeEvent(e)

    def _build_admin_gallery(self, admin_dir: str) -> List[GalleryItem]:
        os.makedirs(admin_dir, exist_ok=True)
        items: List[GalleryItem] = []
        for path in sorted(glob.glob(os.path.join(admin_dir, "*.*"))):
            if os.path.splitext(path)[1].lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            emb, bbox, size = face_engine.get_face_embedding_bgr(img)
            if emb is None:
                continue
            name = os.path.splitext(os.path.basename(path))[0]
            items.append((name, name, None, emb))
        return items
