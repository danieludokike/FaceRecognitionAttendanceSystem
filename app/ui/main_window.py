# app/ui/main_window.py
import os
import csv
import cv2
from datetime import datetime
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QGuiApplication  # <-- added QGuiApplication for beep
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTabWidget,
    QFileDialog, QLineEdit, QMessageBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QSpinBox, QDialog, QSizePolicy, QApplication
)

from services import db
from services import face_engine


class MainWindow(QMainWindow):
    def __init__(self, db_path: str, students_dir: str, attendance_dir: str):
        super().__init__()
        self.setWindowTitle("FaceRecognition Attendance System — Admin Console")
        self.resize(1100, 760)

        self.db_path = db_path
        self.students_dir = students_dir
        self.attendance_dir = attendance_dir

        self.threshold = float(db.get_setting('cosine_threshold', '0.40'))
        self.consecutive = max(1, int(db.get_setting('consecutive_frames', '2')))

        self.tabs = QTabWidget()
        self.attendance_tab = AttendanceTab(attendance_dir=self.attendance_dir, threshold=self.threshold, consecutive=self.consecutive)
        self.enroll_tab = EnrollTab(students_dir=self.students_dir)
        self.logs_tab = LogsTab(attendance_dir=self.attendance_dir)
        self.settings_tab = SettingsTab()

        self.tabs.addTab(self.attendance_tab, "Attendance")
        self.tabs.addTab(self.enroll_tab, "Enroll Students")
        self.tabs.addTab(self.logs_tab, "Logs")
        self.tabs.addTab(self.settings_tab, "Settings")

        self.setCentralWidget(self.tabs)
        self.tabs.currentChanged.connect(self._handle_tab_change)

    def _handle_tab_change(self, idx: int):
        try:
            w = self.tabs.widget(idx)
            if w is self.attendance_tab:
                self.attendance_tab.reload_gallery()
                self.attendance_tab.resume()
                try:
                    self.enroll_tab.stop_preview()
                except Exception:
                    pass
            elif w is self.enroll_tab:
                self.attendance_tab.pause()
                self.enroll_tab.start_preview()
            else:
                self.attendance_tab.pause()
                try:
                    self.enroll_tab.stop_preview()
                except Exception:
                    pass
        except Exception as e:
            print("Tab switch error:", e)


class AttendanceTab(QWidget):
    def __init__(self, attendance_dir: str, threshold: float, consecutive: int):
        super().__init__()
        self.attendance_dir = attendance_dir
        self.threshold = threshold
        self.consecutive = max(1, consecutive)
        self.min_face_px = 110  # reject tiny faces for matching

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111; border-radius:16px; padding:8px;")
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.start_btn = QPushButton("Start Session")
        self.auto_mark_chk = QCheckBox("Auto Mark when recognized")
        self.auto_mark_chk.setChecked(True)

        self.mark_btn = QPushButton("Mark Now")
        self.mark_btn.setEnabled(False)

        # Info label
        self.info_label = QLabel(
            "Click 'Start Session' to create today’s CSV. Then look at the camera to auto-mark attendance."
        )
        self.info_label.setWordWrap(True)

        top_row = QHBoxLayout()
        top_row.addWidget(self.start_btn)
        top_row.addWidget(self.auto_mark_chk)
        top_row.addStretch(1)
        top_row.addWidget(self.mark_btn)

        root = QVBoxLayout(self)
        root.addLayout(top_row)
        root.addWidget(self.video_label, 1)
        root.addWidget(self.info_label)

        # State
        self.session_csv = None
        self.marked_ids = set()
        self.current_match = None
        self.face_present = False
        self._below_threshold = False
        self._stable_counter = 0
        self._last_candidate_id = None
        self._last_warned_duplicate_id = None  # <-- remember which ID we already warned about

        # Camera (prefer DirectShow on Windows)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Cannot open camera.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

        # Data
        self.gallery = db.all_student_embeddings()  # (id, name, matric, emb)

        # Signals
        self.start_btn.clicked.connect(self._start_session)
        self.mark_btn.clicked.connect(self._mark_current)

    def closeEvent(self, e):
        try:
            self.timer.stop()
            if self.cap and self.cap.isOpened():
                self.cap.release()
        finally:
            return super().closeEvent(e)

    def _start_session(self):
        ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.session_csv = os.path.join(self.attendance_dir, f'attendance_{ts}.csv')
        with open(self.session_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'student_id', 'full_name', 'matric', 'similarity'])
        self.marked_ids.clear()
        self._last_warned_duplicate_id = None
        self.info_label.setText(f"Session started → {os.path.basename(self.session_csv)}")
        self.mark_btn.setEnabled(True)

    def _mark_current(self):
        if not self.session_csv:
            QMessageBox.information(self, "No Session", "Click 'Start Session' first.")
            return
        if self.current_match:
            sid, name, matric, sim = self.current_match
            if sid in self.marked_ids:
                # Already marked → yell (beep + dialog) once per student per encounter
                QApplication.beep()
                QMessageBox.warning(self, "Already Marked",
                                    f"{name} has already been marked present in this session.")
                self.info_label.setText(f"Already marked: {name} ({matric or '-'})")
                return
            with open(self.session_csv, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([datetime.now().isoformat(sep=' '), sid, name, matric or '', f"{sim:.4f}"])
            self.marked_ids.add(sid)
            self.info_label.setText(f"Marked: {name} ({matric or '-'})  sim={sim:.2f}")
        else:
            if self.face_present and self._below_threshold:
                QMessageBox.critical(self, "NO matching ID",
                                     "Face detected but similarity is below threshold. "
                                     "Please enroll the student or adjust threshold in Settings.")
                self.info_label.setText("NO matching ID — below threshold.")
            else:
                QMessageBox.information(self, "No Face",
                                        "No face detected. Center the face and ensure good lighting.")

    def _tick(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Reset per-frame flags
        self.face_present = False
        self._below_threshold = False
        self.current_match = None

        emb, bbox, size = face_engine.get_face_embedding_bgr(frame)

        if emb is not None and bbox is not None and size is not None:
            self.face_present = True
            w, h = size
            if min(w, h) < self.min_face_px:
                self.info_label.setText("Face too small for accurate match. Move closer.")
            else:
                match = face_engine.best_match(emb, self.gallery)
                if match:
                    sid, name, matric, sim = match
                    # stability gate: same candidate ID across consecutive frames
                    if sid != self._last_candidate_id:
                        self._last_candidate_id = sid
                        self._stable_counter = 0
                        self._last_warned_duplicate_id = None  # reset duplicate warning when candidate changes

                    if sim >= self.threshold:
                        self._stable_counter += 1
                        if self._stable_counter >= self.consecutive:
                            # Stable recognition reached
                            if sid in self.marked_ids:
                                # Duplicate in current session → do not write again
                                msg = f"Already marked: {name} ({matric or '-'})"
                                self.info_label.setText(msg)
                                # Beep + warn once per re-encounter
                                if self._last_warned_duplicate_id != sid:
                                    QApplication.beep()
                                    QMessageBox.warning(self, "Already Marked",
                                                        f"{name} has already signed in this session.")
                                    self._last_warned_duplicate_id = sid
                                # Keep current_match set so manual 'Mark Now' can still see who it is,
                                # but it will also be blocked by duplicate guard.
                                self.current_match = (sid, name, matric, sim)
                            else:
                                # First time this session → write (if auto) or allow manual
                                self.current_match = (sid, name, matric, sim)
                                msg = f"Recognized: {name} ({matric or '-'})  sim={sim:.2f}"
                                if self.auto_mark_chk.isChecked():
                                    if self.session_csv:
                                        with open(self.session_csv, 'a', newline='', encoding='utf-8') as f:
                                            wcsv = csv.writer(f)
                                            wcsv.writerow([datetime.now().isoformat(sep=' '), sid, name, matric or '', f"{sim:.4f}"])
                                        self.marked_ids.add(sid)
                                        msg += "  → MARKED"
                                self.info_label.setText(msg)
                        else:
                            self.info_label.setText(f"Verifying… ({self._stable_counter}/{self.consecutive})")
                    else:
                        self._below_threshold = True
                        self.info_label.setText(f"NO matching ID (sim={sim:.2f} < {self.threshold}).")
                        self._stable_counter = 0
                else:
                    self._below_threshold = True
                    self.info_label.setText("NO matching ID — no enrolled students yet.")
                    self._stable_counter = 0
        else:
            self.info_label.setText("No face detected. Center your face.")
            self._stable_counter = 0
            self._last_candidate_id = None

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def pause(self):
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        if hasattr(self, "cap") and self.cap and self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass

    def resume(self):
        if not hasattr(self, "cap") or self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Camera Error", "Cannot open camera.")
                return
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(self, "timer") and not self.timer.isActive():
            self.timer.start(30)

    def reload_gallery(self):
        self.gallery = db.all_student_embeddings()


class EnrollTab(QWidget):
    def __init__(self, students_dir: str):
        super().__init__()
        self.students_dir = students_dir

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#111; border-radius:12px; padding:8px;")
        self.preview.setMinimumSize(480, 270)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.pick_btn = QPushButton("Upload Student Photo…")
        self.capture_btn = QPushButton("Capture from Camera")

        self.name_edit = QLineEdit()
        self.matric_edit = QLineEdit()
        self.save_btn = QPushButton("Save Student")

        form = QFormLayout()
        form.addRow("Full Name", self.name_edit)
        form.addRow("Matric (optional)", self.matric_edit)

        row = QHBoxLayout()
        row.addWidget(self.pick_btn)
        row.addWidget(self.capture_btn)
        row.addStretch(1)

        root = QVBoxLayout(self)
        root.addWidget(self.preview, 1)
        root.addLayout(row)
        root.addLayout(form)
        root.addWidget(self.save_btn)

        # Live preview state
        self.cap_enroll = None
        self.timer_enroll = QTimer(self)
        self.timer_enroll.timeout.connect(self._tick_preview)
        self._last_frame = None
        self._preview_running = False
        self._frozen = False

        self.image_path = None
        self.image_bgr = None

        self.pick_btn.clicked.connect(self._pick_photo)
        self.capture_btn.clicked.connect(self._capture_or_retake)
        self.save_btn.clicked.connect(self._save_student)

    def start_preview(self):
        if self._preview_running:
            return
        self.cap_enroll = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap_enroll.isOpened():
            self.cap_enroll = cv2.VideoCapture(0)
        if not self.cap_enroll.isOpened():
            QMessageBox.critical(self, "Camera Error", "Cannot open camera.")
            return
        self.cap_enroll.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap_enroll.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap_enroll.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._last_frame = None
        self._preview_running = True
        self._frozen = False
        self.capture_btn.setText("Capture from Camera")
        self.timer_enroll.start(30)

    def stop_preview(self):
        if self.timer_enroll.isActive():
            self.timer_enroll.stop()
        if self.cap_enroll is not None and self.cap_enroll.isOpened():
            try:
                self.cap_enroll.release()
            except Exception:
                pass
        self.cap_enroll = None
        self._preview_running = False

    def _tick_preview(self):
        if not self.cap_enroll:
            return
        ok, frame = self.cap_enroll.read()
        if not ok:
            return
        self._last_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _capture_or_retake(self):
        if self._preview_running:
            if self._last_frame is None:
                QMessageBox.information(self, "Hold on", "Camera is still starting—try again in a second.")
                return
            frame = self._last_frame.copy()
            self.stop_preview()
            self._set_preview(frame)
            self.image_path = None
            self._frozen = True
            self.capture_btn.setText("Retake")
        else:
            self.start_preview()

    def _pick_photo(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Student Photo", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        if self._preview_running:
            self.stop_preview()
        bgr = cv2.imread(path)
        if bgr is None:
            QMessageBox.warning(self, "Read Error", "Could not read image.")
            return
        self._set_preview(bgr)
        self.image_path = path
        self._frozen = True
        self.capture_btn.setText("Retake")

    def _set_preview(self, bgr):
        self.image_bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _save_student(self):
        name = self.name_edit.text().strip()
        matric = self.matric_edit.text().strip() or None
        if not name:
            QMessageBox.information(self, "Missing", "Enter full name.")
            return
        if self.image_bgr is None:
            QMessageBox.information(self, "No Photo", "Upload or capture a photo first.")
            return
        emb, bbox, size = face_engine.get_face_embedding_bgr(self.image_bgr)
        if emb is None:
            QMessageBox.warning(self, "No Face", "No face detected in the image. Use a clearer, front-facing image.")
            return
        if self.image_path:
            base = os.path.basename(self.image_path)
            dest = os.path.join(self.students_dir, base)
            if os.path.abspath(dest) != os.path.abspath(self.image_path):
                try:
                    import shutil
                    shutil.copy2(self.image_path, dest)
                except Exception as e:
                    QMessageBox.warning(self, "Copy Error", f"Could not copy image: {e}")
                    dest = self.image_path
        else:
            dest = os.path.join(self.students_dir, f"{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(dest, self.image_bgr)

        sid = db.add_student(full_name=name, matric=matric, embedding=emb, image_path=dest)
        QMessageBox.information(self, "Saved", f"Student saved: {name}\nID: {sid}")

        # Reset UI (keep camera stopped)
        self.name_edit.clear()
        self.matric_edit.clear()
        self.preview.clear()
        self.image_bgr = None
        self.image_path = None
        self._frozen = False
        self.capture_btn.setText("Capture from Camera")


class LogsTab(QWidget):
    def __init__(self, attendance_dir: str):
        super().__init__()
        self.attendance_dir = attendance_dir

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["File", "Opened Preview?"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)

        self.refresh_btn = QPushButton("Refresh")
        self.view_btn = QPushButton("View Selected…")
        self.open_folder_btn = QPushButton("Open Folder…")

        row = QHBoxLayout()
        row.addWidget(self.refresh_btn)
        row.addStretch(1)
        row.addWidget(self.view_btn)
        row.addWidget(self.open_folder_btn)

        root = QVBoxLayout(self)
        root.addLayout(row)
        root.addWidget(self.table, 1)

        self.refresh_btn.clicked.connect(self._refresh)
        self.view_btn.clicked.connect(self._view_selected)
        self.open_folder_btn.clicked.connect(self._open_folder)

        self._refresh()

    def _refresh(self):
        files = [f for f in os.listdir(self.attendance_dir) if f.lower().endswith('.csv')]
        files.sort(reverse=True)
        self.table.setRowCount(0)
        for f in files:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(f))
            self.table.setItem(r, 1, QTableWidgetItem(""))

    def _view_selected(self):
        r = self.table.currentRow()
        if r < 0:
            return
        fname = self.table.item(r, 0).text()
        path = os.path.join(self.attendance_dir, fname)
        if not os.path.exists(path):
            QMessageBox.warning(self, "Missing", "File no longer exists.")
            return
        dlg = CSVPreviewDialog(path, self)
        dlg.exec()
        self.table.setItem(r, 1, QTableWidgetItem("Yes"))

    def _open_folder(self):
        os.startfile(self.attendance_dir) if os.name == 'nt' else os.system(f'xdg-open "{self.attendance_dir}"')


class CSVPreviewDialog(QDialog):
    def __init__(self, csv_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(csv_path))
        self.resize(800, 500)
        import csv
        table = QTableWidget()
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if rows:
            headers = rows[0]
            data = rows[1:]
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setRowCount(len(data))
            for i, row in enumerate(data):
                for j, val in enumerate(row):
                    table.setItem(i, j, QTableWidgetItem(val))
        lay = QVBoxLayout(self)
        lay.addWidget(table)


class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.pin_edit = QLineEdit()
        self.pin_edit.setEchoMode(QLineEdit.Password)
        self.save_pin_btn = QPushButton("Save Admin PIN")

        self.thresh_spin = QDoubleSpinBox2(decimals=2, minimum=0.10, maximum=0.90, step=0.01)
        try:
            self.thresh_spin.setValue(float(db.get_setting('cosine_threshold', '0.40')))
        except:
            self.thresh_spin.setValue(0.40)

        self.cons_frames = QSpinBox()
        self.cons_frames.setRange(1, 10)
        try:
            self.cons_frames.setValue(int(db.get_setting('consecutive_frames', '2')))
        except:
            self.cons_frames.setValue(2)

        self.save_thresh_btn = QPushButton("Save Threshold / Confirmations")

        form = QFormLayout()
        form.addRow("Admin PIN", self.pin_edit)
        form.addRow("Cosine Threshold", self.thresh_spin)
        form.addRow("Consecutive Confirmations", self.cons_frames)

        row = QHBoxLayout()
        row.addWidget(self.save_pin_btn)
        row.addStretch(1)
        row.addWidget(self.save_thresh_btn)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addLayout(row)

        pin = db.get_setting('admin_pin', '1234')
        self.pin_edit.setText(pin)

        self.save_pin_btn.clicked.connect(self._save_pin)
        self.save_thresh_btn.clicked.connect(self._save_thresh)

    def _save_pin(self):
        pin = self.pin_edit.text().strip()
        if not pin:
            QMessageBox.information(self, "Missing", "Enter a PIN.")
            return
        db.set_setting('admin_pin', pin)
        QMessageBox.information(self, "Saved", "PIN updated.")

    def _save_thresh(self):
        t = float(self.thresh_spin.value())
        c = int(self.cons_frames.value())
        db.set_setting('cosine_threshold', str(t))
        db.set_setting('consecutive_frames', str(c))
        QMessageBox.information(self, "Saved", f"Threshold={t:.2f}, Confirmations={c}. Restart app to apply to login.")


class QDoubleSpinBox2(QSpinBox):
    # decimal spin using integer scaling (avoids locale issues)
    def __init__(self, decimals=2, minimum=0.0, maximum=1.0, step=0.01):
        super().__init__()
        self.decimals = decimals
        self.factor = 10 ** decimals
        self._min = int(minimum * self.factor)
        self._max = int(maximum * self.factor)
        self._step = int(step * self.factor)
        self.setRange(self._min, self._max)
        self.setSingleStep(self._step)
        self.setValue(int(0.40 * self.factor))

    def value(self) -> float:
        return super().value() / self.factor

    def setValue(self, v):
        if isinstance(v, float):
            v = int(round(v * self.factor))
        super().setValue(v)

    def textFromValue(self, v: int) -> str:
        return f"{v / self.factor:.{self.decimals}f}"

    def valueFromText(self, text: str) -> int:
        try:
            return int(round(float(text) * self.factor))
        except:
            return self._min
