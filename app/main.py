# app/main.py
import os
# Force OpenCV to prefer DirectShow over MSMF (reduces MSMF warnings on Windows)
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # optional: quiet noisy warnings

import sys
from PySide6.QtWidgets import QApplication, QDialog   # <-- add QDialog
from services import db
from ui.login_dialog import LoginDialog
from ui.main_window import MainWindow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STUDENTS_DIR = os.path.join(DATA_DIR, "students")
ATTEND_DIR = os.path.join(DATA_DIR, "attendance")
ADMIN_DIR = os.path.join(DATA_DIR, "admins")   # <-- add this
DB_PATH = os.path.join(DATA_DIR, "embeddings.db")

def ensure_dirs():
    os.makedirs(STUDENTS_DIR, exist_ok=True)
    os.makedirs(ATTEND_DIR, exist_ok=True)
    os.makedirs(ADMIN_DIR, exist_ok=True)      # <-- add this


def main():
    ensure_dirs()
    db.init_db(DB_PATH)

    app = QApplication(sys.argv)
    app.setApplicationName("FaceRecognition Attendance System")

    login = LoginDialog(db_path=DB_PATH, admin_dir=ADMIN_DIR)  # <-- pass admin_dir to LoginDialog
    result = login.exec()
    if result == QDialog.DialogCode.Accepted:    # <-- use the enum explicitly
        login.deleteLater()                      # <-- ensure camera/timer fully released
        mw = MainWindow(db_path=DB_PATH, students_dir=STUDENTS_DIR, attendance_dir=ATTEND_DIR)
        mw.show()
        app.mw = mw                              # <-- keep a strong ref; prevents GC closing the window
        sys.exit(app.exec())
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
