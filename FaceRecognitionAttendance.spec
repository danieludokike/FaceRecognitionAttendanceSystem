# -*- mode: python ; coding: utf-8 -*-
import sys, os
from PyInstaller.utils.hooks import collect_submodules

# Hidden imports for cv2 and PySide6
hidden = collect_submodules('cv2') + [
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtCore',
    'PySide6.QtNetwork',
    'PySide6.QtSvg'
]

block_cipher = None

a = Analysis(
    ['app/main.py'],                      # entry point (relative to project root)
    pathex=['.', 'app'],                  # <-- add 'app' so 'services', 'ui' resolve
    binaries=[],
    datas=[
        ('app/app/models', 'app/app/models'),   # ONNX models
        ('app/data', 'app/data'),               # admins/students/attendance
    ],
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],                     # (optional) could add a path runtime hook here
    excludes=[],
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=False,
    name='FaceRecognitionAttendance',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                        # GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FaceRecognitionAttendance'
)
