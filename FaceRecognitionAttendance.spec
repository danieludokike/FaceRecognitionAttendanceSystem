# -*- mode: python ; coding: utf-8 -*-
import sys, os
from PyInstaller.utils.hooks import collect_submodules, collect_all

# ---- Collect PySide6 runtime (DLLs/plugins/styles/translations) defensively ----
pyside_bins, pyside_datas, pyside_hidden = [], [], []
try:
    bins, datas, hidden = collect_all('PySide6')
    pyside_bins += bins
    pyside_datas += datas
    pyside_hidden += hidden
except Exception:
    pass

# ---- Hidden imports for cv2 + PySide6 (some are loaded lazily) ----
hidden = list(set(
    collect_submodules('cv2') + [
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'PySide6.QtNetwork',
        'PySide6.QtSvg',
    ] + pyside_hidden
))

block_cipher = None

a = Analysis(
    ['app/main.py'],                  # entry point (relative to project root)
    pathex=['.', 'app'],              # include 'app' so 'services', 'ui' resolve
    binaries=pyside_bins,             # include PySide6 runtime DLLs/plugins
    datas=[
        # --- Bundle read-only resources ---
        ('app/app/models', 'app/app/models'),     # ONNX models (YuNet/SFace)

        # OPTIONAL: only if you still need packaged admin sample images.
        # Remove this line if you don't ship admins.
        ('app/data/admins', 'app/data/admins'),
        # DO NOT bundle students/attendance: they now live in %LOCALAPPDATA%.
    ] + pyside_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],                 # not required since paths are handled in code
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
    upx=True,                         # set False if your AV complains about UPX
    console=False,                    # GUI app
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
