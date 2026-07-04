# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).parent

APP_NAME = "HostComputer6"
ENTRY_SCRIPT = PROJECT_ROOT / "main.py"

SOURCE_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "View",
    PROJECT_ROOT / "Algorithm",
    PROJECT_ROOT / "Controller",
]

datas = []
main_window_ui = PROJECT_ROOT / "View" / "MainWindow.ui"
if main_window_ui.exists():
    datas.append((str(main_window_ui), "View"))

hiddenimports = [
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
    "serial",
    "serial.tools.list_ports",
    "sounddevice",
    "scipy.fft",
    "scipy.integrate",
    "scipy.signal",
    "pandas",
    "openpyxl",
    "matplotlib",
    "matplotlib.backends.backend_qt5agg",
]


a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(path) for path in SOURCE_DIRS],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt6",
        "PySide2",
        "PySide6",
        "tkinter",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
