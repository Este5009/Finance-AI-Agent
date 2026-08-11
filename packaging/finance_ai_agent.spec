# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller configuration shared by Windows and future macOS builds."""

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files


ROOT = Path(SPECPATH).parent.parent
streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all("streamlit")
reportlab_datas = collect_data_files("reportlab")

# Only code-owned resources are bundled. Runtime DBs, uploads, reports, caches,
# outputs, .env files, and university/synthetic data are intentionally absent.
datas = streamlit_datas + reportlab_datas + [
    (str(ROOT / "finance_agent" / "memory" / "schema.sql"), "finance_agent/memory"),
    (str(ROOT / "finance_agent" / "ui" / "streamlit_app.py"), "finance_agent/ui"),
]

analysis = Analysis(
    [str(ROOT / "packaging" / "desktop_entry.py")],
    pathex=[str(ROOT)],
    binaries=streamlit_binaries,
    datas=datas,
    hiddenimports=streamlit_hiddenimports + [
        "openpyxl",
        "pandas",
        "pypdf",
        "reportlab",
        "streamlit.web.cli",
        "tkinter",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "tests"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Finance AI Agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    version=str(ROOT / "packaging" / "version_info.txt") if sys.platform.startswith("win") else None,
)
collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name="Finance AI Agent",
)

if sys.platform == "darwin":
    app = BUNDLE(
        collection,
        name="Finance AI Agent.app",
        icon=None,
        bundle_identifier="com.financeaiagent.desktop",
        info_plist={"CFBundleShortVersionString": "0.1.0"},
    )
