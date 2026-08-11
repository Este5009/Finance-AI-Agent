# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller configuration shared by Windows and future macOS builds."""

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules


# PyInstaller exposes SPECPATH as the directory containing this specification,
# not as the specification filename itself.
ROOT = Path(SPECPATH).parent
streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all("streamlit")
reportlab_datas = collect_data_files("reportlab")
finance_agent_hiddenimports = collect_submodules("finance_agent")

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
    hiddenimports=streamlit_hiddenimports + finance_agent_hiddenimports + [
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

helper_analysis = Analysis(
    [str(ROOT / "packaging" / "streamlit_helper.py")],
    pathex=[str(ROOT)],
    binaries=streamlit_binaries,
    datas=datas,
    hiddenimports=streamlit_hiddenimports + finance_agent_hiddenimports + [
        "openpyxl",
        "pandas",
        "pypdf",
        "reportlab",
        "streamlit.web.cli",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "tests", "tkinter"],
    noarchive=False,
)
helper_pyz = PYZ(helper_analysis.pure)

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
helper_exe = EXE(
    helper_pyz,
    helper_analysis.scripts,
    [],
    exclude_binaries=True,
    name="Finance AI Agent Streamlit",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
collection = COLLECT(
    exe,
    helper_exe,
    analysis.binaries,
    analysis.datas,
    helper_analysis.binaries,
    helper_analysis.datas,
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
