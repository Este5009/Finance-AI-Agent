"""Platform-aware resource and writable application-data paths."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


APP_DIRECTORY_NAME = "FinanceAIAgent"
APP_DATA_ENV = "FINANCE_AI_APP_DATA"
RESOURCE_ROOT_ENV = "FINANCE_AI_RESOURCE_ROOT"


@dataclass(frozen=True)
class AppDataPaths:
    """Writable directories used by an installed Finance AI Agent.

    Inputs: one platform-specific application-data root.
    Outputs: stable paths for database, uploads, reports, cache, logs, config,
    outputs, and runtime state.
    Assumptions: package resources are read-only and never stored below root.
    """

    root: Path
    database: Path
    imports: Path
    reports: Path
    cache: Path
    logs: Path
    config: Path
    runtime: Path
    outputs: Path

    def initialize(self) -> "AppDataPaths":
        """Create all writable directories without replacing existing data.

        Inputs: this path collection.
        Outputs: this same collection after idempotent directory creation.
        Assumptions: existing databases and configuration must survive upgrades.
        """

        for path in (
            self.root,
            self.database,
            self.imports,
            self.reports,
            self.cache,
            self.logs,
            self.config,
            self.runtime,
            self.outputs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def memory_database(self) -> Path:
        """Return the production SQLite memory path.

        Inputs: this path collection.
        Outputs: SQLite file below the persistent database directory.
        Assumptions: SQLite creates the file only when first opened.
        """

        return self.database / "finance_memory.db"

    @property
    def config_file(self) -> Path:
        """Return the user-editable desktop JSON configuration path."""

        return self.config / "config.json"

    @property
    def log_file(self) -> Path:
        """Return the rotating launcher log path."""

        return self.logs / "finance-ai-agent.log"

    @property
    def streamlit_log_file(self) -> Path:
        """Return the owned Streamlit helper stdout/stderr log path."""

        return self.logs / "finance-ai-agent-streamlit.log"

    @property
    def active_session_file(self) -> Path:
        """Return the replaceable runtime-session ownership record."""

        return self.runtime / "active-session.json"

    @property
    def launcher_lock_file(self) -> Path:
        """Return the single-launcher ownership record used during startup."""

        return self.runtime / "launcher.lock"


def app_data_root(
    *,
    platform: str | None = None,
    environ: dict[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve the writable production root for Windows or macOS.

    Inputs: optional platform/environment/home overrides for deterministic tests.
    Outputs: platform-aware FinanceAIAgent directory.
    Assumptions: Linux is supported for development using XDG data conventions.
    """

    environment = os.environ if environ is None else environ
    override = str(environment.get(APP_DATA_ENV, "")).strip()
    if override:
        return Path(override).expanduser().resolve()
    active_platform = platform or sys.platform
    user_home = Path.home() if home is None else Path(home)
    if active_platform.startswith("win"):
        local_app_data = environment.get("LOCALAPPDATA")
        if not local_app_data:
            local_app_data = str(user_home / "AppData" / "Local")
        return Path(local_app_data) / APP_DIRECTORY_NAME
    if active_platform == "darwin":
        return user_home / "Library" / "Application Support" / APP_DIRECTORY_NAME
    xdg_root = environment.get("XDG_DATA_HOME")
    return Path(xdg_root) / APP_DIRECTORY_NAME if xdg_root else user_home / ".local" / "share" / APP_DIRECTORY_NAME


def app_data_paths(**kwargs: object) -> AppDataPaths:
    """Build all application-data paths without touching the filesystem.

    Inputs: optional arguments accepted by :func:`app_data_root`.
    Outputs: immutable application-data path collection.
    Assumptions: callers explicitly call ``initialize`` before writing.
    """

    root = app_data_root(**kwargs)
    return AppDataPaths(
        root=root,
        database=root / "database",
        imports=root / "imports",
        reports=root / "reports",
        cache=root / "cache",
        logs=root / "logs",
        config=root / "config",
        runtime=root / "runtime",
        outputs=root / "runtime" / "outputs",
    )


def resource_root(*, frozen_root: str | Path | None = None) -> Path:
    """Resolve the read-only source/resource root in source and PyInstaller runs.

    Inputs: optional frozen-root override used by packaging tests.
    Outputs: absolute directory containing ``finance_agent`` and bundled data.
    Assumptions: PyInstaller exposes extracted resources through ``sys._MEIPASS``.
    """

    if frozen_root is not None:
        return Path(frozen_root).resolve()
    environment_root = os.environ.get(RESOURCE_ROOT_ENV, "").strip()
    if environment_root:
        return Path(environment_root).expanduser().resolve()
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root).resolve()
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str, frozen_root: str | Path | None = None) -> Path:
    """Return one bundled resource path independent of the current directory.

    Inputs: path components and an optional frozen-root override.
    Outputs: absolute resource path.
    Assumptions: callers validate existence when a resource is mandatory.
    """

    return resource_root(frozen_root=frozen_root).joinpath(*parts)
