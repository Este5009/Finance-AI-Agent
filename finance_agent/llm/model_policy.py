"""Configurable hardware-aware policy for installed Ollama models."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from enum import Enum


class ModelTier(str, Enum):
    """User-facing latency/quality policy names."""

    QUALITY = "QUALITY"
    BALANCED = "BALANCED"
    FAST = "FAST"


@dataclass(frozen=True)
class HardwareProfile:
    """Non-sensitive local hardware facts relevant to model selection."""

    os_name: str
    architecture: str
    cpu: str
    total_memory_bytes: int | None
    apple_silicon: bool


def detect_hardware_profile() -> HardwareProfile:
    """Detect bounded cross-platform hardware metadata without external tools.

    Inputs: local OS environment.
    Outputs: HardwareProfile.
    Assumptions: unavailable memory information remains None rather than guessed.
    """

    memory: int | None = None
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                timeout=2, check=False,
            )
            memory = int(result.stdout.strip()) if result.stdout.strip().isdigit() else None
        except (OSError, ValueError, subprocess.SubprocessError):
            memory = None
    elif hasattr(os, "sysconf"):
        try:
            memory = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            memory = None
    architecture = platform.machine()
    return HardwareProfile(
        os_name=platform.system(), architecture=architecture,
        cpu=platform.processor() or "unknown", total_memory_bytes=memory,
        apple_silicon=platform.system() == "Darwin" and architecture == "arm64",
    )


def select_installed_model(
    *, tier: ModelTier, installed_models: list[str], configured_models: dict[ModelTier, str | None]
) -> str | None:
    """Select only a configured model that is already installed.

    Inputs: requested tier, Ollama inventory, and explicit tier-to-model configuration.
    Outputs: exact installed model name or None.
    Assumptions: the policy never invents names and never downloads models.
    """

    candidate = configured_models.get(tier)
    return candidate if candidate and candidate in installed_models else None
