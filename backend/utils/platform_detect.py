"""
Platform detection for backend selection.
"""

import platform
from typing import Literal


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (arm64 macOS).
    
    Returns:
        True if on Apple Silicon, False otherwise
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_backend_type() -> Literal["mlx", "pytorch"]:
    """
    Detect the best backend for the current platform.

    On macOS Apple Silicon, always uses MLX. Falls back to ``"pytorch"``
    only as a safety valve when MLX cannot be imported at all (e.g. Docker
    container, very old macOS, or PyInstaller bundle with missing native
    libs). All TTS engines are expected to use MLX; the pytorch fallback
    exists solely to prevent a hard crash on unsupported platforms.
    """
    if is_apple_silicon():
        try:
            import mlx.core as mx
            if not mx.metal.is_available():
                raise RuntimeError("MLX Metal not available")
            return "mlx"
        except (ImportError, OSError, RuntimeError):
            return "pytorch"
    return "pytorch"
