"""Device detection module."""

import logging
import os
import warnings

import torch

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted, blue_highlighted

logger = logging.getLogger(__name__)

pytorch_gpu_supported = False

# Collects non-deterministic operation warnings during a run.
# Each entry is the warning message string (deduplicated).
nondeterministic_warnings: list[str] = []

_original_showwarning = None


def _capture_deterministic_warning(message, category, filename, lineno, file=None, line=None):
    """Custom warning handler that captures deterministic-algorithm warnings."""
    msg_str = str(message)
    if "does not have a deterministic implementation" in msg_str or \
       "nondeterministic" in msg_str.lower():
        # Deduplicate: only store unique operation names
        # Extract the short form (first line / first 200 chars)
        short = msg_str.split("\n")[0][:200]
        if short not in nondeterministic_warnings:
            nondeterministic_warnings.append(short)
            logger.warning("Non-deterministic op detected: %s", short)
    # Always call the original handler so warnings still appear on stderr
    if _original_showwarning is not None:
        _original_showwarning(message, category, filename, lineno, file, line)


def check_gpu_support() -> str:
    """Check worker device (e.g cuda or cpu) supported by pytorch"""

    print(f"{ULTRASINGER_HEAD} Checking GPU support.")

    pytorch_gpu_supported = __check_pytorch_support()
    device = 'cuda' if pytorch_gpu_supported else 'cpu'

    if pytorch_gpu_supported:
        _enable_deterministic_mode()

    return device


def _enable_deterministic_mode():
    """Enable deterministic CUDA operations for reproducible results.

    Level 1: cuDNN deterministic convolutions (no crash risk).
    Level 2: Full PyTorch deterministic algorithms with warn_only=True
             to diagnose non-deterministic operations without crashing.

    Requires CUBLAS_WORKSPACE_CONFIG for cuBLAS matrix multiplications.
    Note: This does NOT affect CTranslate2 (faster-whisper) or ONNX
    Runtime (SwiftF0), which have their own CUDA kernels.
    """
    global _original_showwarning

    # Level 1: cuDNN — safe, no crash risk
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUBLAS workspace config — required for deterministic matmul
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Level 2: Full PyTorch deterministic mode (warn_only for diagnostics)
    # warn_only=True logs warnings instead of crashing on non-deterministic ops
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Don't fill uninitialized memory — unnecessary for inference, hurts perf
    torch.utils.deterministic.fill_uninitialized_memory = False

    # Install custom warning handler to capture non-deterministic op warnings
    _original_showwarning = warnings.showwarning
    warnings.showwarning = _capture_deterministic_warning

    print(
        f"{ULTRASINGER_HEAD} {blue_highlighted('Deterministic mode')} enabled "
        f"(cuDNN + PyTorch warn_only)."
    )
    logger.info(
        "Deterministic mode: cudnn.deterministic=True, "
        "cudnn.benchmark=False, use_deterministic_algorithms(warn_only=True), "
        "CUBLAS_WORKSPACE_CONFIG=%s",
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def __check_pytorch_support():
    pytorch_gpu_supported = torch.cuda.is_available()
    if not pytorch_gpu_supported:
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('pytorch')} - there are no {red_highlighted('cuda')} devices available -> Using {red_highlighted('cpu')}."
        )
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_vram = round(gpu_properties.total_memory / 1024 ** 3, 2)  # Convert bytes to GB and round to 2 decimal places
        print(f"{ULTRASINGER_HEAD} Found GPU: {blue_highlighted(gpu_name)} VRAM: {blue_highlighted(gpu_vram)} GB.")
        if gpu_vram < 6:
            print(
                f"{ULTRASINGER_HEAD} {red_highlighted('GPU VRAM is less than 6GB. Program may crash due to insufficient memory.')}")
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('pytorch')} - using {red_highlighted('cuda')} gpu.")
    return pytorch_gpu_supported
