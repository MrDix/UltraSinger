"""Separate vocals from audio.

Supports two backends:
- **demucs** (default): Facebook's Hybrid Transformer Demucs.
- **audio-separator**: Unified wrapper around Roformer, MDX-Net, VR Arch and
  Demucs models from the UVR ecosystem.  Install with
  ``pip install audio-separator[cpu]`` (or ``[gpu]`` for CUDA).
"""

from __future__ import annotations

import os
from enum import Enum

import demucs.separate

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    green_highlighted,
    red_highlighted,
)
from modules.os_helper import check_file_exists


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

class SeparatorBackend(Enum):
    """Which vocal-separation library to use."""
    DEMUCS = "demucs"
    AUDIO_SEPARATOR = "audio_separator"


# ---------------------------------------------------------------------------
# Model enums
# ---------------------------------------------------------------------------

class DemucsModel(Enum):
    HTDEMUCS = "htdemucs"           # first version of Hybrid Transformer Demucs. Trained on MusDB + 800 songs. Default model.
    HTDEMUCS_FT = "htdemucs_ft"     # fine-tuned version of htdemucs, separation will take 4 times more time but might be a bit better. Same training set as htdemucs.
    HTDEMUCS_6S = "htdemucs_6s"     # 6 sources version of htdemucs, with piano and guitar being added as sources. Note that the piano source is not working great at the moment.
    HDEMUCS_MMI = "hdemucs_mmi"     # Hybrid Demucs v3, retrained on MusDB + 800 songs.
    MDX = "mdx"                     # trained only on MusDB HQ, winning model on track A at the MDX challenge.
    MDX_EXTRA = "mdx_extra"         # trained with extra training data (including MusDB test set), ranked 2nd on the track B of the MDX challenge.
    MDX_Q = "mdx_q"                 # quantized version of the previous models. Smaller download and storage but quality can be slightly worse.
    MDX_EXTRA_Q = "mdx_extra_q"     # quantized version of mdx_extra. Smaller download and storage but quality can be slightly worse.
    SIG = "SIG"                     # Placeholder for a single model from the model zoo.


class AudioSeparatorModel(Enum):
    """Curated model presets for audio-separator.

    The value is the exact filename expected by
    ``audio_separator.separator.Separator.load_model()``.
    Models are auto-downloaded on first use.
    """
    # Roformer — current state-of-the-art for vocal separation
    BS_ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"          # SDR 12.97 — best quality
    MEL_BAND_ROFORMER = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"  # SDR 11.44 — good balance
    # MDX-Net — faster, slightly lower quality
    KIM_VOCAL_2 = "Kim_Vocal_2.onnx"                                   # ONNX, very fast
    KUIELAB_A_VOCALS = "kuielab_a_vocals.onnx"                         # ONNX, lightweight
    # VR Architecture
    HP_VOCAL_UVR = "4_HP-Vocal-UVR.pth"                                # VR arch, solid quality


# Default model for audio-separator backend
DEFAULT_AUDIO_SEPARATOR_MODEL = AudioSeparatorModel.BS_ROFORMER


# ---------------------------------------------------------------------------
# Demucs backend
# ---------------------------------------------------------------------------

def _separate_with_demucs(
    input_file_path: str,
    output_folder: str,
    model: DemucsModel,
    device: str = "cpu",
) -> None:
    """Separate vocals from audio with Demucs."""
    print(
        f"{ULTRASINGER_HEAD} Separating vocals from audio with "
        f"{blue_highlighted('demucs')} model {blue_highlighted(model.value)} "
        f"on {red_highlighted(device)}"
    )
    demucs.separate.main([
        "--two-stems", "vocals",
        "-d", device,
        "--float32",
        "-n", model.value,
        "--out", os.path.join(output_folder, "separated"),
        input_file_path,
    ])


# ---------------------------------------------------------------------------
# audio-separator backend
# ---------------------------------------------------------------------------

def _separate_with_audio_separator(
    input_file_path: str,
    output_dir: str,
    model: AudioSeparatorModel | str,
    model_file_dir: str | None = None,
) -> None:
    """Separate vocals using the audio-separator library.

    Raises ``ImportError`` if ``audio-separator`` is not installed.
    """
    try:
        from audio_separator.separator import Separator  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "audio-separator is not installed. "
            "Install it with: pip install \"audio-separator[cpu]\" "
            "(or \"audio-separator[gpu]\" for CUDA support)"
        ) from exc

    model_name = model.value if isinstance(model, AudioSeparatorModel) else str(model)
    print(
        f"{ULTRASINGER_HEAD} Separating vocals from audio with "
        f"{blue_highlighted('audio-separator')} model "
        f"{blue_highlighted(model_name)}"
    )

    separator_kwargs: dict = {
        "output_dir": output_dir,
        "output_format": "WAV",
        "sample_rate": 44100,
        "normalization_threshold": 0.9,
    }
    if model_file_dir:
        separator_kwargs["model_file_dir"] = model_file_dir

    separator = Separator(**separator_kwargs)
    separator.load_model(model_filename=model_name)
    separator.separate(
        input_file_path,
        custom_output_names={
            "Vocals": "vocals",
            "Instrumental": "no_vocals",
        },
    )


# ---------------------------------------------------------------------------
# Public API  (backward-compatible)
# ---------------------------------------------------------------------------

# Keep the old function name as a thin wrapper so existing imports still work.
def separate_audio(
    input_file_path: str,
    output_folder: str,
    model: DemucsModel,
    device: str = "cpu",
) -> None:
    """Separate vocals from audio with Demucs (legacy wrapper)."""
    _separate_with_demucs(input_file_path, output_folder, model, device)


def separate_vocal_from_audio(
    cache_folder_path: str,
    audio_output_file_path: str,
    use_separated_vocal: bool,
    create_karaoke: bool,
    pytorch_device: str,
    model: DemucsModel | AudioSeparatorModel | str = DemucsModel.HTDEMUCS,
    skip_cache: bool = False,
    backend: SeparatorBackend = SeparatorBackend.DEMUCS,
) -> str:
    """Separate vocal from audio using the configured backend.

    Returns the directory containing ``vocals.wav`` and ``no_vocals.wav``.
    """
    basename = os.path.splitext(os.path.basename(audio_output_file_path))[0]
    model_key = model.value if hasattr(model, "value") else str(model)
    audio_separation_path = os.path.join(
        cache_folder_path, "separated", model_key, basename,
    )

    vocals_path = os.path.join(audio_separation_path, "vocals.wav")
    instrumental_path = os.path.join(audio_separation_path, "no_vocals.wav")

    if use_separated_vocal or create_karaoke:
        cache_available = (
            check_file_exists(vocals_path)
            and check_file_exists(instrumental_path)
        )
        if skip_cache or not cache_available:
            os.makedirs(audio_separation_path, exist_ok=True)

            if backend == SeparatorBackend.AUDIO_SEPARATOR:
                _separate_with_audio_separator(
                    input_file_path=audio_output_file_path,
                    output_dir=audio_separation_path,
                    model=model if isinstance(model, AudioSeparatorModel)
                    else DEFAULT_AUDIO_SEPARATOR_MODEL,
                )
            else:
                _separate_with_demucs(
                    input_file_path=audio_output_file_path,
                    output_folder=cache_folder_path,
                    model=model if isinstance(model, DemucsModel)
                    else DemucsModel.HTDEMUCS,
                    device=pytorch_device,
                )
        else:
            print(
                f"{ULTRASINGER_HEAD} "
                f"{green_highlighted('cache')} reusing cached separated vocals"
            )

    return audio_separation_path
