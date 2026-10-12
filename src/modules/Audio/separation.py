"""Separate vocals from audio.

Supports two backends:
- **demucs**: Facebook's Hybrid Transformer Demucs.
- **audio-separator** (default): Unified wrapper around Roformer, MDX-Net,
  VR Arch and Demucs models from the UVR ecosystem.  Roformer (BS-Roformer)
  is deterministic and produces higher SDR than htdemucs.
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
from modules.conversion_cache import content_key
from modules.DeviceDetection.device_detection import release_gpu_memory
from modules.os_helper import cap_name_length, check_file_exists

# File next to the stems: the content key of the audio they were made from
SOURCE_KEY_FILE = "source.txt"


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
    BS_ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"          # SDR 12.97 — best quality, deterministic
    MEL_BAND_ROFORMER = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"  # SDR 11.44 — good balance


# Default model for audio-separator backend
DEFAULT_AUDIO_SEPARATOR_MODEL = AudioSeparatorModel.MEL_BAND_ROFORMER


# ---------------------------------------------------------------------------
# Demucs backend
# ---------------------------------------------------------------------------

def _separate_with_demucs(
    input_file_path: str,
    output_folder: str,
    model: DemucsModel,
    device: str = "cpu",
    track_folder: str | None = None,
) -> None:
    """Separate vocals from audio with Demucs.

    The stems go to ``<output_folder>/separated/<model>/<track_folder>``, by
    default a folder named after the input file.
    """
    print(
        f"{ULTRASINGER_HEAD} Separating vocals from audio with "
        f"{blue_highlighted('demucs')} model {blue_highlighted(model.value)} "
        f"on {red_highlighted(device)}"
    )
    # Demucs fills in the names in braces of its file name pattern
    track = "{track}" if track_folder is None else track_folder.replace("{", "{{").replace("}", "}}")
    demucs.separate.main([
        "--two-stems", "vocals",
        "-d", device,
        "--float32",
        "-n", model.value,
        "--out", os.path.join(output_folder, "separated"),
        "--filename", f"{track}/{{stem}}.{{ext}}",
        input_file_path,
    ])
    # The next model (the transcription) gets the separation's GPU memory
    release_gpu_memory()
    print(f"{ULTRASINGER_HEAD} Vocal separation complete.")


# ---------------------------------------------------------------------------
# audio-separator backend
# ---------------------------------------------------------------------------

def create_separator(output_dir: str, force_cpu: bool = False):
    """An audio-separator ``Separator`` that writes WAV stems to ``output_dir``.

    The library picks its device itself: the GPU whenever PyTorch finds one
    (CUDA, or the MPS backend). With ``force_cpu`` (--force_cpu) it keeps to
    the CPU instead.

    Raises ``ImportError`` if ``audio-separator`` is not installed.
    """
    try:
        from audio_separator.separator import Separator  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "audio-separator is not installed. "
            "Install it with: uv add audio-separator"
        ) from exc

    separator_class = _cpu_separator_class(Separator) if force_cpu else Separator
    separator = separator_class(
        output_dir=output_dir,
        output_format="WAV",
        sample_rate=44100,
        normalization_threshold=0.9,
    )
    if force_cpu and getattr(getattr(separator, "torch_device", None), "type", None) != "cpu":
        # A version of the library that picks the device in another way
        raise RuntimeError("audio-separator did not keep to the CPU (--force_cpu): "
                           "it no longer picks its device with setup_torch_device")
    return separator


def _cpu_separator_class(separator_class):
    """``separator_class`` with the CPU in place of the device the library picks."""

    class CpuSeparator(separator_class):
        def setup_torch_device(self, system_info):
            # The constructor calls this to pick the device; the library's own
            # version takes the GPU whenever PyTorch finds one
            import torch

            self.torch_device_cpu = torch.device("cpu")
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CPUExecutionProvider"]

    return CpuSeparator


def _separate_with_audio_separator(
    input_file_path: str,
    output_dir: str,
    model: AudioSeparatorModel | str,
    force_cpu: bool = False,
) -> None:
    """Separate vocals using the audio-separator library.

    Raises ``ImportError`` if ``audio-separator`` is not installed.
    """
    model_name = model.value if isinstance(model, AudioSeparatorModel) else str(model)
    print(
        f"{ULTRASINGER_HEAD} Separating vocals from audio with "
        f"{blue_highlighted('audio-separator')} model "
        f"{blue_highlighted(model_name)}"
        + (f" on {red_highlighted('cpu')}" if force_cpu else "")
    )

    separator = create_separator(output_dir, force_cpu)
    try:
        separator.load_model(model_filename=model_name)
        separator.separate(
            input_file_path,
            custom_output_names={
                "Vocals": "vocals",
                "Instrumental": "no_vocals",
            },
        )
    finally:
        # Drop the model before going on: the next model (the transcription)
        # gets its GPU memory, which Whisper cannot take from PyTorch's cache
        separator = None
        release_gpu_memory()
    print(f"{ULTRASINGER_HEAD} Vocal separation complete.")


# ---------------------------------------------------------------------------
# Public API (backward-compatible)
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
    backend: SeparatorBackend = SeparatorBackend.AUDIO_SEPARATOR,
    force_cpu: bool = False,
) -> str:
    """Separate vocal from audio using the configured backend.

    Returns the directory containing ``vocals.wav`` and ``no_vocals.wav``.
    With ``force_cpu`` (--force_cpu) audio-separator runs on the CPU; Demucs
    runs on ``pytorch_device``.
    """
    # Resolve (backend, model) once so cache key and execution are consistent.
    # Accept raw strings so callers can pass custom model filenames.
    if backend == SeparatorBackend.AUDIO_SEPARATOR:
        resolved_model: DemucsModel | AudioSeparatorModel | str = (
            model if isinstance(model, (AudioSeparatorModel, str))
            else DEFAULT_AUDIO_SEPARATOR_MODEL
        )
    else:
        resolved_model = (
            model if isinstance(model, DemucsModel)
            else DemucsModel.HTDEMUCS
        )

    basename = os.path.splitext(os.path.basename(audio_output_file_path))[0]
    model_key = resolved_model.value if hasattr(resolved_model, "value") else str(resolved_model)
    # The audio file of a txt input keeps its own name, which can be longer than
    # the song names (see os_helper.MAX_NAME_CHARS)
    basename = cap_name_length(basename)
    audio_separation_path = os.path.join(
        cache_folder_path, "separated", model_key, basename,
    )

    vocals_path = os.path.join(audio_separation_path, "vocals.wav")
    instrumental_path = os.path.join(audio_separation_path, "no_vocals.wav")
    source_key_path = os.path.join(audio_separation_path, SOURCE_KEY_FILE)

    if use_separated_vocal or create_karaoke:
        # Stems are only reused when made from this audio: a cache folder
        # that is used again can hold stems of another file of the same name
        source_key = content_key(audio_output_file_path)
        cache_available = (
            check_file_exists(vocals_path)
            and check_file_exists(instrumental_path)
            and _read_source_key(source_key_path) == source_key
        )
        if skip_cache or not cache_available:
            os.makedirs(audio_separation_path, exist_ok=True)
            # Stems of an interrupted separation must never pass as made from this audio
            if os.path.exists(source_key_path):
                os.remove(source_key_path)

            if backend == SeparatorBackend.AUDIO_SEPARATOR:
                _separate_with_audio_separator(
                    input_file_path=audio_output_file_path,
                    output_dir=audio_separation_path,
                    model=resolved_model,
                    force_cpu=force_cpu,
                )
            else:
                _separate_with_demucs(
                    input_file_path=audio_output_file_path,
                    output_folder=cache_folder_path,
                    model=resolved_model,
                    device=pytorch_device,
                    track_folder=basename,
                )
            with open(source_key_path, "w", encoding="utf-8") as file:
                file.write(source_key)
        else:
            print(
                f"{ULTRASINGER_HEAD} "
                f"{green_highlighted('cache')} reusing cached separated vocals"
            )

    return audio_separation_path


def _read_source_key(path: str) -> str | None:
    """The content key in ``path``, or None when there is none."""
    try:
        with open(path, encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return None
