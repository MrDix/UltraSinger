"""Lead-vocal pitch for model-based segmentation.

Backing vocals, harmonies and duet parts make the pitch tracker follow
whichever voice is loudest in the vocal stem - often not the melody that is
charted. A karaoke separation model splits the vocal stem into lead and
backing; the lead stem's pitch is then used for the notes, but only when the
lead stem kept most of the singing: when the separation misjudges the melody
(e.g. unison or group singing) it removes large parts of it, and the full
vocal stem is the safer source.
"""

from __future__ import annotations

import hashlib
import os

from modules.console_colors import ULTRASINGER_HEAD
from modules.DeviceDetection.device_detection import free_gpu_memory, release_gpu_memory
from modules.Segmentation.features import VOICED_CONFIDENCE, VocalAnalysis

KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
# The lead stem must keep at least this share of the vocal stem's voiced time
MIN_VOICED_RATIO = 0.8
# Free GPU memory the karaoke separation needs: it peaked at 1.7 GB, plus
# room for PyTorch's memory cache. With much less, the driver moves it into
# shared system memory, where it ran some 30 times slower (49 instead of 2
# minutes).
MIN_FREE_GPU_BYTES = int(2.2 * 2 ** 30)


class NotEnoughGpuMemory(RuntimeError):
    """The GPU has too little free memory for the lead-vocal separation."""


def voiced_ratio(lead: VocalAnalysis, vocals: VocalAnalysis) -> float:
    """Voiced frames of ``lead`` relative to those of ``vocals``."""
    def voiced(a: VocalAnalysis) -> int:
        return int(((a.f0_conf >= VOICED_CONFIDENCE) & (a.f0_hz > 40)).sum())
    return voiced(lead) / max(voiced(vocals), 1)


def choose_pitch_source(vocals: VocalAnalysis, lead: VocalAnalysis | None,
                        min_ratio: float = MIN_VOICED_RATIO) -> tuple[VocalAnalysis | None, float]:
    """``(lead, ratio)`` when the lead stem is reliable, else ``(None, ratio)``."""
    if lead is None:
        return None, 0.0
    ratio = voiced_ratio(lead, vocals)
    return (lead if ratio >= min_ratio else None), ratio


def _source_key(path: str) -> str:
    """Short key of a file's identity (path, size, modification time)."""
    st = os.stat(path)
    return hashlib.sha256(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()[:16]


def separate_lead_vocal(vocals_path: str, cache_folder: str, model: str = KARAOKE_MODEL,
                        force_cpu: bool = False) -> str:
    """Path of the lead-vocal stem of ``vocals_path`` (separated once, then cached).

    The cache entry is keyed to the vocal file, so a cache folder shared by
    several songs never hands one song's lead stem to another. Raises
    ``NotEnoughGpuMemory`` instead of separating when the GPU has less than
    ``MIN_FREE_GPU_BYTES`` free, or its free memory cannot be read. With
    ``force_cpu`` (--force_cpu) it separates on the CPU.
    """
    out_dir = os.path.join(cache_folder, "lead_vocal", os.path.splitext(model)[0], _source_key(vocals_path))
    lead_path = os.path.join(out_dir, "lead.wav")
    if os.path.isfile(lead_path):
        return lead_path
    release_gpu_memory()
    try:
        # On the CPU the separation needs no GPU memory
        free = None if force_cpu else free_gpu_memory()
    except RuntimeError as e:
        # A GPU whose memory cannot be read may well be short of it
        raise NotEnoughGpuMemory(f"the free GPU memory could not be read ({e})") from e
    if free is not None:
        if free < MIN_FREE_GPU_BYTES:
            raise NotEnoughGpuMemory(f"only {free / 2 ** 30:.1f} GB of GPU memory free, the separation needs "
                                     f"about {MIN_FREE_GPU_BYTES / 2 ** 30:.1f} GB; with less it would run in "
                                     f"shared system memory and take very long")
        print(f"{ULTRASINGER_HEAD} Lead-vocal separation with {free / 2 ** 30:.1f} GB of GPU memory free")
    from modules.Audio.separation import create_separator

    os.makedirs(out_dir, exist_ok=True)
    separator = create_separator(out_dir, force_cpu)
    separator.load_model(model_filename=model)
    separator.separate(vocals_path, custom_output_names={"Vocals": "lead", "Instrumental": "backing"})
    if not os.path.isfile(lead_path):
        raise FileNotFoundError(f"lead-vocal separation produced no {lead_path}")
    return lead_path


def lead_vocal_analysis(vocals_path: str, cache_folder: str, force_cpu: bool = False) -> tuple[str, VocalAnalysis]:
    """Path and pitch/feature analysis of the lead-vocal stem."""
    from modules.Segmentation.features import analyse_vocal, load_vocal

    lead_path = separate_lead_vocal(vocals_path, cache_folder, force_cpu=force_cpu)
    return lead_path, analyse_vocal(load_vocal(lead_path))
