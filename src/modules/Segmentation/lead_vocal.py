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

import os

from modules.Segmentation.features import VOICED_CONFIDENCE, VocalAnalysis

KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
# The lead stem must keep at least this share of the vocal stem's voiced time
MIN_VOICED_RATIO = 0.8


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


def separate_lead_vocal(vocals_path: str, cache_folder: str, model: str = KARAOKE_MODEL) -> str:
    """Path of the lead-vocal stem of ``vocals_path`` (separated once, then cached)."""
    out_dir = os.path.join(cache_folder, "lead_vocal", os.path.splitext(model)[0])
    lead_path = os.path.join(out_dir, "lead.wav")
    if os.path.isfile(lead_path):
        return lead_path
    from audio_separator.separator import Separator  # type: ignore[import-untyped]

    os.makedirs(out_dir, exist_ok=True)
    separator = Separator(output_dir=out_dir, output_format="WAV", sample_rate=44100,
                          normalization_threshold=0.9)
    separator.load_model(model_filename=model)
    separator.separate(vocals_path, custom_output_names={"Vocals": "lead", "Instrumental": "backing"})
    if not os.path.isfile(lead_path):
        raise FileNotFoundError(f"lead-vocal separation produced no {lead_path}")
    return lead_path


def lead_vocal_analysis(vocals_path: str, cache_folder: str) -> VocalAnalysis:
    """Pitch/feature analysis of the lead-vocal stem."""
    from modules.Segmentation.features import analyse_vocal, load_vocal

    return analyse_vocal(load_vocal(separate_lead_vocal(vocals_path, cache_folder)))
