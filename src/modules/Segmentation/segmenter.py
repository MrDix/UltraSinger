"""Pipeline step: replace word-based note segments with model-based ones."""

from __future__ import annotations

import os

from modules.Midi.MidiSegment import MidiSegment
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, gold_highlighted

_MODEL_CACHE: dict = {}


def _get_model(path: str, device: str):
    from modules.Segmentation.model import load_model

    key = (os.path.abspath(path), device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(path, device)
    return _MODEL_CACHE[key]


def _lead_pitch(vocal_analysis, vocals_path: str, cache_folder: str | None):
    """Lead-vocal analysis to take note pitches from, or ``None`` to use the vocal stem."""
    from modules.Segmentation.lead_vocal import MIN_VOICED_RATIO, choose_pitch_source, lead_vocal_analysis

    try:
        folder = cache_folder or os.path.dirname(vocals_path)
        lead, ratio = choose_pitch_source(vocal_analysis, lead_vocal_analysis(vocals_path, folder))
    except Exception as e:  # noqa: BLE001 - lead pitch is an improvement, never a requirement
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} lead-vocal separation failed ({e!r}) "
              f"- note pitches from the full vocal")
        return None
    if lead is None:
        print(f"{ULTRASINGER_HEAD} Lead-vocal pitch not used: the lead stem keeps only {ratio:.0%} of the "
              f"singing (needs {MIN_VOICED_RATIO:.0%})")
    else:
        print(f"{ULTRASINGER_HEAD} Note pitches from the lead vocal (keeps {ratio:.0%} of the singing)")
    return lead


def segment_with_model(
    midi_segments: list[MidiSegment],
    vocals_path: str,
    model_path: str,
    language: str | None,
    device: str = "cpu",
    lead_vocal_pitch: bool = False,
    cache_folder: str | None = None,
) -> list[MidiSegment] | None:
    """Predict notes on the separated vocal and place the existing lyrics onto them.

    ``midi_segments`` only provides the words and their timing. Returns the new
    segments, or ``None`` when the step cannot run (missing model or vocal file,
    no notes found, any error) so the caller keeps the original segments.

    With ``lead_vocal_pitch`` the vocal stem is also split into lead and backing
    vocals (cached in ``cache_folder``); the lead stem's pitch is used for the
    notes when it kept most of the singing (see ``lead_vocal``).
    """
    if not model_path or not os.path.isfile(model_path):
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} segmentation model not found: "
              f"{blue_highlighted(str(model_path))} - keeping word-based notes")
        return None
    if not vocals_path or not os.path.isfile(vocals_path):
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} no separated vocal for model "
              f"segmentation - keeping word-based notes")
        return None
    try:
        from modules.Segmentation.decode import decode_notes
        from modules.Segmentation.features import analyse_vocal, load_vocal, model_input
        from modules.Segmentation.lyrics import place_lyrics, syllables_from_segments
        from modules.Segmentation.model import predict

        print(f"{ULTRASINGER_HEAD} Segmenting notes with model {blue_highlighted(os.path.basename(model_path))}")
        model, decode_cfg = _get_model(model_path, device)
        analysis = analyse_vocal(load_vocal(vocals_path))
        probs, onset = predict(model, model_input(analysis), device)
        pitch_analysis = _lead_pitch(analysis, vocals_path, cache_folder) if lead_vocal_pitch else None
        notes = decode_notes(probs, onset, analysis, pitch_analysis=pitch_analysis, **decode_cfg)
        if not notes:
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} model found no notes - keeping word-based notes")
            return None
        syllables = syllables_from_segments(midi_segments, language)
        segments = place_lyrics(notes, syllables)
        if not segments:
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} lyrics could not be placed onto the "
                  f"model notes - keeping word-based notes")
            return None
        freestyle = sum(1 for s in segments if s.note_type == "F")
        print(f"{ULTRASINGER_HEAD} Model segmentation: {len(segments)} notes "
              f"({freestyle} freestyle) from {len(syllables)} syllables")
        return segments
    except Exception as e:  # noqa: BLE001 - fail open, never lose the chart
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} model segmentation failed ({e!r}) "
              f"- keeping word-based notes")
        return None
