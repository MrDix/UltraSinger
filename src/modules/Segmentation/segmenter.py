"""Pipeline step: replace word-based note segments with model-based ones."""

from __future__ import annotations

import os
from typing import NamedTuple

from modules.Midi.MidiSegment import MidiSegment
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, gold_highlighted

_MODEL_CACHE: dict = {}

# A note keeps its pitch in the pitch refinement (see MidiSegment.pitch_locked) when
# its pitch track held confident pitch in at least this share of the note: the game's
# pitch detection, which the refinement relies on, more often locks onto a harmonic
# (a fourth or fifth off) there than the pitch tracker does.
LOCK_VOICED_SHARE = 0.5


def _get_model(path: str, device: str):
    from modules.Segmentation.model import load_model

    key = (os.path.abspath(path), device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(path, device)
    return _MODEL_CACHE[key]


class ModelSegmentation(NamedTuple):
    segments: list[MidiSegment]
    pitch_audio_path: str  # the stem the note pitches were taken from (lead or full vocal)


class LeadStem(NamedTuple):
    path: str
    analysis: object  # VocalAnalysis
    reliable: bool    # kept most of the singing: note pitches come from it, else it only checks them


def _lead_pitch(vocal_analysis, vocals_path: str, cache_folder: str | None) -> LeadStem | None:
    """The lead-vocal stem, or ``None`` when it could not be made."""
    from modules.Segmentation.lead_vocal import MIN_VOICED_RATIO, choose_pitch_source, lead_vocal_analysis

    try:
        folder = cache_folder or os.path.dirname(vocals_path)
        lead_path, lead_analysis = lead_vocal_analysis(vocals_path, folder)
        lead, ratio = choose_pitch_source(vocal_analysis, lead_analysis)
    except Exception as e:  # noqa: BLE001 - lead pitch is an improvement, never a requirement
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} lead-vocal separation failed ({e!r}) "
              f"- note pitches from the full vocal")
        return None
    if lead is None:
        print(f"{ULTRASINGER_HEAD} Lead-vocal pitch not used: the lead stem keeps only {ratio:.0%} of the "
              f"singing (needs {MIN_VOICED_RATIO:.0%})")
        return LeadStem(lead_path, lead_analysis, False)
    print(f"{ULTRASINGER_HEAD} Note pitches from the lead vocal (keeps {ratio:.0%} of the singing)")
    return LeadStem(lead_path, lead, True)


def lock_pitches(segments: list[MidiSegment], notes) -> int:
    """Mark the segments whose pitch track covered them well (see ``MidiSegment.pitch_locked``)
    and give them the second track's pitch; ``segments`` correspond to ``notes`` one to one.
    Returns the number of locked segments."""
    locked = 0
    for seg, note in zip(segments, notes, strict=True):
        if note.freestyle:
            continue
        seg.pitch_locked = note.voiced >= LOCK_VOICED_SHARE
        seg.check_midi = note.check_midi
        locked += seg.pitch_locked
    return locked


def segment_with_model(
    midi_segments: list[MidiSegment],
    vocals_path: str,
    model_path: str,
    language: str | None,
    device: str = "cpu",
    lead_vocal_pitch: bool = False,
    cache_folder: str | None = None,
) -> ModelSegmentation | None:
    """Predict notes on the separated vocal and place the existing lyrics onto them.

    ``midi_segments`` only provides the words and their timing. Returns the new
    segments, or ``None`` when the step cannot run (missing model or vocal file,
    no notes found, any error) so the caller keeps the original segments.

    With ``lead_vocal_pitch`` the vocal stem is also split into lead and backing
    vocals (cached in ``cache_folder``); the lead stem's pitch is used for the
    notes when it kept most of the singing (see ``lead_vocal``), otherwise it only
    serves as a second opinion on the vocal stem's pitches. The result names the
    stem the pitches came from, so later steps that compare the notes with the
    singing can use the same one. Notes that their pitch track covers well keep
    their pitch in the later pitch refinement (see ``lock_pitches``).
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
        lead = _lead_pitch(analysis, vocals_path, cache_folder) if lead_vocal_pitch else None
        use_lead = lead is not None and lead.reliable
        pitch_audio_path = lead.path if use_lead else vocals_path
        notes = decode_notes(probs, onset, analysis,
                             pitch_analysis=lead.analysis if use_lead else None,
                             check_analysis=lead.analysis if lead is not None and not use_lead else None,
                             **decode_cfg)
        if not notes:
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} model found no notes - keeping word-based notes")
            return None
        syllables = syllables_from_segments(midi_segments, language)
        segments = place_lyrics(notes, syllables)
        if not segments:
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} lyrics could not be placed onto the "
                  f"model notes - keeping word-based notes")
            return None
        locked = lock_pitches(segments, notes)
        freestyle = sum(1 for s in segments if s.note_type == "F")
        print(f"{ULTRASINGER_HEAD} Model segmentation: {len(segments)} notes "
              f"({freestyle} freestyle, {locked} with a well-tracked pitch) from {len(syllables)} syllables")
        return ModelSegmentation(segments, pitch_audio_path)
    except Exception as e:  # noqa: BLE001 - fail open, never lose the chart
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} model segmentation failed ({e!r}) "
              f"- keeping word-based notes")
        return None
