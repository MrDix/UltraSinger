from dataclasses import dataclass, field


@dataclass
class MidiSegment:
  note: str
  start: float
  end: float
  word: str
  note_type: str = ":"          # UltraStar note type: ":" normal, "F" freestyle, "*" golden, "R" rap
  line_break_after: bool = False  # When True, writer emits a linebreak after this note
  # Set by model-based segmentation when a pitch track held confident pitch in most of
  # the note: the pitch refinement then only moves the note to ``check_midi``, the
  # pitch a second, independent pitch track measured for it (None: none there).
  pitch_locked: bool = False
  check_midi: int | None = None
  # Set by the pitch refinement on a note it moved to the tone the games' pitch
  # detection hears in it: the pitch context step, which takes a note from the
  # sung pitch to the pitch a hand-made chart would write, leaves such a note
  # alone. The pieces a later step cuts the note into carry it along.
  pitch_refined: bool = False
