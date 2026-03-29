from dataclasses import dataclass, field


@dataclass
class MidiSegment:
  note: str
  start: float
  end: float
  word: str
  note_type: str = ":"          # UltraStar note type: ":" normal, "F" freestyle, "*" golden, "R" rap
  line_break_after: bool = False  # When True, writer emits a linebreak after this note
