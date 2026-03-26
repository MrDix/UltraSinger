from dataclasses import dataclass


@dataclass
class MidiSegment:
  note: str
  start: float
  end: float
  word: str
  note_type: str = ":"  # UltraStar note type: ":" normal, "F" freestyle
  line_break_after: bool = False  # LRCLIB line break follows this segment
