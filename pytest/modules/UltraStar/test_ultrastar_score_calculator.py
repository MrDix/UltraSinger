"""Tests for the internal (fallback) score of ultrastar_score_calculator.py"""

import os
import tempfile
import unittest
from unittest.mock import patch

import librosa

from modules.Midi.MidiSegment import MidiSegment
from modules.Ultrastar.coverter.ultrastar_converter import ultrastar_note_to_midi_note
from modules.Ultrastar.ultrastar_parser import parse
from modules.Ultrastar.ultrastar_score_calculator import calculate_score

# 300 #BPM: a beat is 0.05 s, so each 20-beat note lasts 1 s (11 parts of 0.09 s)
CHART = """#TITLE:Title
#ARTIST:Artist
#BPM:300
#GAP:0
: 0 20 5 one
* 20 20 7 gold
F 40 20 0 free
: 60 20 9 two
E
"""


def _chart():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "song.txt")
        with open(path, "w", encoding="utf-8") as file:
            file.write(CHART)
        return parse(path)


def _singing(chart, hit_words):
    """Pitch detection that hears the charted pitch in the notes of ``hit_words`` and a wrong one elsewhere."""

    def note_at(start, end, pitched_data, word):
        for note_line in chart.UltrastarNoteLines:
            if note_line.word == word:
                pitch = note_line.pitch if word in hit_words else note_line.pitch + 1
                return MidiSegment(librosa.midi_to_note(ultrastar_note_to_midi_note(pitch)), start, end, word)
        raise AssertionError(word)

    return patch("modules.Ultrastar.ultrastar_score_calculator.create_midi_note_from_pitched_data",
                 side_effect=note_at)


class TestCalculateScore(unittest.TestCase):
    def test_perfect_singing_gives_the_full_score(self):
        chart = _chart()
        with _singing(chart, {"one", "gold", "two"}):
            simple, accurate = calculate_score(None, chart)
        for score in (simple, accurate):
            self.assertEqual(score.score, 10000)
            self.assertEqual(score.line_bonus, 1000)
            # 11 parts per note, a golden part counts twice: 9000 * 22 / 44 each
            self.assertEqual(score.notes, 4500)
            self.assertEqual(score.golden, 4500)

    def test_points_in_proportion_to_the_weighted_parts(self):
        chart = _chart()
        with _singing(chart, {"one"}):
            simple, _ = calculate_score(None, chart)
        self.assertEqual(simple.notes, 2250)  # 9000 * 11 / 44
        self.assertEqual(simple.golden, 0)
        self.assertEqual(simple.line_bonus, 333)  # one of the three scored notes
        self.assertEqual(simple.score, 2583)


if __name__ == "__main__":
    unittest.main()
