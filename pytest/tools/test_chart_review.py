"""Tests for tools/chart_review.py: reading the chart and judging its lines
against a pitch track (the audio analysis and the pages are not part of them)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import chart_review  # noqa: E402
from chart_review import (  # noqa: E402
    ReviewChart, ReviewNote, chart_span, find_chart, find_vocals, format_report, load_chart, page_rows, review_lines,
)

# Word ends are spaces at the end of the note lines
CHART = "\n".join([
    "#TITLE:A Song",
    "#ARTIST:Some Band",
    "#BPM:300",
    "#GAP:1000",
    "#VOCALS:A Song [Vocals].ogg",
    ": 0 10 0 One" + " ",
    ": 10 10 2 two",
    ": 20 10 4 ~" + " ",
    "- 40",
    "F 40 20 0 shout" + " ",
    "* 60 20 12 hold" + " ",
    "E",
    ": 99 9 9 never" + " ",
]) + "\n"


def _write(folder, text=CHART, name="A Song.txt"):
    path = Path(folder) / name
    path.write_text(text, encoding="utf-8")
    return path


def _frames(*spans, end=4.0):
    """Pitch frames every 10 ms: spans of (start, end, MIDI pitch) are sung with a sure pitch."""
    times = np.round(np.arange(0.0, end, 0.01), 3)
    midi = np.zeros(len(times))
    voiced = np.zeros(len(times), dtype=bool)
    for start, stop, pitch in spans:
        inside = (times >= start) & (times < stop)
        midi[inside], voiced[inside] = pitch, True
    return times, midi, voiced


def _chart(*notes):
    return ReviewChart({}, [ReviewNote(*note) for note in notes])


class TestLoadChart(unittest.TestCase):
    def test_notes_with_their_times_and_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            chart = load_chart(_write(folder))
        # BPM 300: a beat is 60 / (300 * 4) = 0.05 s; GAP 1 s
        self.assertEqual([(n.kind, round(n.start, 2), round(n.end, 2), n.line) for n in chart.notes],
                         [(":", 1.0, 1.5, 0), (":", 1.5, 2.0, 0), (":", 2.0, 2.5, 0),
                          ("F", 3.0, 4.0, 1), ("*", 4.0, 5.0, 1)])
        self.assertEqual([n.word for n in chart.notes], ["One ", "two", "~ ", "shout ", "hold "])
        self.assertEqual(chart.notes[4].midi - chart.notes[0].midi, 12)
        self.assertEqual(chart.headers["ARTIST"], "Some Band")

    def test_several_spaces_between_the_numbers_of_a_note(self):
        with tempfile.TemporaryDirectory() as folder:
            text = "#BPM:300\n#GAP:0\n:  0   10  0 one" + " " + "\n: 10 10 2  two\nE\n"
            chart = load_chart(_write(folder, text))
        self.assertEqual([(n.word, round(n.start, 2), round(n.end, 2)) for n in chart.notes],
                         [("one ", 0.0, 0.5), (" two", 0.5, 1.0)])

    def test_tabs_between_the_parts_of_a_note(self):
        with tempfile.TemporaryDirectory() as folder:
            text = "#BPM:300\n#GAP:0\n:\t0\t10\t-3\tone\n*\t10\t10\t2\nE\n"
            chart = load_chart(_write(folder, text))
        self.assertEqual([(n.kind, n.word, round(n.start, 2)) for n in chart.notes],
                         [(":", "one", 0.0), ("*", "", 0.5)])

    def test_of_a_duet_the_notes_of_the_first_singer(self):
        with tempfile.TemporaryDirectory() as folder:
            text = "\n".join(["#BPM:300", "#GAP:0", "P1", ": 0 10 0 one", "- 12", ": 20 10 2 two",
                              "P2", ": 0 10 7 other", "- 12", ": 20 10 9 voice", "E"]) + "\n"
            chart = load_chart(_write(folder, text))
        self.assertEqual([(n.word, n.line) for n in chart.notes], [("one", 0), ("two", 1)])

    def test_the_chart_of_a_song_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            path = _write(folder)
            self.assertEqual(find_chart(Path(folder)), path)
            self.assertEqual(find_chart(path), path)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                find_chart(Path(folder))

    def test_the_vocal_of_the_chart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = _write(folder)
            headers = load_chart(path).headers
            self.assertIsNone(find_vocals(path, headers))  # named, but not there
            vocals = Path(folder) / "A Song [Vocals].ogg"
            vocals.write_bytes(b"")
            self.assertEqual(find_vocals(path, headers), vocals)
            self.assertIsNone(find_vocals(path, {}))


class TestReviewLines(unittest.TestCase):
    def test_a_line_sung_on_its_notes(self):
        chart = _chart((":", 0.0, 1.0, 60, "one ", 0), (":", 1.0, 2.0, 62, "two ", 0))
        (line,) = review_lines(chart, *_frames((0.0, 1.0, 60.2), (1.0, 2.0, 62.0)))
        self.assertEqual((line.text, line.notes, line.start, line.end), ("one two", 2, 0.0, 2.0))
        self.assertAlmostEqual(line.voiced_share, 1.0)
        self.assertAlmostEqual(line.on_pitch_share, 1.0)
        self.assertEqual(line.remarks, [])

    def test_octaves_are_folded(self):
        chart = _chart((":", 0.0, 1.0, 72, "high ", 0))
        (line,) = review_lines(chart, *_frames((0.0, 1.0, 48.0)))
        self.assertAlmostEqual(line.on_pitch_share, 1.0)

    def test_notes_beside_the_sung_pitch(self):
        chart = _chart((":", 0.0, 1.0, 60, "one ", 0), (":", 1.0, 2.0, 60, "two ", 0))
        (line,) = review_lines(chart, *_frames((0.0, 1.0, 60.0), (1.0, 2.0, 65.0)))
        self.assertAlmostEqual(line.on_pitch_share, 0.5)
        self.assertEqual(line.remarks, ["notes beside the sung pitch"])

    def test_a_line_with_little_sure_pitch_is_not_beside_it(self):
        chart = _chart((":", 0.0, 2.0, 60, "one ", 0))
        (line,) = review_lines(chart, *_frames((0.0, 0.3, 65.0)))
        self.assertAlmostEqual(line.voiced_share, 0.15)
        self.assertEqual(line.remarks, [])

    def test_scored_notes_without_a_pitch(self):
        chart = _chart(*((":", k * 0.5, k * 0.5 + 0.5, 60, f"w{k} ", 0) for k in range(4)))
        frames = _frames((0.0, 0.5, 60.0))
        (line,) = review_lines(chart, *frames, unpitched=[False, True, True, True])
        self.assertEqual(line.unpitched, 3)
        self.assertEqual(line.remarks, ["3 scored notes are sung without a pitch"])
        # one such note among many is no remark
        (line,) = review_lines(chart, *frames, unpitched=[False, False, False, True])
        self.assertEqual(line.remarks, [])

    def test_freestyle_notes_are_counted_and_not_judged(self):
        chart = _chart(("F", 0.0, 1.0, 60, "shout ", 0), (":", 1.0, 2.0, 60, "sing ", 0))
        (line,) = review_lines(chart, *_frames((0.0, 1.0, 66.0), (1.0, 2.0, 60.0)), unpitched=[True, False])
        self.assertEqual((line.notes, line.freestyle, line.unpitched), (1, 1, 0))
        self.assertAlmostEqual(line.on_pitch_share, 1.0)

    def test_sung_pitches_without_a_note(self):
        chart = _chart((":", 0.0, 1.0, 60, "run ", 0), (":", 1.0, 2.0, 62, "on ", 0))
        (line,) = review_lines(chart, *_frames((0.0, 2.0, 60.0)), left_out=[2, 1])
        self.assertEqual(line.left_out, 3)
        self.assertIn("3 sung pitches without a note", line.remarks)

    def test_singing_between_the_notes(self):
        chart = _chart((":", 0.0, 0.5, 60, "one ", 0), (":", 2.0, 2.5, 60, "two ", 0))
        (line,) = review_lines(chart, *_frames((0.0, 2.5, 60.0)))
        self.assertAlmostEqual(line.uncharted_s, 1.5, delta=0.02)
        self.assertEqual(line.remarks, ["1.5 s of singing between the notes"])

    def test_overlapping_notes_do_not_depend_on_their_order(self):
        """A frame within two notes is on pitch when it is on that of one of them."""
        low, high = (":", 0.0, 1.0, 60, "one ", 0), (":", 0.0, 1.0, 66, "two ", 0)
        frames = _frames((0.0, 1.0, 60.0))
        for order in ((low, high), (high, low)):
            with self.subTest(first=order[0][3]):
                (line,) = review_lines(_chart(*order), *frames)
                self.assertAlmostEqual(line.on_pitch_share, 1.0)
                self.assertEqual((line.start, line.end), (0.0, 1.0))

    def test_a_line_spans_all_its_notes(self):
        chart = _chart((":", 0.0, 3.0, 60, "long ", 0), (":", 1.0, 2.0, 60, "short ", 0))
        (line,) = review_lines(chart, *_frames((0.0, 3.0, 60.0)))
        self.assertEqual((line.start, line.end), (0.0, 3.0))
        self.assertAlmostEqual(line.uncharted_s, 0.0)

    def test_each_line_on_its_own(self):
        chart = _chart((":", 0.0, 1.0, 60, "first ", 0), (":", 1.0, 2.0, 60, "second ", 1))
        first, second = review_lines(chart, *_frames((0.0, 1.0, 60.0), (1.0, 2.0, 66.0)))
        self.assertEqual((first.line, second.line), (0, 1))
        self.assertEqual((first.remarks, second.remarks), ([], ["notes beside the sung pitch"]))


class TestFormatReport(unittest.TestCase):
    def test_the_lines_that_stand_out_come_first(self):
        chart = _chart((":", 0.0, 1.0, 60, "fine ", 0), (":", 61.0, 62.0, 60, "off | key ", 1))
        lines = review_lines(chart, *_frames((0.0, 1.0, 60.0), (61.0, 62.0, 66.0), end=63.0))
        report = format_report("Some Band - A Song", chart, lines, ["out/page_0000.png"])
        self.assertIn("# Chart review: Some Band - A Song", report)
        self.assertIn("## Lines that stand out (1 of 2)", report)
        self.assertIn('- 1:01.0 "off | key": notes beside the sung pitch', report)
        self.assertIn("| 0:00.0 | fine | 1 | 0 | 100 % | 100 % |  |", report)
        self.assertIn("| 1:01.0 | off / key | 1 | 0 | 100 % | 0 % | notes beside the sung pitch |", report)
        self.assertIn("- page_0000.png", report)

    def test_nothing_stands_out(self):
        chart = _chart((":", 0.0, 1.0, 60, "fine ", 0))
        report = format_report("A Song", chart, review_lines(chart, *_frames((0.0, 1.0, 60.0))), [])
        self.assertIn("## Lines that stand out (0 of 1)\n\nNone.", report)
        self.assertNotIn("## Pages", report)


class TestPages(unittest.TestCase):
    def test_the_pages_of_a_chart_show_all_its_notes(self):
        """Up to the end of the note that ends last, which need not be the last one."""
        chart = _chart((":", 0.0, 1.0, 60, "a ", 0), (":", 25.0, 32.0, 60, "long ", 0), (":", 26.0, 27.0, 62, "b ", 0))
        self.assertEqual(chart_span(chart), (0.0, 33.0))
        self.assertEqual(chart_span(_chart((":", 5.5, 6.0, 60, "a ", 0))), (4.5, 7.0))

    def test_rows_of_ten_seconds_three_to_a_page(self):
        self.assertEqual(page_rows(0.0, 45.0), [[(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)],
                                                [(30.0, 40.0), (40.0, 45.0)]])

    def test_nothing_after_the_last_second_asked_for(self):
        self.assertEqual(page_rows(60.0, 65.0), [[(60.0, 65.0)]])
        self.assertEqual(page_rows(60.0, 70.0), [[(60.0, 70.0)]])
        self.assertEqual(page_rows(60.0, 60.0), [])


class TestMain(unittest.TestCase):
    def test_a_chart_without_a_vocal_is_an_error(self):
        import contextlib
        import io
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(out):
            code = chart_review.main([str(_write(folder)), str(Path(folder) / "out")])
        self.assertEqual(code, 1)
        self.assertIn("names no separated vocal", out.getvalue())


if __name__ == "__main__":
    unittest.main()
