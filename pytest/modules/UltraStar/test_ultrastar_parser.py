"""Tests for ultrastar_parser.py"""

import os
import tempfile
import unittest
from unittest.mock import patch

from modules.Ultrastar.ultrastar_parser import chart_encoding, parse, parse_ultrastar_txt
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue


def _write(folder: str, content: bytes, name: str = "song.txt") -> str:
    path = os.path.join(folder, name)
    with open(path, "wb") as file:
        file.write(content)
    return path


class TestUltraStarParser(unittest.TestCase):
    @patch("modules.Ultrastar.ultrastar_parser._fetch_musicbrainz_cover")
    @patch("modules.Ultrastar.ultrastar_parser.parse")
    def test_parse_ultrastar_txt(self, mock_parse, mock_fetch_cover):
        chart = UltrastarTxtValue()
        chart.mp3 = "test.mp3"
        chart.artist = "  Test Artist  "  # Also test leading and trailing whitespaces
        chart.title = "  Test Title  "
        mock_parse.return_value = chart

        with tempfile.TemporaryDirectory() as tmp:
            input_folder = os.path.join(tmp, "input")
            os.makedirs(input_folder)
            input_file = _write(input_folder, b"")
            _write(input_folder, b"audio", "test.mp3")
            output_folder = os.path.join(tmp, "output")

            result = parse_ultrastar_txt(input_file, output_folder)

            song_folder = os.path.join(output_folder, "Test Artist - Test Title")
            self.assertEqual(result, ("Test Artist - Test Title",
                                      song_folder,
                                      os.path.join(song_folder, "test.mp3"),
                                      chart,
                                      "mp3"))
            self.assertTrue(os.path.isfile(os.path.join(song_folder, "test.mp3")))
        mock_parse.assert_called_once_with(input_file)
        # no cover in the chart: the online lookup is tried (here without network)
        mock_fetch_cover.assert_called_once()

    def test_found_cover_is_named_like_the_song_folder(self):
        import io
        from types import SimpleNamespace

        from PIL import Image

        from modules.Ultrastar.ultrastar_parser import _fetch_musicbrainz_cover

        image = io.BytesIO()
        Image.new("RGB", (4, 3), "red").save(image, "PNG")
        chart = UltrastarTxtValue()
        chart.artist, chart.title = "Artist", "What?"
        found = SimpleNamespace(cover_image_data=image.getvalue(), cover_url="https://example.com/cover.jpg")
        with tempfile.TemporaryDirectory() as tmp, \
                patch("modules.musicbrainz_client.search_musicbrainz", return_value=found):
            _fetch_musicbrainz_cover(chart, tmp)
            self.assertEqual(chart.cover, "Artist - What [CO].jpg")
            self.assertTrue(os.path.isfile(os.path.join(tmp, chart.cover)))
        self.assertEqual(chart.coverUrl, "https://example.com/cover.jpg")


CHART = """#TITLE:Song Title: Part 2
#ARTIST:Artist
#MP3:Song.mp3
#AUDIOURL:https://example.com/audio
#VIDEO:Song.mp4
#VIDEOGAP:1.5
#VIDEOURL:https://example.com/video
#COVER:cover.jpg
#BPM:300
#GAP:1000
: 0 4 5 One
* 4 4 7 two
- 10
F 12 2 0 three
: 16 4 9 four
E
"""


class TestParse(unittest.TestCase):
    def _parse(self, content: bytes) -> UltrastarTxtValue:
        with tempfile.TemporaryDirectory() as tmp:
            return parse(_write(tmp, content))

    def test_tags_are_matched_exactly(self):
        chart = self._parse(CHART.encode("utf-8"))
        self.assertEqual(chart.title, "Song Title: Part 2")
        self.assertEqual(chart.audio, "")  # #AUDIOURL is not #AUDIO
        self.assertEqual(chart.video, "Song.mp4")  # not #VIDEOGAP or #VIDEOURL
        self.assertEqual(chart.videoGap, "1.5")
        self.assertEqual(chart.videoUrl, "https://example.com/video")
        self.assertEqual(chart.cover, "cover.jpg")
        self.assertEqual([n.word for n in chart.UltrastarNoteLines], ["One", "two", "three", "four"])
        # 300 #BPM = 1200 beats a minute: a beat is 0.05 s, after the #GAP of 1 s
        self.assertAlmostEqual(chart.UltrastarNoteLines[3].startTime, 1.8)

    def test_utf8_byte_order_mark_keeps_the_first_tag(self):
        chart = self._parse(b"\xef\xbb\xbf" + "#TITLE:Ça va\n#ARTIST:Artist\n".encode("utf-8"))
        self.assertEqual(chart.title, "Ça va")

    def test_cp1252_chart(self):
        chart = self._parse("#TITLE:Grüße\n#ARTIST:Künstler\n#BPM:300\n: 0 4 5 Grün\n".encode("cp1252"))
        self.assertEqual(chart.title, "Grüße")
        self.assertEqual(chart.UltrastarNoteLines[0].word, "Grün")

    def test_encoding_tag_is_respected(self):
        # CP1250 bytes: without the tag they would be read as CP1252 ("£ódŸ")
        content = "#ENCODING:CP1250\n#TITLE:Łódź\n".encode("cp1250")
        chart = self._parse(content)
        self.assertEqual(chart.title, "Łódź")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(chart_encoding(_write(tmp, content)), "cp1250")
            self.assertEqual(chart_encoding(_write(tmp, b"\xef\xbb\xbf#TITLE:x\n")), "utf-8-sig")
            self.assertEqual(chart_encoding(_write(tmp, "#TITLE:é\n".encode("utf-8"))), "utf-8")

    def test_utf8_chart_with_a_stale_encoding_tag(self):
        # re-saved in UTF-8 by an editor that kept the old tag
        chart = self._parse("#ENCODING:CP1252\n#TITLE:Grüße\n".encode("utf-8"))
        self.assertEqual(chart.title, "Grüße")

    def test_encoding_tag_that_is_no_text_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            for tag in ("rot13", "base64", "UTF-16", "no such codec"):
                content = f"#ENCODING:{tag}\n#TITLE:Grüße\n"
                self.assertEqual(chart_encoding(_write(tmp, content.encode("cp1252"))), "cp1252", tag)
                self.assertEqual(chart_encoding(_write(tmp, content.encode("utf-8"))), "utf-8", tag)
                self.assertEqual(chart_encoding(_write(tmp, f"#ENCODING:{tag}\n#TITLE:x\n".encode())), "utf-8", tag)
        self.assertEqual(self._parse("#ENCODING:rot13\n#TITLE:Grüße\n".encode("cp1252")).title, "Grüße")

    def test_missing_or_empty_gap_starts_at_zero(self):
        for gap_line in ("", "#GAP:\n", "#GAP: \n"):
            chart = self._parse(f"#BPM:300\n{gap_line}: 20 4 5 word\n".encode("utf-8"))
            self.assertAlmostEqual(chart.UltrastarNoteLines[0].startTime, 1.0)

    def test_relative_beats_are_made_absolute(self):
        relative = ("#BPM:300\n#GAP:0\n#RELATIVE:yes\n"
                    ": 0 4 5 one\n: 4 4 5 two\n- 10 12\n"
                    ": 0 4 5 three\n: 4 4 5 four\n- 8 10\n"
                    ": 2 4 5 five\nE\n")
        chart = self._parse(relative.encode("utf-8"))
        self.assertEqual([n.startBeat for n in chart.UltrastarNoteLines], [0, 4, 12, 16, 24])
        self.assertAlmostEqual(chart.UltrastarNoteLines[4].startTime, 24 * 0.05)

    def test_note_line_with_a_tab_after_its_type(self):
        chart = self._parse(b"#RELATIVE:YES\n#BPM:300\n: 0 4 5 one\n- 8\n:\t2 4 7 tab\n")
        self.assertEqual([(n.startBeat, n.pitch, n.word) for n in chart.UltrastarNoteLines],
                         [(0, 5, "one"), (10, 7, "tab")])

    def test_relative_line_break_that_is_not_a_number(self):
        # a break without a number does not move the line start; a second value
        # that is not a number counts as missing (the break moves the start)
        chart = self._parse(b"#RELATIVE:YES\n#BPM:300\n: 0 4 5 one\n- x\n: 2 4 5 two\n"
                            b"- 8 text\n: 1 4 5 three\n- nan\n: 3 4 5 four\n")
        self.assertEqual([n.startBeat for n in chart.UltrastarNoteLines], [0, 2, 9, 11])

    def test_line_breaks_of_absolute_charts_do_not_move_the_beats(self):
        chart = self._parse(b"#BPM:300\n: 0 4 5 one\n- 10 12\n: 20 4 5 two\n")
        self.assertEqual([n.startBeat for n in chart.UltrastarNoteLines], [0, 20])


if __name__ == "__main__":
    unittest.main()
