"""Tests for lyrics_sources.py: the user's own lyrics file and lyrics sources."""

import contextlib
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.modules import lyrics_sources
from src.modules.lrclib_client import LyricsInfo
from src.modules.lyrics_sources import (
    OWN_LYRICS_SOURCE,
    find_song_lyrics,
    is_ultrastar_chart,
    load_lyrics_sources,
    lyrics_from_text,
    lyrics_of_source,
    lyrics_of_sources,
    read_lyrics_file,
    sidecar_lyrics_file,
)

PLAIN = "first line of the song\nsecond line of it\n\nthird line after a gap"
SYNCED = "[00:10.00] first line of the song\n[00:14.50] second line of it\n[00:20.00] third line after a gap"
CHART = "#TITLE:Song\n#ARTIST:Band\n#BPM:300\n: 0 4 60 first \n: 6 4 62 line\nE\n"


def _quiet(function, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return function(*args, **kwargs), out.getvalue()


def _source_file(folder, name, body):
    path = Path(folder) / name
    path.write_text(body, encoding="utf-8")
    return path


class TestLyricsFromText(unittest.TestCase):
    def test_plain_lines(self):
        lyrics = lyrics_from_text(PLAIN, "Song.txt")
        self.assertEqual((lyrics.plain_lyrics, lyrics.synced_lyrics), (PLAIN, None))
        self.assertEqual((lyrics.source, lyrics.trusted, lyrics.track_name), (OWN_LYRICS_SOURCE, True, "Song.txt"))

    def test_lines_with_times_are_synced_lyrics(self):
        lyrics = lyrics_from_text(SYNCED, "Song.lrc")
        self.assertEqual(lyrics.synced_lyrics, SYNCED)
        self.assertEqual(lyrics.plain_lyrics.splitlines(),
                         ["first line of the song", "second line of it", "third line after a gap"])

    def test_most_lines_decide(self):
        """Tags of an LRC file without a time ("[ar:Band]") are no reason to read it as plain text;
        a time at one line of a plain text is none to read it as synced."""
        self.assertIsNotNone(lyrics_from_text("[ar:Band]\n" + SYNCED, "a").synced_lyrics)
        self.assertIsNone(lyrics_from_text("[00:01.00] intro\n" + PLAIN, "a").synced_lyrics)

    def test_the_id_tags_of_an_lrc_file_are_no_lyrics(self):
        tags = "[ar:The Band]\n[ti:A Song]\n[al:An Album]\n[by:a player]"
        # only tags (a file a player wrote for a song it has no lyrics of): no lyrics
        with self.assertRaises(ValueError):
            lyrics_from_text(tags, "Song.lrc")
        # more tags than lines with times: still synced lyrics
        lyrics = lyrics_from_text(tags + "\n[00:10.00] first line\n[00:14.00] second line", "Song.lrc")
        self.assertIsNotNone(lyrics.synced_lyrics)
        self.assertEqual(lyrics.plain_lyrics.splitlines(), ["first line", "second line"])
        # tags above plain lines are left out of them
        lyrics = lyrics_from_text(tags + "\n" + PLAIN, "Song.txt")
        self.assertEqual((lyrics.plain_lyrics, lyrics.synced_lyrics), (PLAIN, None))
        # a section label is no ID tag
        self.assertEqual(lyrics_from_text("[Chorus]\n" + PLAIN, "a").plain_lyrics, "[Chorus]\n" + PLAIN)

    def test_times_with_a_colon_before_the_hundredths(self):
        colon = "[00:10:00] first line of the song\n[00:14:50] second line of it\n[00:20:00] third line after a gap"
        lyrics = lyrics_from_text(colon, "Song.lrc")
        self.assertEqual(lyrics.synced_lyrics, SYNCED)
        self.assertEqual(lyrics.plain_lyrics.splitlines(),
                         ["first line of the song", "second line of it", "third line after a gap"])
        # also among lines written with a dot
        mixed = lyrics_from_text("[00:10.00] one\n[00:14:50] two\n[00:20.00] three", "a")
        self.assertEqual(mixed.plain_lyrics.splitlines(), ["one", "two", "three"])

    def test_line_ends_of_any_kind(self):
        lyrics = lyrics_from_text(PLAIN.replace("\n", "\r\n") + "\r\n", "a")
        self.assertEqual(lyrics.plain_lyrics, PLAIN)

    def test_look_alike_letters_are_folded(self):
        # Cyrillic "о" and "е" in words with Latin letters
        lyrics = lyrics_from_text("hellо wоrld\nsomе mоre", "a")
        self.assertEqual(lyrics.plain_lyrics, "hello world\nsome more")

    def test_no_text_is_no_lyrics(self):
        for text in ("", "  \n\n", "[00:01.00]\n[00:02.00]  "):
            with self.subTest(text=text), self.assertRaises(ValueError):
                lyrics_from_text(text, "a")

    def test_an_ultrastar_chart_is_no_lyrics_text(self):
        self.assertTrue(is_ultrastar_chart(CHART))
        self.assertTrue(is_ultrastar_chart(": 0 4 60 one \n: 6 4 62 two \n* 12 4 64 three\n"))
        self.assertFalse(is_ultrastar_chart(PLAIN))
        self.assertFalse(is_ultrastar_chart("#1 hit\n: this line starts with a colon"))
        with self.assertRaises(ValueError):
            lyrics_from_text(CHART, "Song.txt")


class TestReadLyricsFile(unittest.TestCase):
    def test_encodings(self):
        text = "schöne Grüße\nzweite Zeile"
        with tempfile.TemporaryDirectory() as folder:
            for encoding in ("utf-8", "utf-8-sig", "utf-16", "cp1252"):
                with self.subTest(encoding=encoding):
                    path = Path(folder) / f"{encoding}.txt"
                    path.write_bytes(text.encode(encoding))
                    self.assertEqual(read_lyrics_file(path).plain_lyrics, text)

    def test_a_large_file_is_no_lyrics_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "big.txt"
            path.write_bytes(b"la " * 400_000)
            with self.assertRaises(ValueError):
                read_lyrics_file(path)

    def test_a_missing_file(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(OSError):
            read_lyrics_file(Path(folder) / "none.txt")


class TestSidecarLyricsFile(unittest.TestCase):
    def test_the_file_under_the_name_of_the_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "Band - Song.mp3"
            audio.write_bytes(b"x")
            self.assertIsNone(sidecar_lyrics_file(audio))
            (Path(folder) / "Band - Song.txt").write_text(PLAIN + chr(10) + "a fourth line", encoding="utf-8")
            self.assertEqual(sidecar_lyrics_file(audio), str(Path(folder) / "Band - Song.txt"))
            # synced lyrics come first
            (Path(folder) / "Band - Song.lrc").write_text(SYNCED, encoding="utf-8")
            self.assertEqual(sidecar_lyrics_file(audio), str(Path(folder) / "Band - Song.lrc"))

    def test_the_file_comes_with_its_lyrics(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "Song.mp3"
            audio.write_bytes(b"x")
            self.assertIsNone(lyrics_sources.sidecar_lyrics(audio))
            (Path(folder) / "Song.lrc").write_text(SYNCED, encoding="utf-8")
            path, lyrics = lyrics_sources.sidecar_lyrics(audio)
        self.assertEqual((path, lyrics.synced_lyrics, lyrics.trusted), (str(Path(folder) / "Song.lrc"), SYNCED, True))

    def test_a_chart_or_an_empty_file_is_none(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "Song.mp3"
            audio.write_bytes(b"x")
            (Path(folder) / "Song.txt").write_text(CHART, encoding="utf-8")
            (Path(folder) / "Song.lrc").write_text("", encoding="utf-8")
            self.assertIsNone(sidecar_lyrics_file(audio))

    def test_a_text_file_has_to_look_like_lyrics(self):
        """Notes on a song under its name are not its lyrics; an LRC file is taken as it is."""
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "Song.mp3"
            audio.write_bytes(b"x")
            text = Path(folder) / "Song.txt"
            line = "line" + chr(10)
            for content, taken in ((line * 3, False), (line * 4, True),
                                   (line * 5 + "x" * 301, False), (line * 5 + "x" * 300, True)):
                with self.subTest(content=content[:20]):
                    text.write_text(content, encoding="utf-8")
                    self.assertEqual(sidecar_lyrics_file(audio) is not None, taken)
            text.unlink()
            (Path(folder) / "Song.lrc").write_text("[00:01.00] one line", encoding="utf-8")
            self.assertIsNotNone(sidecar_lyrics_file(audio))

    def test_the_input_itself_is_none(self):
        with tempfile.TemporaryDirectory() as folder:
            lyrics = Path(folder) / "Song.txt"
            lyrics.write_text(PLAIN, encoding="utf-8")
            self.assertIsNone(sidecar_lyrics_file(lyrics))


class TestLoadLyricsSources(unittest.TestCase):
    def test_the_python_files_of_the_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            _source_file(folder, "b_plain.py", "def find_lyrics(artist, title, duration):\n    return None\n")
            _source_file(folder, "a_named.py", "NAME = 'My Collection'\nBEFORE_DATABASES = True\n"
                                               "def find_lyrics(artist, title, duration):\n    return None\n")
            _source_file(folder, "_helper.py", "raise RuntimeError('not loaded')\n")
            _source_file(folder, "notes.txt", "no source")
            sources, out = _quiet(load_lyrics_sources, folder)
        self.assertEqual([(source.name, source.before_databases) for source in sources],
                         [("My Collection", True), ("b_plain", False)])
        self.assertEqual(out, "")

    def test_a_file_that_cannot_be_loaded_is_reported_and_left_out(self):
        with tempfile.TemporaryDirectory() as folder:
            _source_file(folder, "broken.py", "raise RuntimeError('boom')\n")
            _source_file(folder, "empty.py", "X = 1\n")
            _source_file(folder, "good.py", "def find_lyrics(artist, title, duration):\n    return None\n")
            sources, out = _quiet(load_lyrics_sources, folder)
        self.assertEqual([source.name for source in sources], ["good"])
        self.assertIn("broken.py", out)
        self.assertIn("boom", out)
        self.assertIn("has no find_lyrics function", out)

    def test_a_source_is_a_module_python_knows(self):
        """Data classes and type hints look up the module of their class."""
        body = ("from __future__ import annotations\n"
                "from dataclasses import dataclass\n\n\n"
                "@dataclass\nclass Hit:\n    text: str\n    length: float | None = None\n\n\n"
                "def find_lyrics(artist, title, duration):\n    return Hit('la la la').text\n")
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as other:
            _source_file(folder, "typed.py", body)
            _source_file(other, "typed.py", body.replace("la la la", "na na na"))
            (first,), out = _quiet(load_lyrics_sources, folder)
            (second,), _ = _quiet(load_lyrics_sources, other)
        self.assertEqual(out, "")
        # the same file name in two folders: two sources
        self.assertEqual((first.find("a", "t", None), second.find("a", "t", None)), ("la la la", "na na na"))

    def test_a_source_that_ends_the_program_is_a_source_that_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            _source_file(folder, "exits.py", "import sys\nsys.exit(0)\n")
            sources, out = _quiet(load_lyrics_sources, folder)
        self.assertEqual(sources, [])
        self.assertIn("could not be loaded", out)
        self.assertIn("SystemExit", out)
        self.assertEqual([name for name in sys.modules if name.startswith("ultrasinger_lyrics_source_exits")], [])
        lyrics, out = _quiet(lyrics_of_source, _source(SystemExit(3)), "The Band", "A Song", None)
        self.assertIsNone(lyrics)
        self.assertIn("failed", out)

    def test_a_source_that_takes_too_long_to_load_is_left_out(self):
        """A source may ask a server while it is loaded: the conversion does not wait for it."""
        with tempfile.TemporaryDirectory() as folder:
            _source_file(folder, "a_slow.py", "import time\ntime.sleep(1.0)\n"
                                              "def find_lyrics(artist, title, duration):\n    return None\n")
            _source_file(folder, "b_good.py", "def find_lyrics(artist, title, duration):\n    return None\n")
            with patch.object(lyrics_sources, "SOURCE_TIMEOUT_S", 0.05):
                started = time.monotonic()
                sources, out = _quiet(load_lyrics_sources, folder)
        self.assertEqual([source.name for source in sources], ["b_good"])
        self.assertLess(time.monotonic() - started, 0.8)
        self.assertIn("a_slow.py", out)
        self.assertIn("did not load", out)

    def test_no_folder_no_sources(self):
        self.assertEqual(load_lyrics_sources(None), [])
        self.assertEqual(load_lyrics_sources(""), [])
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(load_lyrics_sources(Path(folder) / "none"), [])

    def test_the_folder_is_loaded_once(self):
        with tempfile.TemporaryDirectory() as folder:
            _source_file(folder, "one.py", "def find_lyrics(artist, title, duration):\n    return None\n")
            first = lyrics_sources.lyrics_sources(folder)
            _source_file(folder, "two.py", "def find_lyrics(artist, title, duration):\n    return None\n")
            self.assertIs(lyrics_sources.lyrics_sources(folder), first)


def _source(answer, name="My Source", before=False):
    def find(artist, title, duration):
        if isinstance(answer, BaseException):
            raise answer
        return answer(artist, title, duration) if callable(answer) else answer
    return lyrics_sources.LyricsSource(name=name, find=find, before_databases=before, path="source.py")


class TestLyricsOfSource(unittest.TestCase):
    def test_text_or_a_dict(self):
        for answer in (SYNCED, {"lyrics": SYNCED, "duration": 201.5, "artist": "The Band", "title": "A Song"}):
            with self.subTest(kind=type(answer).__name__):
                lyrics, _ = _quiet(lyrics_of_source, _source(answer), "The Band", "A Song", 200.0)
                self.assertEqual((lyrics.synced_lyrics, lyrics.source, lyrics.trusted), (SYNCED, "My Source", False))
                self.assertEqual((lyrics.artist_name, lyrics.track_name), ("The Band", "A Song"))
        self.assertEqual(lyrics.duration, 201.5)

    def test_what_the_source_is_asked(self):
        asked = []
        _quiet(lyrics_of_source, _source(lambda *args: asked.append(args)), "The Band", "A Song", 200.0)
        self.assertEqual(asked, [("The Band", "A Song", 200.0)])

    def test_lyrics_of_a_song_with_another_name_are_not_taken(self):
        for names in ({"title": "Another Song"}, {"artist": "Somebody Else"},
                      {"artist": "The Band", "title": "Another Song"}):
            with self.subTest(names=names):
                lyrics, out = _quiet(lyrics_of_source, _source({"lyrics": PLAIN, **names}), "The Band", "A Song", None)
                self.assertIsNone(lyrics)
                self.assertIn("are of another song", out)
        lyrics, _ = _quiet(lyrics_of_source, _source({"lyrics": PLAIN, "title": "A Song (Remastered)"}),
                           "The Band", "A Song", None)
        self.assertIsNotNone(lyrics)

    def test_no_lyrics(self):
        for answer in (None, "", "   ", {"lyrics": ""}, {"text": PLAIN}, 42, CHART):
            with self.subTest(answer=answer):
                lyrics, _ = _quiet(lyrics_of_source, _source(answer), "The Band", "A Song", None)
                self.assertIsNone(lyrics)

    def test_a_source_that_fails_is_reported(self):
        lyrics, out = _quiet(lyrics_of_source, _source(RuntimeError("no network")), "The Band", "A Song", None)
        self.assertIsNone(lyrics)
        self.assertIn("failed", out)
        self.assertIn("no network", out)

    def test_a_source_that_does_not_answer_is_left_behind(self):
        with patch.object(lyrics_sources, "SOURCE_TIMEOUT_S", 0.05):
            started = time.monotonic()
            lyrics, out = _quiet(lyrics_of_source, _source(lambda *args: time.sleep(1.0) or PLAIN),
                                 "The Band", "A Song", None)
        self.assertIsNone(lyrics)
        self.assertLess(time.monotonic() - started, 0.8)
        self.assertIn("did not answer", out)

    def test_a_length_that_is_none(self):
        for length in (0, -3, "200", True, None):
            with self.subTest(length=length):
                lyrics, _ = _quiet(lyrics_of_source, _source({"lyrics": PLAIN, "duration": length}), "B", "S", None)
                self.assertIsNone(lyrics.duration)


class TestLyricsOfSources(unittest.TestCase):
    def test_the_first_source_with_lyrics(self):
        sources = [_source(None, "One"), _source(PLAIN, "Two"), _source(SYNCED, "Three")]
        lyrics, out = _quiet(lyrics_of_sources, sources, "B", "S", None)
        self.assertEqual(lyrics.source, "Two")
        self.assertNotIn("Three", out)
        self.assertEqual(_quiet(lyrics_of_sources, [], "B", "S", None)[0], None)

    def test_a_source_that_failed_is_not_asked_again(self):
        """A conversion looks lyrics up twice: it does not wait a second time for a source that
        failed or did not answer."""
        asked = []

        def slow(artist, title, duration):
            asked.append("slow")
            time.sleep(1.0)

        def broken(artist, title, duration):
            asked.append("broken")
            raise RuntimeError("no connection")

        def silent(artist, title, duration):
            asked.append("silent")

        sources = [_source(slow, "Slow"), _source(broken, "Broken"), _source(silent, "Silent")]
        with patch.object(lyrics_sources, "SOURCE_TIMEOUT_S", 0.05):
            _, first = _quiet(lyrics_of_sources, sources, "B", "S", None)
            _, second = _quiet(lyrics_of_sources, sources, "B", "S", None)
        self.assertEqual(asked, ["slow", "broken", "silent", "silent"])
        self.assertEqual([source.failed for source in sources], [True, True, False])
        self.assertIn("did not answer", first)
        self.assertIn("failed", first)
        # (a source without lyrics of the song has not failed, and the others are not named again)
        self.assertIn("Silent", second)
        self.assertNotIn("Slow", second)
        self.assertNotIn("Broken", second)

    def test_a_failed_source_is_one_of_this_run(self):
        """The sources of a folder are loaded once and kept: so is what failed."""
        with tempfile.TemporaryDirectory() as folder:
            _source_file(folder, "broken.py", "CALLS = []\n"
                                              "def find_lyrics(artist, title, duration):\n"
                                              "    CALLS.append(title)\n"
                                              "    raise RuntimeError('no connection')\n")
            with patch.object(lyrics_sources.lrclib_client, "search_lyrics", return_value=None) as search:
                for _ in range(2):
                    self.assertIsNone(_quiet(find_song_lyrics, "The Band", "A Song", 200.0, sources_dir=folder)[0])
            source, = lyrics_sources.lyrics_sources(folder)
            self.assertTrue(source.failed)
            self.assertEqual(sys.modules[source.find.__module__].CALLS, ["A Song"])
            self.assertEqual(search.call_count, 2)

    def test_instead_of_plain_lyrics_only_synced_ones_of_the_same_text(self):
        plain = LyricsInfo(track_name="S", artist_name="B", plain_lyrics=PLAIN, duration=200.0)
        other_text = "[00:10.00] words of another song\n[00:14.00] that has other lines\n[00:20.00] all of them"
        for name, answer, taken in (
                ("the same text with times", SYNCED, True),
                ("plain lyrics", PLAIN, False),
                ("another text", other_text, False),
                ("another recording", {"lyrics": SYNCED, "duration": 300.0}, False),
                ("the same recording", {"lyrics": SYNCED, "duration": 203.0}, True)):
            with self.subTest(name):
                lyrics, _ = _quiet(lyrics_of_sources, [_source(answer)], "B", "S", 200.0, instead_of=plain)
                self.assertEqual(lyrics is not None, taken)

    def test_without_the_length_of_the_audio_the_version_of_the_plain_lyrics_counts(self):
        plain = LyricsInfo(track_name="S", artist_name="B", plain_lyrics=PLAIN, duration=200.0)
        for length, taken in ((300.0, False), (205.0, True), (None, True)):
            with self.subTest(length=length):
                lyrics, _ = _quiet(lyrics_of_sources, [_source({"lyrics": SYNCED, "duration": length})], "B", "S",
                                   None, instead_of=plain)
                self.assertEqual(lyrics is not None, taken)
        # plain lyrics without a length either: nothing to compare with
        unknown = LyricsInfo(track_name="S", artist_name="B", plain_lyrics=PLAIN, duration="200")
        lyrics, _ = _quiet(lyrics_of_sources, [_source({"lyrics": SYNCED, "duration": 300.0})], "B", "S", None,
                           instead_of=unknown)
        self.assertIsNotNone(lyrics)


class TestFindSongLyrics(unittest.TestCase):
    def _find(self, database=None, sources=(), **kwargs):
        """find_song_lyrics with the databases answering ``database``; returns the lyrics, the
        searches of the databases and what was printed."""
        with patch.object(lyrics_sources.lrclib_client, "search_lyrics", return_value=database) as search, \
                patch.object(lyrics_sources.lrclib_client, "report_found_lyrics"), \
                patch.object(lyrics_sources, "lyrics_sources", return_value=list(sources)):
            lyrics, out = _quiet(find_song_lyrics, "The Band", "A Song", 200.0, **kwargs)
        return lyrics, search, out

    def test_your_own_lyrics_and_nothing_else(self):
        own = lyrics_from_text(SYNCED, "mine.lrc")
        lyrics, search, out = self._find(database=LyricsInfo("S", "B", plain_lyrics="other"),
                                         sources=[_source(PLAIN, before=True)], own_lyrics=own,
                                         own_lyrics_file="D:/lyrics/mine.lrc")
        self.assertIs(lyrics, own)
        search.assert_not_called()
        self.assertIn("Found lyrics in your file", out)
        self.assertIn("mine.lrc", out)
        self.assertNotIn("D:/lyrics", out)
        self.assertIn("[plain (14 words), synced]", out)

    def test_your_own_lyrics_also_without_a_lookup_or_names(self):
        own = lyrics_from_text(PLAIN, "mine.txt")
        lyrics, search, _ = self._find(own_lyrics=own, lookup=False)
        self.assertIs(lyrics, own)
        with patch.object(lyrics_sources.lrclib_client, "search_lyrics") as search:
            lyrics, out = _quiet(find_song_lyrics, "", "", None, own_lyrics=own)
        self.assertIs(lyrics, own)
        search.assert_not_called()
        self.assertIn("mine.txt", out)  # (the name the lyrics came with, when no file is told)

    def test_no_lookup_no_lyrics(self):
        lyrics, search, _ = self._find(database=LyricsInfo("S", "B", plain_lyrics=PLAIN),
                                       sources=[_source(PLAIN, before=True)], lookup=False)
        self.assertIsNone(lyrics)
        search.assert_not_called()

    def test_a_song_without_a_name_is_not_looked_up(self):
        with patch.object(lyrics_sources.lrclib_client, "search_lyrics") as search:
            self.assertIsNone(_quiet(find_song_lyrics, "The Band", " ", 200.0)[0])
            self.assertIsNone(_quiet(find_song_lyrics, "", "A Song", 200.0)[0])
        search.assert_not_called()

    def test_a_source_before_the_databases(self):
        lyrics, search, _ = self._find(database=LyricsInfo("S", "B", synced_lyrics="[00:01.00] other"),
                                       sources=[_source(PLAIN, "Late"), _source(SYNCED, "Early", before=True)])
        self.assertEqual(lyrics.source, "Early")
        search.assert_not_called()
        # it has none: the databases
        lyrics, search, _ = self._find(database=LyricsInfo("S", "B", synced_lyrics=SYNCED),
                                       sources=[_source(None, "Early", before=True)])
        self.assertEqual(lyrics.source, "LRCLIB")
        search.assert_called_once_with("The Band", "A Song", duration=200.0)

    def test_the_other_sources_only_for_what_the_databases_do_not_have(self):
        asked = []
        late = _source(lambda *args: asked.append(args) or SYNCED, "Late")
        # synced lyrics found: not asked
        found = LyricsInfo("S", "B", plain_lyrics=PLAIN, synced_lyrics=SYNCED)
        self.assertIs(self._find(database=found, sources=[late])[0], found)
        # known as instrumental: not asked
        found = LyricsInfo("S", "B", instrumental=True)
        self.assertIs(self._find(database=found, sources=[late])[0], found)
        self.assertEqual(asked, [])
        # nothing found: its lyrics
        self.assertEqual(self._find(database=None, sources=[late])[0].source, "Late")
        # plain lyrics found: its synced ones of the same text
        found = LyricsInfo("S", "B", plain_lyrics=PLAIN, duration=200.0)
        self.assertEqual(self._find(database=found, sources=[late])[0].source, "Late")
        # ... but no plain ones, and none of another text: the plain ones of the database stay
        self.assertIs(self._find(database=found, sources=[_source(PLAIN, "Late")])[0], found)
        self.assertIs(self._find(database=found, sources=[_source("[00:01.00] other words here", "Late")])[0], found)

    def test_the_sources_of_the_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            _source_file(folder, "mine.py", "NAME = 'Mine'\ndef find_lyrics(artist, title, duration):\n"
                                            "    return {'lyrics': '[00:01.00] ' + artist + ' sings ' + title}\n")
            with patch.object(lyrics_sources.lrclib_client, "search_lyrics", return_value=None), \
                    patch.object(lyrics_sources.lrclib_client, "report_found_lyrics") as report:
                lyrics, _ = _quiet(find_song_lyrics, "The Band", "A Song", 200.0, sources_dir=folder)
        self.assertEqual((lyrics.source, lyrics.plain_lyrics), ("Mine", "The Band sings A Song"))
        report.assert_called_once()

    def test_the_default_folder_is_in_the_settings_folder(self):
        self.assertEqual(lyrics_sources.default_sources_dir().name, "lyrics_sources")
        self.assertEqual(lyrics_sources.default_sources_dir().parent, lyrics_sources._SETTINGS_DIR)
        # (the tests have a settings folder of their own: see conftest.py)
        self.assertNotEqual(lyrics_sources._SETTINGS_DIR, Path.home() / ".ultrasinger")


if __name__ == "__main__":
    unittest.main()
