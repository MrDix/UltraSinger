"""Tests for metadata reader and writer modules."""

import os
import tempfile
import unittest

from src.modules.Audio.metadata_reader import read_media_metadata, format_display_title


class TestMetadataReader(unittest.TestCase):
    def test_nonexistent_file(self):
        result = read_media_metadata("/nonexistent/path.mp3")
        self.assertIsNone(result["title"])
        self.assertIsNone(result["artist"])

    def test_empty_path(self):
        result = read_media_metadata("")
        self.assertIsNone(result["title"])

    def test_none_path(self):
        result = read_media_metadata(None)
        self.assertIsNone(result["title"])

    def test_format_display_title_both(self):
        meta = {"artist": "TestArtist", "title": "TestSong"}
        self.assertEqual(format_display_title(meta), "TestArtist - TestSong")

    def test_format_display_title_only_title(self):
        meta = {"artist": None, "title": "TestSong"}
        self.assertEqual(format_display_title(meta), "TestSong")

    def test_format_display_title_only_artist(self):
        meta = {"artist": "TestArtist", "title": None}
        self.assertEqual(format_display_title(meta), "TestArtist")

    def test_format_display_title_fallback(self):
        meta = {"artist": None, "title": None}
        self.assertEqual(format_display_title(meta, "fallback"), "fallback")


class TestMetadataWriter(unittest.TestCase):
    def test_write_unsupported_format(self):
        from src.modules.Audio.metadata_writer import write_metadata_to_audio
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"not audio")
            path = f.name
        try:
            result = write_metadata_to_audio(path, title="Test")
            self.assertFalse(result)
        finally:
            os.unlink(path)

    def test_write_nonexistent_file(self):
        from src.modules.Audio.metadata_writer import write_metadata_to_audio
        result = write_metadata_to_audio("/nonexistent/file.mp3", title="Test")
        self.assertFalse(result)

    def test_roundtrip_mp3(self):
        """Write and read back MP3 tags."""
        try:
            from pydub import AudioSegment
            from pydub.generators import Sine
        except ImportError:
            self.skipTest("pydub not available")

        from src.modules.Audio.metadata_writer import write_metadata_to_audio

        # Create a minimal MP3
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name

        try:
            tone = Sine(440).to_audio_segment(duration=500)
            tone.export(path, format="mp3")

            # Write tags
            success = write_metadata_to_audio(
                path, title="TestTitle", artist="TestArtist",
                year="2024", genre="Rock",
            )
            self.assertTrue(success)

            # Read back
            meta = read_media_metadata(path)
            self.assertEqual(meta["title"], "TestTitle")
            self.assertEqual(meta["artist"], "TestArtist")
            self.assertEqual(meta["year"], "2024")
            self.assertEqual(meta["genre"], "Rock")
        finally:
            os.unlink(path)

    def test_roundtrip_ogg(self):
        """Write and read back OGG Vorbis tags."""
        try:
            from pydub import AudioSegment
            from pydub.generators import Sine
        except ImportError:
            self.skipTest("pydub not available")

        from src.modules.Audio.metadata_writer import write_metadata_to_audio

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            path = f.name

        try:
            tone = Sine(440).to_audio_segment(duration=500)
            tone.export(path, format="ogg")

            success = write_metadata_to_audio(
                path, title="OggTitle", artist="OggArtist",
            )
            self.assertTrue(success)

            meta = read_media_metadata(path)
            self.assertEqual(meta["title"], "OggTitle")
            self.assertEqual(meta["artist"], "OggArtist")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
