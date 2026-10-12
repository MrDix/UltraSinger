"""Tests for video detection in modules.ffmpeg_helper.

Embedded cover art in audio files shows up in ffprobe as a video stream with
the ``attached_pic`` disposition and must not turn an audio file into a
"video" input.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from modules.ffmpeg_helper import (
    _has_real_video_stream,
    convert_to_ultrastar_format,
    extract_audio,
    get_audio_codec_and_extension,
    is_video_file,
)


class TestHasRealVideoStream:
    def test_cover_art_only(self):
        assert _has_real_video_stream("video,1\n") is False

    def test_real_video(self):
        assert _has_real_video_stream("video,0\n") is True

    def test_cover_art_plus_real_video(self):
        assert _has_real_video_stream("video,1\nvideo,0\n") is True

    def test_no_streams(self):
        assert _has_real_video_stream("") is False

    def test_missing_disposition_counts_as_video(self):
        assert _has_real_video_stream("video\n") is True

    def test_windows_line_endings(self):
        assert _has_real_video_stream("video,1\r\n") is False


needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not installed"
)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


@needs_ffmpeg
class TestIsVideoFile:
    def test_plain_audio(self, tmp_path):
        out = tmp_path / "plain.mp3"
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(out))
        assert is_video_file(str(out)) is False

    def test_audio_with_cover_art(self, tmp_path):
        cover = tmp_path / "cover.png"
        _ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1", str(cover))
        out = tmp_path / "with_cover.mp3"
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-i", str(cover),
                "-map", "0:a", "-map", "1:v", "-c:v", "mjpeg", "-disposition:v", "attached_pic",
                "-id3v2_version", "3", str(out))
        assert is_video_file(str(out)) is False

    def test_real_video(self, tmp_path):
        out = tmp_path / "clip.mp4"
        _ffmpeg("-f", "lavfi", "-i", "color=c=blue:s=64x64:d=1", "-f", "lavfi",
                "-i", "sine=frequency=440:duration=1", "-shortest", "-pix_fmt", "yuv420p", str(out))
        assert is_video_file(str(out)) is True

    def test_missing_file(self, tmp_path):
        assert is_video_file(str(tmp_path / "missing.mp4")) is False


@needs_ffmpeg
class TestSameBytesEveryTime:
    """The song audio is written with the same bytes in every conversion: the cache keys
    of the steps after it (see conversion_cache.content_key) follow its bytes."""

    def test_conversion_to_ogg(self, tmp_path):
        source = tmp_path / "source.m4a"
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(source))
        outputs = []
        for run in ("a", "b"):
            song = tmp_path / run
            song.mkdir()
            shutil.copy(source, song / "Song.m4a")  # the conversion removes its input
            _, ext = convert_to_ultrastar_format(str(song / "Song.m4a"), "Song", str(song), "m4a")
            assert ext == "ogg"
            outputs.append((song / "Song.ogg").read_bytes())
        assert outputs[0] == outputs[1]

    def test_extracted_audio(self, tmp_path):
        source = tmp_path / "source.ogg"
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "libvorbis", str(source))
        for run in ("a", "b"):
            extract_audio(str(source), str(tmp_path / f"{run}.ogg"))
        assert (tmp_path / "a.ogg").read_bytes() == (tmp_path / "b.ogg").read_bytes()


@needs_ffmpeg
class TestSeveralAudioStreams:
    """A video with audio in several languages: the first audio stream is extracted, whose
    codec names the file extension (ffmpeg alone takes the default one, else the one with
    the most channels)."""

    def test_first_audio_stream_is_extracted(self, tmp_path):
        video = tmp_path / "video.mkv"
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
                "-map", "0:a", "-map", "1:a",
                "-c:a:0", "libvorbis", "-ac:a:0", "2", "-c:a:1", "ac3", "-ac:a:1", "6",
                "-disposition:a:0", "0", "-disposition:a:1", "default", str(video))
        ext = get_audio_codec_and_extension(str(video))
        assert ext == "ogg"
        audio = tmp_path / f"audio.{ext}"
        extract_audio(str(video), str(audio))
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,channels",
                                "-of", "csv=p=0", str(audio)], check=True, capture_output=True, text=True)
        assert probe.stdout.split() == ["vorbis,2"]
