"""Tests for video detection in modules.ffmpeg_helper.

Embedded cover art in audio files shows up in ffprobe as a video stream with
the ``attached_pic`` disposition and must not turn an audio file into a
"video" input.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from modules.ffmpeg_helper import _has_real_video_stream, is_video_file


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
