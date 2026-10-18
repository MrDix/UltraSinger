"""Tests for video detection in modules.ffmpeg_helper.

Embedded cover art in audio files shows up in ffprobe as a video stream with
the ``attached_pic`` disposition and must not turn an audio file into a
"video" input.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from unittest.mock import Mock, patch

import pytest

from modules import ffmpeg_helper
from modules.ffmpeg_helper import (
    _has_real_video_stream,
    convert_to_ultrastar_format,
    extract_audio,
    get_audio_codec_and_extension,
    get_duration,
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


class TestDuration:
    """The length of an input file, of a video too, for the song lookup."""

    @needs_ffmpeg
    def test_audio_and_video(self, tmp_path):
        audio = tmp_path / "song.mp3"
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(audio))
        video = tmp_path / "clip.mp4"
        _ffmpeg("-f", "lavfi", "-i", "color=c=blue:s=64x64:d=2", "-f", "lavfi",
                "-i", "sine=frequency=440:duration=2", "-shortest", "-pix_fmt", "yuv420p", str(video))
        for path in (audio, video):
            assert get_duration(str(path)) == pytest.approx(2.0, abs=0.1)

    @needs_ffmpeg
    def test_a_file_without_a_length(self, tmp_path):
        no_audio = tmp_path / "song.mp3"
        no_audio.write_bytes(b"audio")
        assert get_duration(str(no_audio)) is None
        assert get_duration(str(tmp_path / "missing.mp3")) is None

    def test_without_ffprobe(self):
        with patch.object(ffmpeg_helper, "get_ffmpeg_and_ffprobe_paths",
                          side_effect=FileNotFoundError("FFprobe executable not found in the system path.")):
            assert get_duration("song.mp3") is None


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
        # Encoders that every ffmpeg build has
        video = tmp_path / "video.mkv"
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
                "-map", "0:a", "-map", "1:a",
                "-c:a:0", "flac", "-ac:a:0", "2", "-c:a:1", "ac3", "-ac:a:1", "6",
                "-disposition:a:0", "0", "-disposition:a:1", "default", str(video))
        ext = get_audio_codec_and_extension(str(video))
        assert ext == "flac"
        audio = tmp_path / f"audio.{ext}"
        extract_audio(str(video), str(audio))
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,channels",
                                "-of", "csv=p=0", str(audio)], check=True, capture_output=True, text=True)
        assert probe.stdout.split() == ["flac,2"]


class TestKeepAudioInVideo:
    """With --keep_audio_in_video the downloaded video only gets its final name."""

    def _separate(self, video_path, folder, basename):
        with (
            patch.object(ffmpeg_helper, "get_audio_codec_and_extension", return_value="opus"),
            patch.object(ffmpeg_helper, "extract_audio"),
            patch.object(ffmpeg_helper, "convert_to_ultrastar_format",
                         side_effect=lambda path, base, out, ext: (path, ext)),
            patch.object(ffmpeg_helper.os, "rename") as rename,
        ):
            result = ffmpeg_helper.separate_audio_video(video_path, basename, folder,
                                                        keep_audio_in_video=True)
        return rename, result

    def test_same_file_written_differently_is_not_renamed(self, tmp_path):
        # the download reports "D:\folder\Song.mp4", the output folder is "D:/folder"
        video = tmp_path / "Artist - Title.mp4"
        video.write_bytes(b"video")
        folder = str(tmp_path).replace(os.sep, "/")
        rename, (_, final_video, _, _) = self._separate(str(video), folder, "Artist - Title")
        rename.assert_not_called()
        assert os.path.samefile(final_video, video)

    def test_letter_case_of_the_path_on_windows(self, tmp_path):
        video = tmp_path / "Artist - Title.mp4"
        video.write_bytes(b"video")
        folder = str(tmp_path).upper() if os.name == "nt" else str(tmp_path)
        rename, _ = self._separate(str(video), folder, "Artist - Title")
        rename.assert_not_called()

    def test_another_name_is_renamed(self, tmp_path):
        video = tmp_path / "download.mp4"
        video.write_bytes(b"video")
        rename, (_, final_video, _, _) = self._separate(str(video), str(tmp_path), "Artist - Title")
        rename.assert_called_once_with(str(video), final_video)
        assert final_video == os.path.join(str(tmp_path), "Artist - Title.mp4")


class TestFileInUse:
    """A sync client or virus scanner may hold a file that was just written."""

    def test_waits_until_the_file_is_free(self):
        action = Mock(side_effect=[PermissionError("in use"), PermissionError("in use"), "done"])
        with patch.object(ffmpeg_helper.time, "sleep") as sleep:
            assert ffmpeg_helper._when_not_in_use(action, "a", "b") == "done"
        assert action.call_count == 3
        assert sleep.call_count == 2
        action.assert_called_with("a", "b")

    def test_gives_up_after_a_few_seconds(self):
        action = Mock(side_effect=PermissionError("in use"))
        with patch.object(ffmpeg_helper.time, "sleep"), pytest.raises(PermissionError):
            ffmpeg_helper._when_not_in_use(action, "a")
        assert action.call_count == ffmpeg_helper._IN_USE_RETRIES

    def test_other_errors_are_not_retried(self):
        action = Mock(side_effect=FileNotFoundError("gone"))
        with patch.object(ffmpeg_helper.time, "sleep") as sleep, pytest.raises(FileNotFoundError):
            ffmpeg_helper._when_not_in_use(action, "a")
        assert action.call_count == 1
        sleep.assert_not_called()
