"""Tests for the media interceptor URL parsing and stream management."""

import time

import pytest

from src.gui.media_interceptor import (
    CapturedAudioStream,
    MediaInterceptor,
    _extract_video_id_from_url,
    _is_audio_request,
    _strip_range_params,
)


class TestExtractVideoId:
    """Test stream ID extraction from googlevideo.com URLs."""

    def test_extracts_id_param(self):
        url = "https://rr5---sn-abc.googlevideo.com/videoplayback?expire=999&id=o-AHol2Kw8g6PjnvA&itag=251"
        assert _extract_video_id_from_url(url) == "o-AHol2Kw8g6PjnvA"

    def test_no_id_param(self):
        url = "https://rr1.googlevideo.com/videoplayback?itag=140&expire=999"
        assert _extract_video_id_from_url(url) == ""


class TestIsAudioRequest:
    """Test audio vs video stream detection."""

    def test_audio_webm_mime(self):
        url = "https://rr1.googlevideo.com/videoplayback?mime=audio%2Fwebm&itag=251"
        assert _is_audio_request(url) is True

    def test_audio_mp4_mime(self):
        url = "https://rr1.googlevideo.com/videoplayback?mime=audio%2Fmp4&itag=140"
        assert _is_audio_request(url) is True

    def test_video_mime(self):
        url = "https://rr1.googlevideo.com/videoplayback?mime=video%2Fwebm&itag=243"
        assert _is_audio_request(url) is False

    def test_audio_itag_fallback(self):
        url = "https://rr1.googlevideo.com/videoplayback?itag=251"
        assert _is_audio_request(url) is True

    def test_video_itag(self):
        url = "https://rr1.googlevideo.com/videoplayback?itag=137"
        assert _is_audio_request(url) is False

    def test_no_mime_no_itag(self):
        url = "https://rr1.googlevideo.com/videoplayback?expire=999"
        assert _is_audio_request(url) is False

    def test_itag_140_m4a(self):
        url = "https://rr1.googlevideo.com/videoplayback?itag=140"
        assert _is_audio_request(url) is True

    def test_itag_249_opus(self):
        url = "https://rr1.googlevideo.com/videoplayback?itag=249"
        assert _is_audio_request(url) is True


class TestStripRangeParams:
    """Test range parameter stripping for full-file download URLs."""

    def test_removes_range(self):
        url = "https://rr1.googlevideo.com/videoplayback?itag=251&range=0-1048576&expire=999"
        result = _strip_range_params(url)
        assert "range=" not in result
        assert "itag=251" in result
        assert "expire=999" in result

    def test_removes_rn_and_rbuf(self):
        url = "https://rr1.googlevideo.com/videoplayback?itag=251&rn=3&rbuf=12345&expire=999"
        result = _strip_range_params(url)
        assert "rn=" not in result
        assert "rbuf=" not in result
        assert "itag=251" in result

    def test_preserves_other_params(self):
        url = "https://rr1.googlevideo.com/videoplayback?itag=251&sig=abc&n=xyz&expire=999"
        result = _strip_range_params(url)
        assert "sig=abc" in result
        assert "n=xyz" in result
        assert "expire=999" in result

    def test_no_range_params(self):
        url = "https://rr1.googlevideo.com/videoplayback?itag=251&expire=999"
        result = _strip_range_params(url)
        assert "itag=251" in result
        assert "expire=999" in result


class TestCapturedAudioStream:
    """Test the CapturedAudioStream dataclass."""

    def test_not_expired_when_no_expire(self):
        stream = CapturedAudioStream(url="http://x", video_id="abc")
        assert stream.is_expired is False

    def test_expired_when_past(self):
        stream = CapturedAudioStream(
            url="http://x", video_id="abc",
            expire_time=int(time.time()) - 100,
        )
        assert stream.is_expired is True

    def test_not_expired_when_future(self):
        stream = CapturedAudioStream(
            url="http://x", video_id="abc",
            expire_time=int(time.time()) + 3600,
        )
        assert stream.is_expired is False

    def test_expired_within_safety_margin(self):
        # 30s from now — within the 60s safety margin
        stream = CapturedAudioStream(
            url="http://x", video_id="abc",
            expire_time=int(time.time()) + 30,
        )
        assert stream.is_expired is True

    def test_seconds_until_expiry(self):
        future = int(time.time()) + 3660  # 1h + 60s safety = ~1h
        stream = CapturedAudioStream(
            url="http://x", video_id="abc",
            expire_time=future,
        )
        assert 3500 < stream.seconds_until_expiry < 3700


class TestMediaInterceptorStreamManagement:
    """Test stream storage and retrieval (without QWebEngine)."""

    def test_get_stream_returns_none_for_unknown(self):
        interceptor = MediaInterceptor()
        assert interceptor.get_stream("unknown_id") is None

    def test_assign_and_get_stream(self):
        interceptor = MediaInterceptor()
        stream = CapturedAudioStream(
            url="http://x", video_id="stream-id",
            expire_time=int(time.time()) + 3600,
        )
        interceptor.assign_to_video("dQw4w9WgXcQ", stream)
        result = interceptor.get_stream("dQw4w9WgXcQ")
        assert result is not None
        assert result.url == "http://x"

    def test_get_stream_removes_expired(self):
        interceptor = MediaInterceptor()
        stream = CapturedAudioStream(
            url="http://x", video_id="stream-id",
            expire_time=int(time.time()) - 100,
        )
        interceptor._streams["dQw4w9WgXcQ"] = stream
        assert interceptor.get_stream("dQw4w9WgXcQ") is None
        assert "dQw4w9WgXcQ" not in interceptor._streams

    def test_clear(self):
        interceptor = MediaInterceptor()
        interceptor._streams["a"] = CapturedAudioStream(url="x", video_id="a")
        interceptor._streams["b"] = CapturedAudioStream(url="y", video_id="b")
        interceptor.clear()
        assert len(interceptor._streams) == 0

    def test_get_all_prunes_expired(self):
        interceptor = MediaInterceptor()
        interceptor._streams["fresh"] = CapturedAudioStream(
            url="x", video_id="fresh",
            expire_time=int(time.time()) + 3600,
        )
        interceptor._streams["stale"] = CapturedAudioStream(
            url="y", video_id="stale",
            expire_time=int(time.time()) - 100,
        )
        result = interceptor.get_all_streams()
        assert "fresh" in result
        assert "stale" not in result

    def test_assign_none_stream_is_noop(self):
        interceptor = MediaInterceptor()
        interceptor.assign_to_video("dQw4w9WgXcQ", None)
        assert interceptor.get_stream("dQw4w9WgXcQ") is None

    def test_assign_empty_video_id_is_noop(self):
        interceptor = MediaInterceptor()
        stream = CapturedAudioStream(url="http://x", video_id="s")
        interceptor.assign_to_video("", stream)
        assert len(interceptor._streams) == 0
