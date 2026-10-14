"""Tests for remote (cloud) speech-to-text transcription.

All ``requests`` calls are mocked -- no real network access happens here.
"""

import io
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import requests
import soundfile as sf

from src.modules.Speech_Recognition.remote_stt import (
    RemoteTranscription,
    transcribe_remote,
    MAX_UPLOAD_BYTES,
    UPLOAD_SAMPLE_RATE,
)


def _make_audio_file(content: bytes = b"fake audio bytes") -> str:
    """Create a small temp file standing in for an audio file."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return path


class TestTranscribeRemote(unittest.TestCase):
    def setUp(self):
        self.audio_path = _make_audio_file()

    def tearDown(self):
        if os.path.exists(self.audio_path):
            os.remove(self.audio_path)

    def _kwargs(self, **overrides):
        defaults = {
            "audio_path": self.audio_path,
            "api_base_url": "https://api.groq.com/openai/v1",
            "api_key": "test-key",
            "model": "whisper-large-v3",
            "language": "en",
        }
        defaults.update(overrides)
        return defaults

    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_successful_transcription_returns_text(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "hello world"}
        mock_post.return_value = mock_response

        result = transcribe_remote(**self._kwargs())

        self.assertEqual(result.text, "hello world")
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(call_kwargs["data"]["model"], "whisper-large-v3")
        self.assertEqual(call_kwargs["data"]["language"], "en")
        self.assertIn(
            "https://api.groq.com/openai/v1/audio/transcriptions", mock_post.call_args.args
        )

    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_strips_whitespace_from_text(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "  hello world  \n"}
        mock_post.return_value = mock_response

        result = transcribe_remote(**self._kwargs())
        self.assertEqual(result.text, "hello world")

    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_no_language_hint_omits_language_field(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "hello"}
        mock_post.return_value = mock_response

        transcribe_remote(**self._kwargs(language=None))

        call_kwargs = mock_post.call_args.kwargs
        self.assertNotIn("language", call_kwargs["data"])

    def test_no_api_key_returns_none(self):
        result = transcribe_remote(**self._kwargs(api_key=""))
        self.assertIsNone(result)

    def test_none_api_key_returns_none(self):
        result = transcribe_remote(**self._kwargs(api_key=None))
        self.assertIsNone(result)

    def test_missing_audio_file_returns_none(self):
        result = transcribe_remote(**self._kwargs(audio_path="/nonexistent/path/audio.wav"))
        self.assertIsNone(result)

    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_http_error_returns_none(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        result = transcribe_remote(**self._kwargs())
        self.assertIsNone(result)

    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_timeout_returns_none(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")

        result = transcribe_remote(**self._kwargs(timeout=5))
        self.assertIsNone(result)

    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_network_error_returns_none(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("connection refused")

        result = transcribe_remote(**self._kwargs())
        self.assertIsNone(result)

    @patch("src.modules.Speech_Recognition.remote_stt.MAX_UPLOAD_BYTES", 5)
    def test_oversized_file_returns_none_without_network_call(self):
        with patch("src.modules.Speech_Recognition.remote_stt.requests.post") as mock_post:
            result = transcribe_remote(**self._kwargs())
            self.assertIsNone(result)
            mock_post.assert_not_called()

    def test_ssrf_blocked_scheme_returns_none(self):
        with patch("src.modules.Speech_Recognition.remote_stt.requests.post") as mock_post:
            result = transcribe_remote(**self._kwargs(api_base_url="ftp://evil.example.com"))
            self.assertIsNone(result)
            mock_post.assert_not_called()

    def test_file_scheme_blocked(self):
        with patch("src.modules.Speech_Recognition.remote_stt.requests.post") as mock_post:
            result = transcribe_remote(**self._kwargs(api_base_url="file:///etc/passwd"))
            self.assertIsNone(result)
            mock_post.assert_not_called()

    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_empty_response_text_returns_none(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": ""}
        mock_post.return_value = mock_response

        result = transcribe_remote(**self._kwargs())
        self.assertIsNone(result)

    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_whitespace_only_response_text_returns_none(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "   \n  "}
        mock_post.return_value = mock_response

        result = transcribe_remote(**self._kwargs())
        self.assertIsNone(result)

    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_missing_text_key_returns_none(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "shape"}
        mock_post.return_value = mock_response

        result = transcribe_remote(**self._kwargs())
        self.assertIsNone(result)

    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_unparseable_json_returns_none(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("not json")
        mock_post.return_value = mock_response

        result = transcribe_remote(**self._kwargs())
        self.assertIsNone(result)

    def test_max_upload_bytes_is_25mb(self):
        self.assertEqual(MAX_UPLOAD_BYTES, 25 * 1024 * 1024)

    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_other_http_error_is_not_retried(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        result = transcribe_remote(**self._kwargs())

        self.assertIsNone(result)
        mock_post.assert_called_once()


def _response(status_code, body=None, text="ok"):
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.headers = {}
    response.json.return_value = body if body is not None else {"text": "hello world"}
    return response


class TestWordsAndSegments(unittest.TestCase):
    """The words come back with the times the service gave them, the text in its segments."""

    def setUp(self):
        self.audio_path = _make_audio_file()
        self.addCleanup(os.remove, self.audio_path)

    def _transcribe(self, *responses):
        with patch("src.modules.Speech_Recognition.remote_stt.requests.post",
                   side_effect=list(responses)) as mock_post:
            result = transcribe_remote(audio_path=self.audio_path, api_base_url="https://api.groq.com/openai/v1",
                                       api_key="test-key", model="whisper-large-v3", language="en")
        return result, mock_post

    def test_asks_for_word_and_segment_times(self):
        _, mock_post = self._transcribe(_response(200))
        data = mock_post.call_args.kwargs["data"]
        self.assertEqual(data["response_format"], "verbose_json")
        self.assertEqual(data["timestamp_granularities[]"], ["word", "segment"])
        self.assertEqual((data["model"], data["language"], data["temperature"]), ("whisper-large-v3", "en", "0"))

    def test_words_and_segments_with_their_times(self):
        body = {
            "text": " Hello world. Again",
            "words": [{"word": "Hello", "start": 1.0, "end": 1.4}, {"word": "world.", "start": 1.5, "end": 2},
                      {"word": "Again", "start": 3.0, "end": 3.5}],
            "segments": [{"text": " Hello world.", "start": 0.9, "end": 2.1, "id": 0},
                         {"text": " Again", "start": 2.9, "end": 3.6, "id": 1}],
        }
        result, _ = self._transcribe(_response(200, body))
        self.assertEqual(result.text, "Hello world. Again")
        self.assertEqual(result.words, [("Hello", 1.0, 1.4), ("world.", 1.5, 2.0), ("Again", 3.0, 3.5)])
        self.assertEqual(result.segments, [" Hello world.", " Again"])
        self.assertEqual(result.lines(), "Hello world.\nAgain")

    def test_words_without_usable_times_are_left_out(self):
        body = {
            "text": "a b c d e f",
            "words": [{"word": "a", "start": 1.0, "end": 1.2}, {"word": "b", "start": None, "end": 1.4},
                      {"word": "c", "start": True, "end": 1.6}, {"word": "d", "start": float("nan"), "end": 1.8},
                      {"word": 5, "start": 2.0, "end": 2.2}, "e", {"word": "f", "start": "3", "end": 3.2},
                      {"word": "g", "start": -1.0, "end": 0.5}, {"word": "h", "start": 4.0, "end": 3.9}],
            "segments": {"not": "a list"},
        }
        result, _ = self._transcribe(_response(200, body))
        self.assertEqual(result.words, [("a", 1.0, 1.2)])
        self.assertEqual(result.segments, [])
        self.assertEqual(result.lines(), "a b c d e f")

    def test_a_service_without_times_gives_the_text(self):
        # it turns the request for times down (HTTP 400, or 422 as many self-hosted servers do),
        # the text alone it gives
        for status in (400, 422):
            with self.subTest(status=status):
                result, mock_post = self._transcribe(_response(status, text="response_format not supported"),
                                                     _response(200, {"text": "hello world"}))
                self.assertEqual(mock_post.call_count, 2)
                data = mock_post.call_args.kwargs["data"]
                self.assertEqual(data["response_format"], "json")
                self.assertNotIn("timestamp_granularities[]", data)
                self.assertEqual((data["model"], data["language"]), ("whisper-large-v3", "en"))
                self.assertEqual((result.text, result.words, result.segments), ("hello world", [], []))

    @patch("src.modules.Speech_Recognition.remote_stt.time.sleep")
    def test_the_request_for_the_text_is_retried_when_rate_limited(self, mock_sleep):
        limited = _response(429)
        limited.headers = {"Retry-After": "2"}
        result, mock_post = self._transcribe(_response(400, text="no times"), limited,
                                             _response(200, {"text": "hello world"}))
        self.assertEqual(result.text, "hello world")
        self.assertEqual(mock_post.call_count, 3)
        mock_sleep.assert_called_once_with(2.0)

    def test_segments_without_a_text_give_no_lines(self):
        # the lines would leave the words of that segment out: the text as it came
        body = {"text": "a b c", "segments": [{"text": " a"}, {"start": 1.0, "end": 2.0}, {"text": " c"}]}
        result, _ = self._transcribe(_response(200, body))
        self.assertEqual((result.segments, result.lines()), ([], "a b c"))

    def test_a_service_that_ignores_the_times_gives_the_text(self):
        result, _ = self._transcribe(_response(200, {"text": "hello world"}))
        self.assertEqual((result.text, result.words, result.segments, result.lines()),
                         ("hello world", [], [], "hello world"))

    def test_bad_request_also_for_the_text_returns_none(self):
        result, mock_post = self._transcribe(_response(400, text="bad"), _response(400, text="bad"))
        self.assertIsNone(result)
        self.assertEqual(mock_post.call_count, 2)


class TestUploadedAudio(unittest.TestCase):
    """The audio goes out as a small mono FLAC copy at the Whisper models' sample rate."""

    def setUp(self):
        handle, self.audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        self.addCleanup(os.remove, self.audio_path)
        t = np.arange(2 * 44100) / 44100
        sf.write(self.audio_path, (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 44100)

    def _transcribe(self):
        uploaded = {}

        def post(url, headers, data, files, timeout):
            name, audio_file = files["file"]
            uploaded["name"], uploaded["path"] = name, audio_file.name
            uploaded["audio"], uploaded["rate"] = sf.read(io.BytesIO(audio_file.read()))
            return _response(200)

        with patch("src.modules.Speech_Recognition.remote_stt.requests.post", side_effect=post) as mock_post:
            result = transcribe_remote(audio_path=self.audio_path, api_base_url="https://api.groq.com/openai/v1",
                                       api_key="test-key", model="whisper-large-v3")
        return result, uploaded, mock_post

    def test_mono_flac_at_16_khz(self):
        result, uploaded, _ = self._transcribe()
        self.assertEqual(result.text, "hello world")
        self.assertTrue(uploaded["name"].endswith(".flac"))
        self.assertEqual(uploaded["rate"], UPLOAD_SAMPLE_RATE)
        self.assertEqual(uploaded["audio"].ndim, 1)
        self.assertAlmostEqual(len(uploaded["audio"]) / UPLOAD_SAMPLE_RATE, 2.0, places=2)
        self.assertGreater(np.abs(uploaded["audio"]).max(), 0.4)
        # the copy is gone afterwards, the audio itself is kept
        self.assertFalse(os.path.exists(uploaded["path"]))
        self.assertTrue(os.path.exists(self.audio_path))

    def test_the_limit_applies_to_the_copy(self):
        # the 44.1 kHz WAV is over the limit, its copy is not
        with patch("src.modules.Speech_Recognition.remote_stt.MAX_UPLOAD_BYTES",
                   os.path.getsize(self.audio_path) - 1):
            result, uploaded, _ = self._transcribe()
        self.assertEqual(result.text, "hello world")
        self.assertTrue(uploaded["name"].endswith(".flac"))

    def test_the_copy_is_removed_when_the_request_fails(self):
        made = []
        real_mkstemp = tempfile.mkstemp

        def mkstemp(*args, **kwargs):
            handle, path = real_mkstemp(*args, **kwargs)
            made.append(path)
            return handle, path

        with patch("src.modules.Speech_Recognition.remote_stt.tempfile.mkstemp", side_effect=mkstemp), \
                patch("src.modules.Speech_Recognition.remote_stt.requests.post",
                      side_effect=requests.exceptions.ConnectionError("refused")):
            result = transcribe_remote(audio_path=self.audio_path, api_base_url="https://api.groq.com/openai/v1",
                                       api_key="test-key", model="whisper-large-v3")
        self.assertIsNone(result)
        self.assertEqual(len(made), 1)
        self.assertFalse(os.path.exists(made[0]))

    def test_stereo_is_mixed_down_and_loud_samples_are_clipped(self):
        t = np.arange(44100) / 44100
        left = 1.5 * np.sin(2 * np.pi * 220 * t)
        sf.write(self.audio_path, np.stack([left, left], axis=1).astype(np.float32), 44100, subtype="FLOAT")
        _, uploaded, _ = self._transcribe()
        self.assertEqual(uploaded["audio"].ndim, 1)
        # full scale, never wrapped around to the other sign
        peak = int(np.argmax(np.abs(uploaded["audio"])))
        self.assertGreater(np.abs(uploaded["audio"]).max(), 0.99)
        self.assertEqual(np.sign(uploaded["audio"][peak]),
                         np.sign(np.sin(2 * np.pi * 220 * peak / UPLOAD_SAMPLE_RATE)))

    def test_audio_that_cannot_be_read_is_sent_as_it_is(self):
        fake = _make_audio_file(b"not audio")
        self.addCleanup(os.remove, fake)
        with patch("src.modules.Speech_Recognition.remote_stt.requests.post",
                   return_value=_response(200)) as mock_post:
            transcribe_remote(audio_path=fake, api_base_url="https://api.groq.com/openai/v1",
                              api_key="test-key", model="whisper-large-v3")
        self.assertEqual(mock_post.call_args.kwargs["files"]["file"][0], os.path.basename(fake))


class TestRemoteTranscriptionLines(unittest.TestCase):
    def test_segments_are_lines(self):
        result = RemoteTranscription("one two three", segments=[" one two", "  ", " three "])
        self.assertEqual(result.lines(), "one two\nthree")

    def test_text_without_segments(self):
        self.assertEqual(RemoteTranscription("one two").lines(), "one two")


class TestTranscribeRemoteRateLimitRetry(unittest.TestCase):
    """HTTP 429 retry behavior, mirroring the LLM corrector's retry mechanic."""

    def setUp(self):
        self.audio_path = _make_audio_file()

    def tearDown(self):
        if os.path.exists(self.audio_path):
            os.remove(self.audio_path)

    def _kwargs(self, **overrides):
        defaults = {
            "audio_path": self.audio_path,
            "api_base_url": "https://api.groq.com/openai/v1",
            "api_key": "test-key",
            "model": "whisper-large-v3",
            "language": "en",
        }
        defaults.update(overrides)
        return defaults

    @staticmethod
    def _response(status_code, text="ok", headers=None, body=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        resp.headers = headers if headers is not None else {}
        resp.json.return_value = body if body is not None else {"text": "hello world"}
        return resp

    @patch("src.modules.Speech_Recognition.remote_stt.time.sleep")
    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_rate_limited_then_success_retries_and_returns_text(self, mock_post, mock_sleep):
        mock_post.side_effect = [self._response(429), self._response(200)]

        result = transcribe_remote(**self._kwargs())

        self.assertEqual(result.text, "hello world")
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(60.0)

    @patch("src.modules.Speech_Recognition.remote_stt.time.sleep")
    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_retry_after_header_is_used(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            self._response(429, headers={"Retry-After": "5"}),
            self._response(200),
        ]

        result = transcribe_remote(**self._kwargs())

        self.assertEqual(result.text, "hello world")
        mock_sleep.assert_called_once_with(5.0)

    @patch("src.modules.Speech_Recognition.remote_stt.time.sleep")
    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_retry_after_header_is_capped(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            self._response(429, headers={"Retry-After": "99999"}),
            self._response(200),
        ]

        result = transcribe_remote(**self._kwargs())

        self.assertEqual(result.text, "hello world")
        mock_sleep.assert_called_once_with(300)

    @patch("src.modules.Speech_Recognition.remote_stt.time.sleep")
    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_non_numeric_retry_after_header_falls_back_to_retry_wait(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            self._response(429, headers={"Retry-After": "not-a-number"}),
            self._response(200),
        ]

        result = transcribe_remote(**self._kwargs(retry_wait=10.0))

        self.assertEqual(result.text, "hello world")
        mock_sleep.assert_called_once_with(10.0)

    @patch("src.modules.Speech_Recognition.remote_stt.time.sleep")
    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_retry_max_exhausted_returns_none(self, mock_post, mock_sleep):
        mock_post.side_effect = [self._response(429) for _ in range(4)]

        result = transcribe_remote(**self._kwargs(retry_max=3))

        self.assertIsNone(result)
        self.assertEqual(mock_post.call_count, 4)  # 1 initial + 3 retries
        self.assertEqual(mock_sleep.call_count, 3)

    @patch("src.modules.Speech_Recognition.remote_stt.time.sleep")
    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_retry_disabled_fails_open_immediately(self, mock_post, mock_sleep):
        mock_post.return_value = self._response(429)

        result = transcribe_remote(**self._kwargs(retry_on_rate_limit=False))

        self.assertIsNone(result)
        mock_post.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("src.modules.Speech_Recognition.remote_stt.time.sleep")
    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_other_http_error_is_never_retried_even_with_retry_enabled(self, mock_post, mock_sleep):
        mock_post.return_value = self._response(500, text="Internal Server Error")

        result = transcribe_remote(**self._kwargs())

        self.assertIsNone(result)
        mock_post.assert_called_once()
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
