"""Tests for remote (cloud) speech-to-text transcription.

All ``requests`` calls are mocked -- no real network access happens here.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import requests

from src.modules.Speech_Recognition.remote_stt import (
    transcribe_remote,
    MAX_UPLOAD_BYTES,
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

        self.assertEqual(result, "hello world")
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
        self.assertEqual(result, "hello world")

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

        self.assertEqual(result, "hello world")
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

        self.assertEqual(result, "hello world")
        mock_sleep.assert_called_once_with(5.0)

    @patch("src.modules.Speech_Recognition.remote_stt.time.sleep")
    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_retry_after_header_is_capped(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            self._response(429, headers={"Retry-After": "99999"}),
            self._response(200),
        ]

        result = transcribe_remote(**self._kwargs())

        self.assertEqual(result, "hello world")
        mock_sleep.assert_called_once_with(300)

    @patch("src.modules.Speech_Recognition.remote_stt.time.sleep")
    @patch("src.modules.Speech_Recognition.remote_stt.requests.post")
    def test_non_numeric_retry_after_header_falls_back_to_retry_wait(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            self._response(429, headers={"Retry-After": "not-a-number"}),
            self._response(200),
        ]

        result = transcribe_remote(**self._kwargs(retry_wait=10.0))

        self.assertEqual(result, "hello world")
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
