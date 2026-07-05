"""Tests for the Remote STT "Model" field becoming a filterable dropdown.

Covers two things:

1. ``filter_stt_models`` (src/gui/widgets/llm_provider_list.py) — the pure
   filter function that narrows a provider's raw ``/v1/models`` response
   down to speech-to-text-capable models. Verified against real Groq
   ``/models`` metadata shapes (``input_modalities`` / ``output_modalities``)
   as well as plain OpenAI-compatible responses that carry no modality
   metadata at all (falls back to an id name heuristic).
2. The GUI wiring in ``ConversionSettingsForm``: the Model field is now an
   editable combobox (not a QLineEdit), and ``collect_config()`` must read
   its current text.
"""

import os
import unittest

import pytest

# Skip the whole module (instead of failing collection) when PySide6 or its
# native Qt libraries are unavailable, e.g. on a runner without the gui extra.
pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox

from src.gui.settings_tab import ConversionSettingsForm
from src.gui.widgets.llm_provider_list import filter_stt_models

_app = QApplication.instance() or QApplication([])


class TestFilterSttModelsGroqStyle(unittest.TestCase):
    """Groq-style /models responses carry input_modalities/output_modalities."""

    def test_whisper_model_kept(self):
        raw = [
            {
                "id": "whisper-large-v3",
                "input_modalities": ["audio"],
                "output_modalities": ["transcription"],
            },
        ]
        self.assertEqual(filter_stt_models(raw), ["whisper-large-v3"])

    def test_tts_model_excluded(self):
        raw = [
            {
                "id": "whisper-large-v3",
                "input_modalities": ["audio"],
                "output_modalities": ["transcription"],
            },
            {
                "id": "playai-tts-orpheus",
                "input_modalities": ["text"],
                "output_modalities": ["speech"],
            },
        ]
        self.assertEqual(filter_stt_models(raw), ["whisper-large-v3"])

    def test_llm_excluded(self):
        raw = [
            {
                "id": "whisper-large-v3",
                "input_modalities": ["audio"],
                "output_modalities": ["transcription"],
            },
            {
                "id": "llama-3.3-70b-versatile",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
        ]
        self.assertEqual(filter_stt_models(raw), ["whisper-large-v3"])

    def test_mixed_catalog_keeps_only_audio_input_models(self):
        raw = [
            {"id": "whisper-large-v3", "input_modalities": ["audio"],
             "output_modalities": ["transcription"]},
            {"id": "whisper-large-v3-turbo", "input_modalities": ["audio"],
             "output_modalities": ["transcription"]},
            {"id": "playai-tts-orpheus", "input_modalities": ["text"],
             "output_modalities": ["speech"]},
            {"id": "llama-3.3-70b-versatile", "input_modalities": ["text"],
             "output_modalities": ["text"]},
            {"id": "qwen/qwen3-32b", "input_modalities": ["text"],
             "output_modalities": ["text"]},
        ]
        self.assertEqual(
            filter_stt_models(raw),
            ["whisper-large-v3", "whisper-large-v3-turbo"],
        )


class TestFilterSttModelsOpenAiStyle(unittest.TestCase):
    """Plain OpenAI-compatible /models responses carry no modality metadata."""

    def test_whisper_and_transcribe_kept_via_heuristic(self):
        raw = [
            {"id": "whisper-1", "object": "model", "owned_by": "openai"},
            {"id": "gpt-4o-transcribe", "object": "model", "owned_by": "openai"},
            {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
        ]
        self.assertEqual(
            filter_stt_models(raw),
            ["gpt-4o-transcribe", "whisper-1"],
        )

    def test_voxtral_kept_via_heuristic(self):
        """Mistral's speech models don't always carry whisper/transcribe."""
        raw = [
            {"id": "voxtral-mini-latest", "object": "model"},
            {"id": "mistral-large-latest", "object": "model"},
        ]
        self.assertEqual(filter_stt_models(raw), ["voxtral-mini-latest"])


class TestFilterSttModelsMistralCapabilities(unittest.TestCase):
    """Mistral-style /models entries carry a capabilities dict (verified live)."""

    def test_transcription_capability_kept_tts_and_realtime_excluded(self):
        raw = [
            {"id": "voxtral-mini-latest",
             "capabilities": {"audio_transcription": True}},
            {"id": "voxtral-mini-tts-latest",
             "capabilities": {"audio_speech": True, "function_calling": True}},
            {"id": "voxtral-mini-realtime-latest",
             "capabilities": {"audio_transcription_realtime": True}},
            {"id": "mistral-large-latest",
             "capabilities": {"completion_chat": True}},
        ]
        self.assertEqual(filter_stt_models(raw), ["voxtral-mini-latest"])

    def test_llm_excluded_via_heuristic(self):
        raw = [
            {"id": "whisper-1", "object": "model", "owned_by": "openai"},
            {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
            {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
        ]
        self.assertEqual(filter_stt_models(raw), ["whisper-1"])

    def test_heuristic_is_case_insensitive(self):
        raw = [{"id": "Whisper-Large-V3", "object": "model"}]
        self.assertEqual(filter_stt_models(raw), ["Whisper-Large-V3"])


class TestFilterSttModelsFallback(unittest.TestCase):
    """When filtering removes everything, fall back to the full catalog."""

    def test_empty_filter_result_returns_all_models(self):
        raw = [
            {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
            {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
        ]
        self.assertEqual(filter_stt_models(raw), ["gpt-3.5-turbo", "gpt-4o"])

    def test_empty_raw_list_returns_empty(self):
        self.assertEqual(filter_stt_models([]), [])

    def test_entries_without_id_are_ignored(self):
        raw = [{"object": "model"}, {"id": "whisper-1"}]
        self.assertEqual(filter_stt_models(raw), ["whisper-1"])


class TestRemoteSttModelDropdownWidget(unittest.TestCase):
    """The Model field is now an editable QComboBox, not a QLineEdit."""

    def test_model_field_is_editable_combobox(self):
        form = ConversionSettingsForm({})
        self.assertIsInstance(form._remote_stt_model, QComboBox)
        self.assertTrue(form._remote_stt_model.isEditable())

    def test_model_field_defaults_to_config_value(self):
        form = ConversionSettingsForm({"remote_stt_model": "whisper-large-v3"})
        self.assertEqual(form._remote_stt_model.currentText(), "whisper-large-v3")

    def test_fetch_button_disabled_when_remote_stt_off(self):
        form = ConversionSettingsForm({"remote_stt": False})
        self.assertFalse(form._remote_stt_model_fetch_btn.isEnabled())

    def test_fetch_button_enabled_when_remote_stt_on(self):
        form = ConversionSettingsForm({"remote_stt": True})
        self.assertTrue(form._remote_stt_model_fetch_btn.isEnabled())

    def test_on_models_fetched_populates_and_filters(self):
        form = ConversionSettingsForm({"remote_stt": True,
                                        "remote_stt_model": "whisper-large-v3"})
        raw_models = [
            {"id": "whisper-large-v3", "input_modalities": ["audio"],
             "output_modalities": ["transcription"]},
            {"id": "llama-3.3-70b-versatile", "input_modalities": ["text"],
             "output_modalities": ["text"]},
        ]
        form._on_remote_stt_models_fetched(raw_models)
        items = [form._remote_stt_model.itemText(i)
                 for i in range(form._remote_stt_model.count())]
        self.assertEqual(items, ["whisper-large-v3"])
        # Previous selection is preserved.
        self.assertEqual(form._remote_stt_model.currentText(), "whisper-large-v3")


class TestCollectConfigReadsCurrentText(unittest.TestCase):
    """collect_config() must read the combobox's current text, not .text()."""

    def test_collect_config_reads_typed_model_name(self):
        form = ConversionSettingsForm({"remote_stt": True})
        form._remote_stt_model.setEditText("custom-whisper-model")

        config = form.collect_config()

        self.assertEqual(config["remote_stt_model"], "custom-whisper-model")


if __name__ == "__main__":
    unittest.main()
