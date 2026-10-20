"""The language badge of a finished queue item names the pipeline that made the notes."""

import os
import unittest

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui.models import QueueItem
from src.gui.widgets.queue_list import QueueItemWidget

_app = QApplication.instance() or QApplication([])


class TestPipelineInLanguageBadge(unittest.TestCase):
    def _tooltip(self, pipeline):
        item = QueueItem(input_source="https://example.com/video", input_type="url",
                         title="Song Title", status="done")
        widget = QueueItemWidget(item)
        widget.set_result_info({"language": "en", "pipeline": pipeline, "lyrics_source": "plain"})
        return widget._lang_badge.toolTip()

    def test_pipeline_names(self):
        for pipeline, name in (("reference", "Reference"), ("plain", "Plain lyrics alignment"),
                               ("remote", "Remote speech-to-text"), ("whisper", "Transcription")):
            with self.subTest(pipeline=pipeline):
                self.assertIn(f"Pipeline: {name}", self._tooltip(pipeline))


class TestLyricsOfAnotherVersion(unittest.TestCase):
    """Lyrics of a version much shorter than the audio (a live one) get their own label."""

    def test_label_and_tooltip_name_the_lengths(self):
        item = QueueItem(input_source="https://example.com/video", input_type="url",
                         title="Song Title", status="done")
        widget = QueueItemWidget(item)
        widget.set_result_info({"language": "en", "pipeline": "reference", "lyrics_source": "other version",
                                "lyrics_length_s": 245, "audio_length_s": 391})
        self.assertEqual(widget._lyrics_label.text(), "Lyrics of a 4:05 version")
        self.assertIn("245 s long, the audio 391 s", widget._lang_badge.toolTip())


if __name__ == "__main__":
    unittest.main()
