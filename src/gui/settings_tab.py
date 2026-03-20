"""Reusable conversion settings form.

``ConversionSettingsForm`` is embedded both in the unified Settings tab
and in the per-song override dialog.  The old standalone ``SettingsTab``
is replaced by this form.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import importlib.metadata
from pathlib import Path

from .config import _DEFAULTS
from .models import LLMProvider
from .widgets import SettingsCard, ToggleSwitch


def _is_package_available(name: str) -> bool:
    """Check if an optional package is installed."""
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


# ── Scroll-safe spin boxes ───────────────────────────────────────────────
# Prevent accidental value changes when scrolling the settings page.

class _NoScrollSpinBox(QSpinBox):
    """QSpinBox that ignores wheel events unless explicitly focused."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores wheel events unless explicitly focused."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NoScrollComboBox(QComboBox):
    """QComboBox that never reacts to mouse wheel.

    Prevents accidental value changes when scrolling a settings page.
    """

    def wheelEvent(self, event):
        event.ignore()


class ConversionSettingsForm(QWidget):
    """Reusable conversion settings form.

    Used both inline in the unified Settings tab and inside the
    per-song override dialog.
    """

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(4)

        self._build_transcription_section()
        self._build_language_section()
        self._build_postprocessing_section()
        self._build_output_section()
        self._build_device_section()
        self._build_paths_section()

    # ─── Transcription ────────────────────────────────────────────────────

    def _build_transcription_section(self):
        card = SettingsCard("Transcription (Whisper)")

        self._whisper_model = _NoScrollComboBox()
        models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3",
                  "tiny.en", "base.en", "small.en", "medium.en"]
        self._whisper_model.addItems(models)
        self._whisper_model.setCurrentText(self._config.get("whisper_model", "large-v2"))
        card.add_row("Whisper Model", self._whisper_model,
                     "The AI model used for speech-to-text. "
                     "Larger models produce more accurate lyrics but take longer. "
                     "'large-v2' is recommended for best quality. "
                     "Models ending in '.en' are English-only but slightly faster.",
                     reset_callback=lambda: self._whisper_model.setCurrentText(
                         _DEFAULTS["whisper_model"]))

        self._whisper_batch_size = _NoScrollSpinBox()
        self._whisper_batch_size.setRange(1, 64)
        self._whisper_batch_size.setValue(self._config.get("whisper_batch_size", 16))
        card.add_row("Batch Size", self._whisper_batch_size,
                     "How many audio segments Whisper processes at once. "
                     "Higher values are faster but use more GPU memory. "
                     "Reduce to 4–8 if you get out-of-memory errors.",
                     reset_callback=lambda: self._whisper_batch_size.setValue(
                         _DEFAULTS["whisper_batch_size"]))

        self._whisper_compute = _NoScrollComboBox()
        self._whisper_compute.addItems(["auto", "float32", "float16", "int8"])
        raw_ct = self._config.get("whisper_compute_type", "")
        self._whisper_compute.setCurrentText(raw_ct if raw_ct else "auto")
        card.add_row("Compute Type", self._whisper_compute,
                     "Numeric precision for Whisper inference. "
                     "'auto' picks float16 on GPU (fast) or int8 on CPU (memory-efficient). "
                     "float32 is most accurate but slowest. "
                     "int8 uses least memory at a small accuracy trade-off.",
                     reset_callback=lambda: self._whisper_compute.setCurrentText("auto"))

        self._align_model = QLineEdit()
        self._align_model.setPlaceholderText("e.g., gigant/romanian-wav2vec2")
        self._align_model.setText(self._config.get("whisper_align_model", ""))
        card.add_row("Custom Align Model", self._align_model,
                     "Override the default forced-alignment model (wav2vec2). "
                     "Useful for languages where the default model performs poorly. "
                     "Enter a HuggingFace model ID. Leave empty to use the built-in model.",
                     reset_callback=lambda: self._align_model.setText(
                         _DEFAULTS["whisper_align_model"]))

        self._demucs_model = _NoScrollComboBox()
        demucs_models = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi",
                        "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"]
        self._demucs_model.addItems(demucs_models)
        self._demucs_model.setCurrentText(self._config.get("demucs_model", "htdemucs"))
        card.add_row("Vocal Separation Model", self._demucs_model,
                     "The AI model that isolates vocals from background music. "
                     "'htdemucs' is the default and works well for most songs. "
                     "'htdemucs_ft' is fine-tuned and may be slightly better. "
                     "'mdx_extra' can handle complex mixes but is slower.",
                     reset_callback=lambda: self._demucs_model.setCurrentText(
                         _DEFAULTS["demucs_model"]))

        self._main_layout.addWidget(card)

    # ─── Language ─────────────────────────────────────────────────────────

    def _build_language_section(self):
        card = SettingsCard("Language")

        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        self._lang_mode_group = QButtonGroup(self)
        self._lang_auto = QRadioButton("Auto-detect")
        self._lang_manual = QRadioButton("Manual selection")
        self._lang_mode_group.addButton(self._lang_auto, 0)
        self._lang_mode_group.addButton(self._lang_manual, 1)
        is_manual = self._config.get("language_mode") == "manual"
        self._lang_auto.setChecked(not is_manual)
        self._lang_manual.setChecked(is_manual)
        mode_row.addWidget(self._lang_auto)
        mode_row.addWidget(self._lang_manual)
        mode_row.addStretch()
        card.add_widget(card._wrap_row(mode_row))

        self._language_combo = _NoScrollComboBox()
        languages = [
            "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt",
            "ru", "ko", "ar", "hi", "pl", "sv", "tr", "fi", "da", "no",
            "cs", "ro", "hu", "el", "he", "th", "id", "vi", "ms", "tl",
        ]
        self._language_combo.addItems(languages)
        self._language_combo.setCurrentText(self._config.get("language", "en"))
        self._language_combo.setEnabled(is_manual)
        card.add_row("Language", self._language_combo,
                     "The language of the song lyrics. "
                     "Only used when 'Manual selection' is chosen above.",
                     reset_callback=lambda: self._language_combo.setCurrentText(
                         _DEFAULTS["language"]))

        self._lang_mode_group.idClicked.connect(
            lambda mid: self._language_combo.setEnabled(mid == 1)
        )

        card.add_info(
            "Auto-detect uses Whisper's built-in language detection. "
            "Manual selection forces a specific language."
        )

        self._main_layout.addWidget(card)

    # ─── Post-processing ──────────────────────────────────────────────────

    def _build_postprocessing_section(self):
        card = SettingsCard("Post-Processing")

        # Hyphenation
        self._hyphenation = ToggleSwitch(checked=self._config.get("hyphenation", True))
        card.add_toggle_row("Hyphenation", self._hyphenation,
                           "Split words into syllables (e.g., 'beautiful' → 'beau-ti-ful'). "
                           "Required by most karaoke games for proper syllable highlighting. "
                           "Disable only if you want whole-word notes.",
                           reset_callback=lambda: self._hyphenation.setChecked(
                               _DEFAULTS["hyphenation"]))

        # Quantize to key
        self._quantize = ToggleSwitch(
            checked=not self._config.get("disable_quantization", False)
        )
        card.add_toggle_row("Quantize to Key", self._quantize,
                           "Snap detected pitches to the song's musical key. "
                           "Reduces off-key notes and makes scoring more forgiving. "
                           "Disable for songs with unusual scales or heavy chromaticism.",
                           reset_callback=lambda: self._quantize.setChecked(
                               not _DEFAULTS["disable_quantization"]))

        # Vocal center correction
        self._vocal_center = ToggleSwitch(
            checked=not self._config.get("disable_vocal_center", False)
        )
        card.add_toggle_row("Vocal Center Correction", self._vocal_center,
                           "Fixes notes that are consistently shifted by one octave. "
                           "Compares detected pitches against the typical vocal range "
                           "(roughly C3–C6) and corrects ±12 semitone errors.",
                           reset_callback=lambda: self._vocal_center.setChecked(
                               not _DEFAULTS["disable_vocal_center"]))

        # Onset correction
        self._onset_correction = ToggleSwitch(
            checked=not self._config.get("disable_onset_correction", False)
        )
        card.add_toggle_row("Onset Timing Correction", self._onset_correction,
                           "Detects the exact moment each syllable starts in the audio "
                           "and adjusts note start times accordingly. "
                           "Improves timing accuracy, especially for fast-paced songs.",
                           reset_callback=lambda: self._onset_correction.setChecked(
                               not _DEFAULTS["disable_onset_correction"]))

        # Vocal separation
        self._separation = ToggleSwitch(
            checked=not self._config.get("disable_separation", False)
        )
        card.add_toggle_row("Vocal Separation", self._separation,
                           "Use AI (Demucs) to isolate the vocal track from the full mix. "
                           "Dramatically improves transcription and pitch accuracy. "
                           "Only disable if your input is already an isolated vocal track.",
                           reset_callback=lambda: self._separation.setChecked(
                               not _DEFAULTS["disable_separation"]))

        # Denoise
        self._denoise = ToggleSwitch(
            checked=not self._config.get("disable_denoise_track_noise", False)
        )
        card.add_toggle_row("Noise Reduction", self._denoise,
                           "Remove background noise from the vocal track before analysis. "
                           "Uses adaptive noise-floor tracking to reduce hiss and hum. "
                           "Usually improves results but may remove quiet vocal nuances.",
                           reset_callback=lambda: self._denoise.setChecked(
                               not _DEFAULTS["disable_denoise_track_noise"]))

        card.add_separator()

        # Denoise parameters
        self._denoise_nr = _NoScrollDoubleSpinBox()
        self._denoise_nr.setRange(0.01, 97.0)
        self._denoise_nr.setValue(self._config.get("denoise_nr", 20))
        self._denoise_nr.setSuffix(" dB")
        card.add_row("Noise Reduction Level", self._denoise_nr,
                     "How aggressively noise is removed (in dB). "
                     "Higher values remove more noise but risk distorting the vocals. "
                     "Default (20 dB) is a safe balance.",
                     reset_callback=lambda: self._denoise_nr.setValue(
                         _DEFAULTS["denoise_nr"]))

        self._denoise_nf = _NoScrollSpinBox()
        self._denoise_nf.setRange(-80, -20)
        self._denoise_nf.setValue(self._config.get("denoise_nf", -80))
        self._denoise_nf.setSuffix(" dB")
        card.add_row("Noise Floor", self._denoise_nf,
                     "The threshold below which audio is considered noise (in dB). "
                     "Lower values (-80 dB) only treat very quiet sounds as noise. "
                     "Raise towards -40 dB for recordings with louder background noise.",
                     reset_callback=lambda: self._denoise_nf.setValue(
                         _DEFAULTS["denoise_nf"]))

        self._main_layout.addWidget(card)

        self._build_experimental_section()
        self._build_refinement_section()
        self._build_llm_section()
        self._build_scoring_section()

    # ─── Experimental Features ────────────────────────────────────────────

    def _build_experimental_section(self):
        card = SettingsCard("Experimental Features")

        card.add_info(
            "These features may improve results for specific songs but can "
            "reduce overall quality on average. Use them selectively — "
            "try them on individual songs via per-song settings before "
            "enabling them globally."
        )

        self._syllable_split = ToggleSwitch(
            checked=self._config.get("syllable_split", False)
        )
        card.add_toggle_row("Syllable-Level Splitting", self._syllable_split,
                           "Keep each syllable as a separate note, even when adjacent "
                           "syllables have the same pitch. Without this, same-pitch "
                           "syllables get merged into one long note. "
                           "Recommended for karaoke games that highlight per syllable.",
                           reset_callback=lambda: self._syllable_split.setChecked(
                               _DEFAULTS["syllable_split"]))

        self._vocal_gap_fill = ToggleSwitch(
            checked=self._config.get("vocal_gap_fill", False)
        )
        card.add_toggle_row("Vocal Gap Fill", self._vocal_gap_fill,
                           "Detect vocal segments that Whisper couldn't transcribe "
                           "(e.g., ad-libs, melismas, humming) and insert placeholder "
                           "notes marked with '~'. Helps cover the full vocal performance.",
                           reset_callback=lambda: self._vocal_gap_fill.setChecked(
                               _DEFAULTS["vocal_gap_fill"]))

        self._keep_numbers = ToggleSwitch(
            checked=self._config.get("keep_numbers", False)
        )
        card.add_toggle_row("Keep Numbers as Digits", self._keep_numbers,
                           "Show numbers as digits (1, 2, 3) instead of words "
                           "(one, two, three). Useful for songs that prominently "
                           "feature counting or number sequences.",
                           reset_callback=lambda: self._keep_numbers.setChecked(
                               _DEFAULTS["keep_numbers"]))

        self._main_layout.addWidget(card)

    # ─── Refinement ────────────────────────────────────────────────────

    def _build_refinement_section(self):
        card = SettingsCard("Refinement (experimental)")

        card.add_info(
            "Re-analyses the vocal audio after initial note generation "
            "to correct pitch values and note timings. This is a second "
            "pass that fixes deviations the initial detection missed."
        )

        self._refine_from_vocal = ToggleSwitch(
            checked=self._config.get("refine_from_vocal", False)
        )
        card.add_toggle_row("Enable Refinement", self._refine_from_vocal,
                           "Run a reverse-scoring refinement pass after note generation.",
                           reset_callback=lambda: self._refine_from_vocal.setChecked(
                               _DEFAULTS.get("refine_from_vocal", False)))

        self._refine_pitch = ToggleSwitch(
            checked=self._config.get("refine_pitch", True)
        )
        card.add_toggle_row("Refine Pitch", self._refine_pitch,
                           "Correct note pitches by comparing against the vocal audio.",
                           reset_callback=lambda: self._refine_pitch.setChecked(
                               _DEFAULTS.get("refine_pitch", True)))

        self._refine_timing = ToggleSwitch(
            checked=self._config.get("refine_timing", True)
        )
        card.add_toggle_row("Refine Timing", self._refine_timing,
                           "Correct note start/end times using detected audio onsets.",
                           reset_callback=lambda: self._refine_timing.setChecked(
                               _DEFAULTS.get("refine_timing", True)))

        # Difficulty combo
        self._refine_difficulty = _NoScrollComboBox()
        self._refine_difficulty.addItems(["easy", "medium", "hard"])
        current_diff = self._config.get("refine_difficulty", "easy")
        idx = self._refine_difficulty.findText(current_diff)
        if idx >= 0:
            self._refine_difficulty.setCurrentIndex(idx)
        card.add_row("Difficulty Tolerance", self._refine_difficulty,
                     "Easy: lenient (only fix gross errors, +2 HT tolerance). "
                     "Medium: moderate (+1 HT). Hard: strict (correct any deviation).",
                     reset_callback=lambda: self._refine_difficulty.setCurrentText(
                         _DEFAULTS.get("refine_difficulty", "easy")))

        # Hit ratio threshold
        self._refine_hit_ratio = _NoScrollDoubleSpinBox()
        self._refine_hit_ratio.setRange(0.0, 1.0)
        self._refine_hit_ratio.setSingleStep(0.05)
        self._refine_hit_ratio.setDecimals(2)
        self._refine_hit_ratio.setValue(
            self._config.get("refine_hit_ratio", 0.5))
        card.add_row("Hit Ratio Threshold", self._refine_hit_ratio,
                     "Notes scoring below this hit ratio (0.0-1.0) are pitch-corrected by the game engine.",
                     reset_callback=lambda: self._refine_hit_ratio.setValue(
                         _DEFAULTS.get("refine_hit_ratio", 0.5)))

        # Timing threshold
        self._refine_timing_threshold = _NoScrollDoubleSpinBox()
        self._refine_timing_threshold.setRange(5.0, 200.0)
        self._refine_timing_threshold.setSingleStep(5.0)
        self._refine_timing_threshold.setSuffix(" ms")
        self._refine_timing_threshold.setValue(
            self._config.get("refine_timing_threshold", 30.0))
        card.add_row("Timing Threshold", self._refine_timing_threshold,
                     "Milliseconds deviation before correcting note start/end times.",
                     reset_callback=lambda: self._refine_timing_threshold.setValue(
                         _DEFAULTS.get("refine_timing_threshold", 30.0)))

        # Vibrato damping window
        self._refine_vibrato_window = _NoScrollSpinBox()
        self._refine_vibrato_window.setRange(3, 15)
        self._refine_vibrato_window.setValue(
            self._config.get("refine_vibrato_window", 5))
        card.add_row("Vibrato Smoothing Window", self._refine_vibrato_window,
                     "Number of frames for the moving-average vibrato damping filter.",
                     reset_callback=lambda: self._refine_vibrato_window.setValue(
                         _DEFAULTS.get("refine_vibrato_window", 5)))

        # Vibrato threshold
        self._refine_vibrato_threshold = _NoScrollDoubleSpinBox()
        self._refine_vibrato_threshold.setRange(10.0, 200.0)
        self._refine_vibrato_threshold.setSingleStep(10.0)
        self._refine_vibrato_threshold.setSuffix(" cents")
        self._refine_vibrato_threshold.setValue(
            self._config.get("refine_vibrato_threshold", 50.0))
        card.add_row("Vibrato Threshold", self._refine_vibrato_threshold,
                     "Minimum pitch spread in cents to trigger vibrato damping. "
                     "50 cents = half a semitone.",
                     reset_callback=lambda: self._refine_vibrato_threshold.setValue(
                         _DEFAULTS.get("refine_vibrato_threshold", 50.0)))

        # Toggle sub-settings with main switch
        def _toggle_refine(on):
            self._refine_pitch.setEnabled(on)
            self._refine_timing.setEnabled(on)
            self._refine_difficulty.setEnabled(on)
            self._refine_hit_ratio.setEnabled(on)
            self._refine_timing_threshold.setEnabled(on)
            self._refine_vibrato_window.setEnabled(on)
            self._refine_vibrato_threshold.setEnabled(on)

        self._refine_from_vocal.toggled.connect(_toggle_refine)
        _toggle_refine(self._refine_from_vocal.isChecked())

        self._main_layout.addWidget(card)

    # ─── LLM Lyric Correction ────────────────────────────────────────────

    def _build_llm_section(self):
        card = SettingsCard("LLM Lyric Correction")

        self._llm_correct = ToggleSwitch(
            checked=self._config.get("llm_correct", False)
        )
        card.add_toggle_row("Enable LLM Correction", self._llm_correct,
                           "Send transcribed lyrics to a Large Language Model (AI) "
                           "for grammar and spelling correction. "
                           "Requires an LLM provider configured in Settings. "
                           "Models with 32B+ parameters recommended (e.g., Qwen3-32b via Groq).",
                           reset_callback=lambda: self._llm_correct.setChecked(
                               _DEFAULTS["llm_correct"]))

        # Provider selector (populated externally via set_llm_providers)
        self._llm_provider = _NoScrollComboBox()
        self._llm_provider.setEnabled(self._llm_correct.isChecked())
        card.add_row("LLM Provider", self._llm_provider,
                     "Which LLM service to use for lyric correction. "
                     "Configure providers in the Settings tab under 'LLM Providers'. "
                     "Groq offers free API access with rate limits.")

        # Retry settings
        self._llm_retry = ToggleSwitch(
            checked=self._config.get("llm_retry_on_rate_limit", True)
        )
        card.add_toggle_row("Retry on rate limit (429)", self._llm_retry,
                           "Automatically retry when the LLM provider returns "
                           "HTTP 429 (Too Many Requests). Common with free tier APIs like Groq.",
                           reset_callback=lambda: self._llm_retry.setChecked(
                               _DEFAULTS["llm_retry_on_rate_limit"]))

        self._llm_retry_wait = QSpinBox()
        self._llm_retry_wait.setRange(5, 300)
        self._llm_retry_wait.setSuffix(" s")
        self._llm_retry_wait.setValue(self._config.get("llm_retry_wait", 60))
        card.add_row("Retry wait time", self._llm_retry_wait,
                     "How many seconds to wait before retrying after a rate limit error.",
                     reset_callback=lambda: self._llm_retry_wait.setValue(
                         _DEFAULTS["llm_retry_wait"]))

        self._llm_retry_max = QSpinBox()
        self._llm_retry_max.setRange(1, 10)
        self._llm_retry_max.setValue(self._config.get("llm_retry_max", 3))
        card.add_row("Max retries per chunk", self._llm_retry_max,
                     "Maximum number of retries per text chunk before giving up.",
                     reset_callback=lambda: self._llm_retry_max.setValue(
                         _DEFAULTS["llm_retry_max"]))

        # Enable/disable LLM sub-settings together
        def _toggle_llm(on):
            self._llm_provider.setEnabled(on)
            self._llm_retry.setEnabled(on)
            self._llm_retry_wait.setEnabled(on)
            self._llm_retry_max.setEnabled(on)

        self._llm_correct.toggled_signal.connect(_toggle_llm)
        _toggle_llm(self._llm_correct.isChecked())

        self._main_layout.addWidget(card)

    # ─── Scoring ─────────────────────────────────────────────────────────

    def _build_scoring_section(self):
        card = SettingsCard("Scoring")

        scoring_available = _is_package_available("ultrastar-score")
        self._calculate_score = ToggleSwitch(
            checked=self._config.get("calculate_score", False)
            and scoring_available
        )
        card.add_toggle_row("Calculate Score", self._calculate_score,
                           "After conversion, play back the generated notes against the audio "
                           "and calculate a karaoke score. Helps evaluate conversion quality. "
                           "Requires the optional 'ultrastar-score' package.",
                           reset_callback=lambda: self._calculate_score.setChecked(
                               _DEFAULTS["calculate_score"]))
        if not scoring_available:
            self._calculate_score.setEnabled(False)
            app_dir = Path(__file__).resolve().parent.parent.parent
            import sys
            if sys.platform == "win32":
                cd_cmd = f"cd /D {app_dir}"
            else:
                cd_cmd = f"cd {app_dir}"
            card.add_info(
                "ultrastar-score package not installed. To install:<br>"
                "1. Close UltraSinger<br>"
                f"2. Open a terminal and run:&nbsp; <b>{cd_cmd}</b><br>"
                "3. Run:&nbsp; <b>uv sync --extra scoring</b><br>"
                "4. Restart UltraSinger"
            )

        self._main_layout.addWidget(card)

    # ─── Output ───────────────────────────────────────────────────────────

    def _build_output_section(self):
        card = SettingsCard("Output")

        self._format_version = _NoScrollComboBox()
        self._format_version.addItems(["0.3.0", "1.0.0", "1.1.0", "1.2.0"])
        self._format_version.setCurrentText(
            self._config.get("format_version", "1.2.0")
        )
        card.add_row("Format Version", self._format_version,
                     "UltraStar TXT format version. "
                     "1.2.0 is the latest standard with full metadata support. "
                     "Use older versions only for compatibility with legacy software.",
                     reset_callback=lambda: self._format_version.setCurrentText(
                         _DEFAULTS["format_version"]))

        self._create_plot = ToggleSwitch(
            checked=self._config.get("create_plot", False)
        )
        card.add_toggle_row("Generate Plots", self._create_plot,
                           "Create visual charts showing detected pitches and note timing. "
                           "Useful for debugging conversion quality. "
                           "Saved as PNG images in the output folder.",
                           reset_callback=lambda: self._create_plot.setChecked(
                               _DEFAULTS["create_plot"]))

        self._create_midi = ToggleSwitch(
            checked=self._config.get("create_midi", True)
        )
        card.add_toggle_row("Generate MIDI", self._create_midi,
                           "Export the detected melody as a MIDI file. "
                           "Can be opened in music editors like MuseScore or FL Studio "
                           "for manual fine-tuning.",
                           reset_callback=lambda: self._create_midi.setChecked(
                               _DEFAULTS["create_midi"]))

        self._create_chunks = ToggleSwitch(
            checked=self._config.get("create_audio_chunks", False)
        )
        card.add_toggle_row("Create Audio Chunks", self._create_chunks,
                           "Save each detected note as a separate audio file. "
                           "Useful for debugging which audio segment maps to which note.",
                           reset_callback=lambda: self._create_chunks.setChecked(
                               _DEFAULTS["create_audio_chunks"]))

        self._create_karaoke = ToggleSwitch(
            checked=self._config.get("create_karaoke", True)
        )
        card.add_toggle_row("Create Karaoke File", self._create_karaoke,
                           "Generate an instrumental-only audio file (vocals removed). "
                           "Used as background music during karaoke playback.",
                           reset_callback=lambda: self._create_karaoke.setChecked(
                               _DEFAULTS["create_karaoke"]))

        self._keep_audio_in_video = ToggleSwitch(
            checked=self._config.get("keep_audio_in_video", False)
        )
        card.add_toggle_row("Keep Audio in Video", self._keep_audio_in_video,
                           "Keep the full audio track (vocals + instrumental) embedded "
                           "in the output video file. Makes the video self-contained so "
                           "it can be re-imported into UltraSinger with different settings "
                           "without downloading again. Increases file size.",
                           reset_callback=lambda: self._keep_audio_in_video.setChecked(
                               _DEFAULTS["keep_audio_in_video"]))

        self._write_settings_info = ToggleSwitch(
            checked=self._config.get("write_settings_info", False)
        )
        card.add_toggle_row("Write Settings Info", self._write_settings_info,
                           "Save a 'ultrasinger_parameter.info' file in the output folder "
                           "documenting all conversion settings used. If scoring is enabled, "
                           "score results are included. Useful for comparing results across "
                           "multiple conversion runs with different settings.",
                           reset_callback=lambda: self._write_settings_info.setChecked(
                               _DEFAULTS["write_settings_info"]))

        self._keep_cache = ToggleSwitch(
            checked=self._config.get("keep_cache", False)
        )
        card.add_toggle_row("Keep Cache", self._keep_cache,
                           "Keep intermediate processing files (separated audio, "
                           "raw transcription data) after conversion. "
                           "Speeds up re-processing but uses more disk space.",
                           reset_callback=lambda: self._keep_cache.setChecked(
                               _DEFAULTS["keep_cache"]))

        self._main_layout.addWidget(card)

    # ─── Device / Performance ─────────────────────────────────────────────

    def _build_device_section(self):
        card = SettingsCard("Device / Performance")

        self._force_cpu = ToggleSwitch(
            checked=self._config.get("force_cpu", False)
        )
        card.add_toggle_row("Force CPU Only", self._force_cpu,
                           "Run all processing on the CPU, ignoring any NVIDIA GPU. "
                           "Slower but works on systems without CUDA support. "
                           "Enable this if you get CUDA errors.",
                           reset_callback=lambda: self._force_cpu.setChecked(
                               _DEFAULTS["force_cpu"]))

        self._force_whisper_cpu = ToggleSwitch(
            checked=self._config.get("force_whisper_cpu", False)
        )
        card.add_toggle_row("Force Whisper CPU", self._force_whisper_cpu,
                           "Run only the Whisper transcription on CPU while keeping "
                           "other stages (vocal separation, pitch detection) on GPU. "
                           "Useful when Whisper alone exceeds your GPU memory.",
                           reset_callback=lambda: self._force_whisper_cpu.setChecked(
                               _DEFAULTS["force_whisper_cpu"]))

        # BPM override
        self._bpm_override = QLineEdit()
        self._bpm_override.setPlaceholderText("Auto-detect")
        bpm_validator = QDoubleValidator(1.0, 500.0, 2, self)
        bpm_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self._bpm_override.setValidator(bpm_validator)
        bpm_val = self._config.get("bpm_override", "")
        if bpm_val:
            self._bpm_override.setText(str(bpm_val))
        card.add_row("BPM Override", self._bpm_override,
                     "Manually set the song's tempo in beats per minute. "
                     "Leave empty to auto-detect the BPM. "
                     "Use this when auto-detection picks the wrong tempo "
                     "(common with very slow or very fast songs).",
                     reset_callback=lambda: self._bpm_override.setText(
                         str(_DEFAULTS["bpm_override"])))

        # Octave shift
        self._octave_shift = QLineEdit()
        self._octave_shift.setPlaceholderText("None")
        self._octave_shift.setValidator(QIntValidator(-10, 10, self))
        oct_val = self._config.get("octave_shift", "")
        if oct_val:
            self._octave_shift.setText(str(oct_val))
        card.add_row("Octave Shift", self._octave_shift,
                     "Shift all detected notes up or down by this many octaves. "
                     "Use +1 if notes appear too low, -1 if too high. "
                     "Leave empty for automatic octave detection.",
                     reset_callback=lambda: self._octave_shift.setText(
                         str(_DEFAULTS["octave_shift"])))

        self._main_layout.addWidget(card)

    # ─── Paths ────────────────────────────────────────────────────────────

    def _build_paths_section(self):
        card = SettingsCard("Paths")

        # MuseScore
        self._musescore_path = QLineEdit()
        self._musescore_path.setPlaceholderText("Optional: Path to MuseScore")
        self._musescore_path.setText(self._config.get("musescore_path", ""))
        self._musescore_path.setToolTip(
            "Path to the MuseScore executable. "
            "Used to convert generated MIDI files into sheet music (PDF). "
            "Leave empty if MuseScore is not installed."
        )
        ms_row = QHBoxLayout()
        ms_row.addWidget(self._musescore_path, 1)
        ms_browse = QPushButton("Browse")
        ms_browse.clicked.connect(
            lambda: self._browse_file(self._musescore_path, "MuseScore Executable (*)")
        )
        ms_row.addWidget(ms_browse)
        card.add_row("MuseScore Path", QWidget(),
                     "Path to the MuseScore executable. "
                     "Used to convert generated MIDI files into sheet music (PDF). "
                     "Leave empty if MuseScore is not installed.")
        # Replace the empty widget with the row
        card.remove_last_item()
        card.add_layout(ms_row)

        # FFmpeg
        self._ffmpeg_path = QLineEdit()
        self._ffmpeg_path.setPlaceholderText("Optional: Custom FFmpeg path")
        self._ffmpeg_path.setText(self._config.get("ffmpeg_path", ""))
        self._ffmpeg_path.setToolTip(
            "Custom path to FFmpeg. "
            "FFmpeg handles audio/video format conversion. "
            "Leave empty to use the system-installed version."
        )
        ff_row = QHBoxLayout()
        ff_row.addWidget(self._ffmpeg_path, 1)
        ff_browse = QPushButton("Browse")
        ff_browse.clicked.connect(
            lambda: self._browse_file(self._ffmpeg_path, "FFmpeg (*)")
        )
        ff_row.addWidget(ff_browse)
        card.add_layout(ff_row)

        self._main_layout.addWidget(card)

    def _browse_file(self, line_edit: QLineEdit, filter_text: str):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_text)
        if path:
            line_edit.setText(path)

    # ─── LLM Provider API ────────────────────────────────────────────────

    def set_llm_providers(self, providers: list[LLMProvider],
                          selected_id: str = ""):
        """Populate the LLM provider combobox.

        Args:
            providers: Available LLM providers.
            selected_id: ID of the provider to pre-select.  If empty,
                         the default provider is selected.
        """
        self._llm_provider.clear()
        default_idx = 0
        for i, p in enumerate(providers):
            label = f"{p.name} ({p.default_model})" if p.name else p.default_model
            self._llm_provider.addItem(label, userData=p.id)
            if selected_id and p.id == selected_id:
                default_idx = i
            elif not selected_id and p.is_default:
                default_idx = i
        if self._llm_provider.count() > 0:
            self._llm_provider.setCurrentIndex(default_idx)

    def get_selected_provider_id(self) -> str:
        """Return the ID of the currently selected LLM provider."""
        return self._llm_provider.currentData() or ""

    # ─── Public API ───────────────────────────────────────────────────────

    def collect_config(self) -> dict:
        """Collect all conversion settings into a config dictionary."""
        return {
            "whisper_model": self._whisper_model.currentText(),
            "whisper_batch_size": self._whisper_batch_size.value(),
            "whisper_compute_type": None if self._whisper_compute.currentText() == "auto"
                                     else self._whisper_compute.currentText(),
            "whisper_align_model": self._align_model.text(),
            "demucs_model": self._demucs_model.currentText(),
            "language_mode": "manual" if self._lang_manual.isChecked() else "auto",
            "language": self._language_combo.currentText(),
            "hyphenation": self._hyphenation.isChecked(),
            "disable_separation": not self._separation.isChecked(),
            "disable_quantization": not self._quantize.isChecked(),
            "disable_vocal_center": not self._vocal_center.isChecked(),
            "disable_onset_correction": not self._onset_correction.isChecked(),
            "disable_denoise_track_noise": not self._denoise.isChecked(),
            "denoise_nr": self._denoise_nr.value(),
            "denoise_nf": self._denoise_nf.value(),
            "syllable_split": self._syllable_split.isChecked(),
            "vocal_gap_fill": self._vocal_gap_fill.isChecked(),
            "keep_numbers": self._keep_numbers.isChecked(),
            "llm_correct": self._llm_correct.isChecked(),
            "llm_provider_id": self.get_selected_provider_id(),
            "llm_retry_on_rate_limit": self._llm_retry.isChecked(),
            "llm_retry_wait": self._llm_retry_wait.value(),
            "llm_retry_max": self._llm_retry_max.value(),
            "calculate_score": self._calculate_score.isChecked(),
            "format_version": self._format_version.currentText(),
            "create_plot": self._create_plot.isChecked(),
            "create_midi": self._create_midi.isChecked(),
            "create_audio_chunks": self._create_chunks.isChecked(),
            "create_karaoke": self._create_karaoke.isChecked(),
            "keep_audio_in_video": self._keep_audio_in_video.isChecked(),
            "write_settings_info": self._write_settings_info.isChecked(),
            "keep_cache": self._keep_cache.isChecked(),
            "force_cpu": self._force_cpu.isChecked(),
            "force_whisper_cpu": self._force_whisper_cpu.isChecked(),
            "bpm_override": self._bpm_override.text(),
            "octave_shift": self._octave_shift.text(),
            "musescore_path": self._musescore_path.text(),
            "ffmpeg_path": self._ffmpeg_path.text(),
            "refine_from_vocal": self._refine_from_vocal.isChecked(),
            "refine_pitch": self._refine_pitch.isChecked(),
            "refine_timing": self._refine_timing.isChecked(),
            "refine_difficulty": self._refine_difficulty.currentText(),
            "refine_hit_ratio": self._refine_hit_ratio.value(),
            "refine_timing_threshold": self._refine_timing_threshold.value(),
            "refine_vibrato_window": self._refine_vibrato_window.value(),
            "refine_vibrato_threshold": self._refine_vibrato_threshold.value(),
        }
