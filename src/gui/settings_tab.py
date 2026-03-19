"""Conversion settings panel with all UltraSinger parameters."""

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
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .config import _DEFAULTS
from .widgets import SettingsCard, ToggleSwitch


# ── Scroll-safe spin boxes ───────────────────────────────────────────────
# Prevent accidental value changes when scrolling the settings page.
# The wheel event is only accepted when the widget has explicit focus
# (i.e. the user clicked into it), not when it merely receives focus
# from being under the mouse cursor while scrolling.

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
    """QComboBox that ignores wheel events unless explicitly focused."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class SettingsTab(QWidget):
    """Conversion settings form with all UltraSinger CLI parameters.

    Input source selection (video URL / local file) is handled by
    the sidebar, not by this tab. This tab only contains conversion options.
    """

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config

        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self._main_layout = QVBoxLayout(container)
        self._main_layout.setContentsMargins(24, 24, 24, 24)
        self._main_layout.setSpacing(16)

        # Section header
        header = QLabel("Conversion Settings")
        header.setObjectName("sectionHeader")
        self._main_layout.addWidget(header)

        self._build_transcription_section()
        self._build_language_section()
        self._build_pitch_section()
        self._build_postprocessing_section()
        self._build_output_section()
        self._build_device_section()
        self._build_paths_section()

        self._main_layout.addStretch(1)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ─── Transcription ────────────────────────────────────────────────────

    def _build_transcription_section(self):
        card = SettingsCard("Transcription (Whisper)")

        self._whisper_model = _NoScrollComboBox()
        models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3",
                  "tiny.en", "base.en", "small.en", "medium.en"]
        self._whisper_model.addItems(models)
        self._whisper_model.setCurrentText(self._config.get("whisper_model", "large-v2"))
        card.add_row("Whisper Model", self._whisper_model,
                     "Larger models are more accurate but slower",
                     reset_callback=lambda: self._whisper_model.setCurrentText(
                         _DEFAULTS["whisper_model"]))

        self._whisper_batch_size = _NoScrollSpinBox()
        self._whisper_batch_size.setRange(1, 64)
        self._whisper_batch_size.setValue(self._config.get("whisper_batch_size", 16))
        card.add_row("Batch Size", self._whisper_batch_size,
                     "Reduce if running out of GPU memory",
                     reset_callback=lambda: self._whisper_batch_size.setValue(
                         _DEFAULTS["whisper_batch_size"]))

        self._whisper_compute = _NoScrollComboBox()
        self._whisper_compute.addItems(["", "float32", "float16", "int8"])
        self._whisper_compute.setCurrentText(self._config.get("whisper_compute_type", ""))
        card.add_row("Compute Type", self._whisper_compute,
                     "Leave empty for auto-detection (float16 on GPU, int8 on CPU)",
                     reset_callback=lambda: self._whisper_compute.setCurrentText(
                         _DEFAULTS["whisper_compute_type"]))

        self._align_model = QLineEdit()
        self._align_model.setPlaceholderText("e.g., gigant/romanian-wav2vec2")
        self._align_model.setText(self._config.get("whisper_align_model", ""))
        card.add_row("Custom Align Model", self._align_model,
                     "HuggingFace model for non-English alignment",
                     reset_callback=lambda: self._align_model.setText(
                         _DEFAULTS["whisper_align_model"]))

        self._demucs_model = _NoScrollComboBox()
        demucs_models = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi",
                        "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"]
        self._demucs_model.addItems(demucs_models)
        self._demucs_model.setCurrentText(self._config.get("demucs_model", "htdemucs"))
        card.add_row("Vocal Separation Model", self._demucs_model,
                     "Model used for separating vocals from instrumentals",
                     reset_callback=lambda: self._demucs_model.setCurrentText(
                         _DEFAULTS["demucs_model"]))

        self._main_layout.addWidget(card)

    # ─── Language ─────────────────────────────────────────────────────────

    def _build_language_section(self):
        card = SettingsCard("Language")

        mode_row = QHBoxLayout()
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
        card.add_layout(mode_row)

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

    # ─── Pitch ────────────────────────────────────────────────────────────

    def _build_pitch_section(self):
        card = SettingsCard("Pitch Detection")
        card.add_info(
            "Pitch Detection: SwiftF0 (ONNX-based, no configuration needed). "
            "Uses combined AKF/AMDF autocorrelation for accurate pitch tracking."
        )
        self._main_layout.addWidget(card)

    # ─── Post-processing ──────────────────────────────────────────────────

    def _build_postprocessing_section(self):
        card = SettingsCard("Post-Processing")

        # Hyphenation
        self._hyphenation = ToggleSwitch(checked=self._config.get("hyphenation", True))
        card.add_toggle_row("Hyphenation", self._hyphenation,
                           "Split words into syllables using hyphenation rules",
                           reset_callback=lambda: self._hyphenation.setChecked(
                               _DEFAULTS["hyphenation"]))

        # Quantize to key
        self._quantize = ToggleSwitch(
            checked=not self._config.get("disable_quantization", False)
        )
        card.add_toggle_row("Quantize to Key", self._quantize,
                           "Quantize detected notes to the song's detected musical key",
                           reset_callback=lambda: self._quantize.setChecked(
                               not _DEFAULTS["disable_quantization"]))

        # Vocal center correction
        self._vocal_center = ToggleSwitch(
            checked=not self._config.get("disable_vocal_center", False)
        )
        card.add_toggle_row("Vocal Center Correction", self._vocal_center,
                           "Safety-net for consistently wrong octave detection",
                           reset_callback=lambda: self._vocal_center.setChecked(
                               not _DEFAULTS["disable_vocal_center"]))

        # Onset correction
        self._onset_correction = ToggleSwitch(
            checked=not self._config.get("disable_onset_correction", False)
        )
        card.add_toggle_row("Onset Timing Correction", self._onset_correction,
                           "Snap note starts to detected audio onsets for better timing",
                           reset_callback=lambda: self._onset_correction.setChecked(
                               not _DEFAULTS["disable_onset_correction"]))

        # Vocal separation
        self._separation = ToggleSwitch(
            checked=not self._config.get("disable_separation", False)
        )
        card.add_toggle_row("Vocal Separation", self._separation,
                           "Separate vocals from instrumentals using Demucs",
                           reset_callback=lambda: self._separation.setChecked(
                               not _DEFAULTS["disable_separation"]))

        # Denoise
        self._denoise = ToggleSwitch(
            checked=not self._config.get("disable_denoise_track_noise", False)
        )
        card.add_toggle_row("Noise Reduction", self._denoise,
                           "Adaptive noise floor tracking",
                           reset_callback=lambda: self._denoise.setChecked(
                               not _DEFAULTS["disable_denoise_track_noise"]))

        card.add_separator()

        # Denoise parameters
        self._denoise_nr = _NoScrollDoubleSpinBox()
        self._denoise_nr.setRange(0.01, 97.0)
        self._denoise_nr.setValue(self._config.get("denoise_nr", 20))
        self._denoise_nr.setSuffix(" dB")
        card.add_row("Noise Reduction Level", self._denoise_nr,
                     reset_callback=lambda: self._denoise_nr.setValue(
                         _DEFAULTS["denoise_nr"]))

        self._denoise_nf = _NoScrollSpinBox()
        self._denoise_nf.setRange(-80, -20)
        self._denoise_nf.setValue(self._config.get("denoise_nf", -80))
        self._denoise_nf.setSuffix(" dB")
        card.add_row("Noise Floor", self._denoise_nf,
                     reset_callback=lambda: self._denoise_nf.setValue(
                         _DEFAULTS["denoise_nf"]))

        card.add_separator()

        # Experimental features
        exp_label = QLabel("Experimental Features")
        exp_label.setObjectName("subsectionHeader")
        card.add_widget(exp_label)

        self._syllable_split = ToggleSwitch(
            checked=self._config.get("syllable_split", False)
        )
        card.add_toggle_row("Syllable-Level Splitting", self._syllable_split,
                           "Preserve syllable-level note splits (prevents same-pitch merge)",
                           reset_callback=lambda: self._syllable_split.setChecked(
                               _DEFAULTS["syllable_split"]))

        self._vocal_gap_fill = ToggleSwitch(
            checked=self._config.get("vocal_gap_fill", False)
        )
        card.add_toggle_row("Vocal Gap Fill", self._vocal_gap_fill,
                           "Fill un-transcribed vocal segments with placeholder notes",
                           reset_callback=lambda: self._vocal_gap_fill.setChecked(
                               _DEFAULTS["vocal_gap_fill"]))

        self._keep_numbers = ToggleSwitch(
            checked=self._config.get("keep_numbers", False)
        )
        card.add_toggle_row("Keep Numbers as Digits", self._keep_numbers,
                           "Transcribe numbers as 1, 2, 3 instead of one, two, three",
                           reset_callback=lambda: self._keep_numbers.setChecked(
                               _DEFAULTS["keep_numbers"]))

        card.add_separator()

        # LLM Correction
        llm_label = QLabel("LLM Lyric Correction")
        llm_label.setObjectName("subsectionHeader")
        card.add_widget(llm_label)

        self._llm_correct = ToggleSwitch(
            checked=self._config.get("llm_correct", False)
        )
        card.add_toggle_row("Enable LLM Correction", self._llm_correct,
                           reset_callback=lambda: self._llm_correct.setChecked(
                               _DEFAULTS["llm_correct"]))

        self._llm_model = _NoScrollComboBox()
        self._llm_model.setEditable(True)
        self._llm_model.addItems([
            "qwen/qwen3-32b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-3.3-70b-versatile",
            "gpt-4o-mini",
        ])
        self._llm_model.setCurrentText(
            self._config.get("llm_model", "qwen/qwen3-32b")
        )
        card.add_row("Model", self._llm_model,
                     reset_callback=lambda: self._llm_model.setCurrentText(
                         _DEFAULTS["llm_model"]))
        self._llm_model.setEnabled(self._llm_correct.isChecked())

        card.add_info(
            "API URL and API key are configured in Preferences. "
            "Recommended: Groq with qwen/qwen3-32b (free plan)."
        )

        # Wire LLM toggle to enable/disable model selector
        self._llm_correct.toggled_signal.connect(
            lambda on: self._llm_model.setEnabled(on)
        )

        card.add_separator()

        # Score calculation
        score_label = QLabel("Scoring")
        score_label.setObjectName("subsectionHeader")
        card.add_widget(score_label)

        self._calculate_score = ToggleSwitch(
            checked=self._config.get("calculate_score", False)
        )
        card.add_toggle_row("Calculate Score", self._calculate_score,
                           "Score the generated UltraStar file against the audio",
                           reset_callback=lambda: self._calculate_score.setChecked(
                               _DEFAULTS["calculate_score"]))
        card.add_info(
            "Requires the optional ultrastar-score package. "
            "Install with: uv sync --extra scoring"
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
                     reset_callback=lambda: self._format_version.setCurrentText(
                         _DEFAULTS["format_version"]))

        self._create_plot = ToggleSwitch(
            checked=self._config.get("create_plot", False)
        )
        card.add_toggle_row("Generate Plots", self._create_plot,
                           reset_callback=lambda: self._create_plot.setChecked(
                               _DEFAULTS["create_plot"]))

        self._create_midi = ToggleSwitch(
            checked=self._config.get("create_midi", True)
        )
        card.add_toggle_row("Generate MIDI", self._create_midi,
                           reset_callback=lambda: self._create_midi.setChecked(
                               _DEFAULTS["create_midi"]))

        self._create_chunks = ToggleSwitch(
            checked=self._config.get("create_audio_chunks", False)
        )
        card.add_toggle_row("Create Audio Chunks", self._create_chunks,
                           reset_callback=lambda: self._create_chunks.setChecked(
                               _DEFAULTS["create_audio_chunks"]))

        self._create_karaoke = ToggleSwitch(
            checked=self._config.get("create_karaoke", True)
        )
        card.add_toggle_row("Create Karaoke File", self._create_karaoke,
                           reset_callback=lambda: self._create_karaoke.setChecked(
                               _DEFAULTS["create_karaoke"]))

        self._keep_cache = ToggleSwitch(
            checked=self._config.get("keep_cache", False)
        )
        card.add_toggle_row("Keep Cache", self._keep_cache,
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
                           "Disable CUDA GPU acceleration entirely",
                           reset_callback=lambda: self._force_cpu.setChecked(
                               _DEFAULTS["force_cpu"]))

        self._force_whisper_cpu = ToggleSwitch(
            checked=self._config.get("force_whisper_cpu", False)
        )
        card.add_toggle_row("Force Whisper CPU", self._force_whisper_cpu,
                           "Run only Whisper on CPU (keep other stages on GPU)",
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
                     "Manual BPM value (leave empty for auto-detection)",
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
                     "Manual octave shift (e.g., 1 or -1)",
                     reset_callback=lambda: self._octave_shift.setText(
                         str(_DEFAULTS["octave_shift"])))

        self._main_layout.addWidget(card)

    # ─── Paths ────────────────────────────────────────────────────────────

    def _build_paths_section(self):
        card = SettingsCard("Paths")

        # MuseScore
        ms_row = QHBoxLayout()
        self._musescore_path = QLineEdit()
        self._musescore_path.setPlaceholderText("Optional: Path to MuseScore")
        self._musescore_path.setText(self._config.get("musescore_path", ""))
        ms_row.addWidget(self._musescore_path, 1)
        ms_browse = QPushButton("Browse")
        ms_browse.clicked.connect(
            lambda: self._browse_file(self._musescore_path, "MuseScore Executable (*)")
        )
        ms_row.addWidget(ms_browse)
        card.add_row("MuseScore Path", QWidget())
        # Replace the empty widget with the row
        card.remove_last_item()
        card.add_layout(ms_row)

        # FFmpeg
        ff_row = QHBoxLayout()
        self._ffmpeg_path = QLineEdit()
        self._ffmpeg_path.setPlaceholderText("Optional: Custom FFmpeg path")
        self._ffmpeg_path.setText(self._config.get("ffmpeg_path", ""))
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

    # ─── Public API ───────────────────────────────────────────────────────

    def collect_config(self) -> dict:
        """Collect all conversion settings into a config dictionary.

        Note: output_folder, llm_api_base_url, llm_api_key and cookie_file
        are managed exclusively by the Preferences tab to avoid conflicts.
        """
        return {
            "whisper_model": self._whisper_model.currentText(),
            "whisper_batch_size": self._whisper_batch_size.value(),
            "whisper_compute_type": self._whisper_compute.currentText(),
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
            "llm_model": self._llm_model.currentText(),
            "calculate_score": self._calculate_score.isChecked(),
            "format_version": self._format_version.currentText(),
            "create_plot": self._create_plot.isChecked(),
            "create_midi": self._create_midi.isChecked(),
            "create_audio_chunks": self._create_chunks.isChecked(),
            "create_karaoke": self._create_karaoke.isChecked(),
            "keep_cache": self._keep_cache.isChecked(),
            "force_cpu": self._force_cpu.isChecked(),
            "force_whisper_cpu": self._force_whisper_cpu.isChecked(),
            "bpm_override": self._bpm_override.text(),
            "octave_shift": self._octave_shift.text(),
            "musescore_path": self._musescore_path.text(),
            "ffmpeg_path": self._ffmpeg_path.text(),
        }
