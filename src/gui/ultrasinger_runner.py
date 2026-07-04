"""QThread-based runner for UltraSinger CLI subprocess."""

import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"


def _find_project_root() -> Path:
    """Find the UltraSinger project root (contains pyproject.toml)."""
    # Start from this file's location and walk up
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def _build_command(project_root: Path) -> list[str]:
    """Determine the best way to invoke UltraSinger."""
    entry = project_root / "src" / "UltraSinger.py"
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "python", str(entry)]
    return [sys.executable, str(entry)]


class ConversionWorker(QObject):
    """Worker that runs UltraSinger in a subprocess and emits log lines."""

    line_output = Signal(str)
    finished = Signal(int)  # exit code
    stage_changed = Signal(str)  # high-level stage description

    def __init__(self, args: list[str], parent=None):
        super().__init__(parent)
        self._args = args
        self._process: subprocess.Popen | None = None
        self._cancelled = False
        self._terminated_by_cancel = False

    def run(self):
        """Execute UltraSinger and stream output."""
        if self._cancelled:
            self.finished.emit(-2)
            return

        project_root = _find_project_root()
        cmd = _build_command(project_root) + self._args

        # Redact API key values from logged command
        display_cmd = list(cmd)
        for i, token in enumerate(display_cmd):
            if token in ("--llm_api_key", "--remote_stt_api_key") and i + 1 < len(display_cmd):
                display_cmd[i + 1] = "***"
        self.line_output.emit(f"[GUI] Running: {' '.join(display_cmd)}")
        self.line_output.emit("")

        try:
            # On Windows, create a new process group so we can kill the
            # entire tree (Python + yt-dlp + ffmpeg children) on cancel.
            # On Unix, use a new session via os.setsid.
            kwargs: dict = {}
            if _IS_WINDOWS:
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True

            # Force unbuffered stdout so every print() in the child
            # process is immediately visible in the GUI — without this,
            # Python block-buffers piped stdout and the user sees nothing
            # until the buffer fills (e.g. silence after demucs 100%).
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(project_root),
                encoding="utf-8",
                errors="replace",
                env=env,
                **kwargs,
            )

            # Re-check cancellation after process started
            if self._cancelled:
                self._process.terminate()
                self._terminated_by_cancel = True
                self.line_output.emit("[GUI] Conversion cancelled by user.")
                self.finished.emit(-2)
                return

            for line in self._process.stdout:
                line = line.rstrip("\n\r")
                self.line_output.emit(line)
                self._detect_stage(line)

            self._process.wait()
            exit_code = self._process.returncode

        except FileNotFoundError as e:
            self.line_output.emit(f"[Error] Command not found: {e}")
            exit_code = -1
        except Exception as e:
            self.line_output.emit(f"[Error] {e}")
            exit_code = -1

        if self._terminated_by_cancel:
            self.line_output.emit("[GUI] Conversion cancelled by user.")
            exit_code = -2

        self.finished.emit(exit_code)

    def cancel(self):
        """Terminate the subprocess and all its children (non-blocking)."""
        self._cancelled = True
        self._terminated_by_cancel = True
        if self._process and self._process.poll() is None:
            self._kill_tree(self._process)
            threading.Thread(
                target=self._wait_and_kill, daemon=True
            ).start()

    @staticmethod
    def _kill_tree(proc: subprocess.Popen):
        """Kill the process and its entire child tree.

        On Windows, ``taskkill /F /T`` kills the tree.
        On Unix, ``os.killpg`` sends SIGTERM to the process group.
        """
        pid = proc.pid
        try:
            if _IS_WINDOWS:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=10,
                )
            else:
                os.killpg(os.getpgid(pid), 15)  # SIGTERM
        except (OSError, subprocess.SubprocessError):
            # Fallback: just terminate the main process
            try:
                proc.terminate()
            except OSError:
                pass

    def _wait_and_kill(self):
        """Wait for process to terminate; force-kill if it doesn't."""
        proc = self._process
        if not proc:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass

    _STAGE_KEYWORDS = [
        ("Separating vocals", "Separating Vocals..."),
        ("Transcribing", "Transcribing Audio..."),
        ("Pitch detection", "Detecting Pitch..."),
        ("BPM", "Detecting BPM..."),
        ("Creating UltraStar", "Creating UltraStar File..."),
        ("Onset correction", "Onset Correction..."),
        ("LLM", "LLM Lyric Correction..."),
        ("Score:", "Scoring..."),
        ("Done", "Completed!"),
    ]

    def _detect_stage(self, line: str):
        for keyword, stage_name in self._STAGE_KEYWORDS:
            if keyword.lower() in line.lower():
                self.stage_changed.emit(stage_name)
                return


class UltraSingerRunner(QObject):
    """Manages UltraSinger conversion runs in background threads."""

    line_output = Signal(str)
    finished = Signal(int)
    stage_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: ConversionWorker | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, args: list[str]):
        """Start a conversion with the given CLI arguments."""
        if self.is_running:
            logger.warning("A conversion is already running")
            return

        self._thread = QThread()
        self._worker = ConversionWorker(args)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.line_output.connect(self.line_output.emit)
        self._worker.stage_changed.connect(self.stage_changed.emit)
        self._worker.finished.connect(self._on_finished)

        self._thread.start()

    def cancel(self):
        """Cancel the current conversion."""
        if self._worker:
            self._worker.cancel()

    def _on_finished(self, exit_code: int):
        # Clean up the thread BEFORE emitting finished, so that
        # is_running returns False when the QueueManager calls start()
        # for the next item in the slot connected to finished.
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
        self._worker = None
        self.finished.emit(exit_code)

    def build_args(self, config: dict, input_source: str) -> list[str]:
        """Build CLI argument list from configuration dictionary."""
        args = ["-i", input_source]

        if config.get("output_folder"):
            args.extend(["-o", config["output_folder"]])

        # Cookie file
        if config.get("cookie_file") and Path(config["cookie_file"]).exists():
            args.extend(["--cookiefile", config["cookie_file"]])

        # Whisper settings
        if config.get("whisper_model"):
            args.extend(["--whisper", config["whisper_model"]])
        if config.get("whisper_batch_size"):
            args.extend(["--whisper_batch_size", str(config["whisper_batch_size"])])
        if config.get("whisper_compute_type"):
            args.extend(["--whisper_compute_type", config["whisper_compute_type"]])
        if config.get("whisper_align_model"):
            args.extend(["--whisper_align_model", config["whisper_align_model"]])

        # VAD / ASR thresholds
        vad_onset = config.get("vad_onset", 0.35)
        if vad_onset != 0.35:
            args.extend(["--vad_onset", str(vad_onset)])
        vad_offset = config.get("vad_offset", 0.20)
        if vad_offset != 0.20:
            args.extend(["--vad_offset", str(vad_offset)])
        no_speech = config.get("no_speech_threshold", 0.4)
        if no_speech != 0.4:
            args.extend(["--no_speech_threshold", str(no_speech)])

        # Language
        if config.get("language_mode") == "manual" and config.get("language"):
            args.extend(["--language", config["language"]])

        # Format version
        if config.get("format_version"):
            args.extend(["--format_version", config["format_version"]])

        # Vocal separation backend — only forward backend-relevant model arg
        separator = config.get("separator_backend", "audio_separator")
        if separator:
            args.extend(["--separator", separator])
        if separator == "audio_separator" and config.get("audio_separator_model"):
            args.extend(["--audio_separator_model", config["audio_separator_model"]])
        elif separator == "demucs" and config.get("demucs_model"):
            args.extend(["--demucs", config["demucs_model"]])

        # BPM override
        bpm = config.get("bpm_override", "")
        if bpm:
            args.extend(["--bpm", str(bpm)])

        # Octave shift
        octave = config.get("octave_shift", "")
        if octave:
            args.extend(["--octave", str(octave)])

        # Denoise
        if config.get("denoise_nr") is not None:
            args.extend(["--denoise_nr", str(config["denoise_nr"])])
        if config.get("denoise_nf") is not None:
            args.extend(["--denoise_nf", str(config["denoise_nf"])])

        # Boolean flags (disabled by default in CLI)
        if config.get("disable_separation"):
            args.append("--disable_separation")
        if config.get("disable_quantization"):
            args.append("--disable_quantization")
        if config.get("disable_vocal_center"):
            args.append("--disable_vocal_center")
        if config.get("disable_lyrics_lookup"):
            args.append("--disable_lyrics_lookup")
        if config.get("disable_reference_lyrics"):
            args.append("--disable_reference_lyrics")
        if config.get("detect_growl"):
            args.append("--detect_freestyle")
            harmonicity = config.get("freestyle_harmonicity", 0.40)
            if harmonicity != 0.40:
                args.extend(["--freestyle_harmonicity", str(harmonicity)])
            energy = config.get("freestyle_energy", 0.01)
            if energy != 0.01:
                args.extend(["--freestyle_energy", str(energy)])
            confidence = config.get("freestyle_confidence", 0.35)
            if confidence != 0.35:
                args.extend(["--freestyle_confidence", str(confidence)])
            pitch_stdev = config.get("freestyle_pitch_stdev", 4.0)
            if pitch_stdev != 4.0:
                args.extend(["--freestyle_pitch_stdev", str(pitch_stdev)])
            spectral_flatness = config.get("freestyle_spectral_flatness", 0.25)
            if spectral_flatness != 0.25:
                args.extend(["--freestyle_spectral_flatness", str(spectral_flatness)])
            if not config.get("freestyle_use_spectral", True):
                args.append("--no_freestyle_spectral")
        if config.get("disable_onset_correction"):
            args.append("--disable_onset_correction")
        if config.get("disable_denoise_track_noise"):
            args.append("--disable_denoise_track_noise")
        if not config.get("hyphenation", True):
            args.append("--disable_hyphenation")
        if not config.get("create_karaoke", True):
            args.append("--disable_karaoke")
        if not config.get("create_midi", True):
            args.append("--disable_midi")
        if not config.get("write_metadata_tags", True):
            args.append("--no_metadata_tags")

        # Boolean flags (enabled by CLI flag)
        if config.get("keep_cache"):
            args.append("--keep_cache")
        if config.get("create_plot"):
            args.append("--plot")
        if config.get("create_audio_chunks"):
            args.append("--create_audio_chunks")
        if config.get("keep_numbers"):
            args.append("--keep_numbers")
        if config.get("keep_audio_in_video"):
            args.append("--keep_audio_in_video")
        if config.get("write_settings_info"):
            args.append("--write_settings_info")
        if config.get("force_cpu"):
            args.append("--force_cpu")
        if config.get("force_whisper_cpu"):
            args.append("--force_whisper_cpu")

        # Pitcher backend
        # Always pass the pitcher explicitly so the GUI selection wins
        # regardless of the CLI default
        pitcher = config.get("pitcher", "fcpe")
        if pitcher:
            args.extend(["--pitcher", pitcher])

        # Experimental features
        if config.get("syllable_split"):
            args.append("--syllable_split")
        if config.get("vocal_gap_fill"):
            args.append("--vocal_gap_fill")
        if config.get("pitch_change_split", True):
            args.append("--pitch_change_split")
        else:
            args.append("--no_pitch_change_split")
        if config.get("pitch_notes"):
            args.append("--pitch_notes")
        if config.get("golden_notes"):
            args.append("--golden_notes")

        # LLM correction
        if config.get("llm_correct"):
            args.append("--llm_correct")

            # Resolve LLM provider: look up by provider ID, fall back to
            # flat config keys for backward compatibility.
            provider_id = config.get("llm_provider_id", "")
            providers = config.get("llm_providers", [])
            provider = None
            if provider_id and providers:
                provider = next(
                    (p for p in providers
                     if (p.get("id") if isinstance(p, dict) else getattr(p, "id", "")) == provider_id),
                    None,
                )

            if provider:
                p = provider if isinstance(provider, dict) else provider.__dict__
                url = p.get("api_base_url", "")
                model = p.get("default_model", "")
                api_key = config.get(f"llm_api_key_{provider_id}", "")
                if url:
                    args.extend(["--llm_api_base_url", url])
                if api_key:
                    args.extend(["--llm_api_key", api_key])
                if model:
                    args.extend(["--llm_model", model])
            else:
                # Flat keys (legacy / no provider selected)
                if config.get("llm_api_base_url"):
                    args.extend(["--llm_api_base_url", config["llm_api_base_url"]])
                if config.get("llm_api_key"):
                    args.extend(["--llm_api_key", config["llm_api_key"]])
                if config.get("llm_model"):
                    args.extend(["--llm_model", config["llm_model"]])

            # Retry settings
            if not config.get("llm_retry_on_rate_limit", True):
                args.append("--llm_no_retry")
            else:
                retry_wait = config.get("llm_retry_wait", 60)
                if retry_wait != 60:
                    args.extend(["--llm_retry_wait", str(retry_wait)])
                retry_max = config.get("llm_retry_max", 3)
                if retry_max != 3:
                    args.extend(["--llm_retry_max", str(retry_max)])

        # Remote speech-to-text (text-only Whisper alternative)
        if config.get("remote_stt"):
            args.append("--remote_stt")
            if config.get("remote_stt_api_base_url"):
                args.extend(["--remote_stt_api_base_url", config["remote_stt_api_base_url"]])
            if config.get("remote_stt_api_key"):
                args.extend(["--remote_stt_api_key", config["remote_stt_api_key"]])
            if config.get("remote_stt_model"):
                args.extend(["--remote_stt_model", config["remote_stt_model"]])
            timeout = config.get("remote_stt_timeout", 120)
            if timeout != 120:
                args.extend(["--remote_stt_timeout", str(timeout)])

        # Refinement
        if config.get("refine_from_vocal", False):
            args.append("--refine_from_vocal")
            if not config.get("refine_pitch", True):
                args.append("--disable_refine_pitch")
            if not config.get("refine_timing", True):
                args.append("--disable_refine_timing")
            hit_ratio = config.get("refine_hit_ratio", 0.4)
            if hit_ratio != 0.4:
                args.extend(["--refine_hit_ratio", str(hit_ratio)])
            timing_thr = config.get("refine_timing_threshold", 30.0)
            if timing_thr != 30.0:
                args.extend(["--refine_timing_threshold", str(timing_thr)])

        # ptAKF chart refit (enabled by default; send disables when off)
        if config.get("ptakf_refit", True):
            refit_min_ms = config.get("ptakf_refit_min_note_ms", 100.0)
            if refit_min_ms != 100.0:
                args.extend(["--ptakf_refit_min_note_ms", str(refit_min_ms)])
            if config.get("ptakf_refit_fill", True):
                fill_min_ms = config.get("ptakf_refit_fill_min_ms", 300.0)
                if fill_min_ms != 300.0:
                    args.extend(["--ptakf_refit_fill_min_ms", str(fill_min_ms)])
            else:
                args.append("--disable_ptakf_refit_fill")
        else:
            args.append("--disable_ptakf_refit")

        # Scoring (enabled by default; the toggle sends the disable)
        if not config.get("calculate_score", True):
            args.append("--disable_score")

        # Video platform metadata URL (when input is pre-downloaded audio)
        if config.get("video_url"):
            args.extend(["--video_url", config["video_url"]])

        # Browser-captured GVS PO token (full yt-dlp format list)
        if config.get("yt_po_token"):
            args.extend(["--yt_po_token", config["yt_po_token"]])

        # Paths
        if config.get("musescore_path"):
            args.extend(["--musescore_path", config["musescore_path"]])
        if config.get("ffmpeg_path"):
            args.extend(["--ffmpeg", config["ffmpeg_path"]])

        return args
