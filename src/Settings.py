from dataclasses import dataclass

from dataclasses_json import dataclass_json

from modules.Audio.separation import (
    AudioSeparatorModel,
    DemucsModel,
    SeparatorBackend,
)
from modules.Speech_Recognition.Whisper import WhisperModel
from modules.Ultrastar.ultrastar_txt import FormatVersion


@dataclass_json
@dataclass
class Settings:

    APP_VERSION = "0.0.13.dev16"
    CONFIDENCE_THRESHOLD = 0.6
    CONFIDENCE_PROMPT_TIMEOUT = 4

    create_midi = True
    create_plot = False
    create_audio_chunks = False
    hyphenation = True
    use_separated_vocal = True
    create_karaoke = True
    ignore_audio = False
    input_file_is_ultrastar_txt = False # todo: to process_data
    keep_cache = False
    interactive_mode = False
    user_ffmpeg_path = ""
    quantize_to_key = True  # Quantize notes to the detected key of the song
    bpm_override = None  # Manual BPM override (float), skips auto-detection when set
    octave_shift = None  # Manual octave shift (int), shifts all notes by N octaves after detection
    vocal_center_correction = True  # Safety-net octave correction for consistently wrong-octave detection
    onset_correction = True  # Snap note start times to detected audio onsets
    syllable_split = False  # Preserve syllable-level note splits at pitch changes
    vocal_gap_fill = False  # Fill un-transcribed vocal gaps with placeholder notes
    pitch_change_split = False  # Split notes at pitch change boundaries (melismas, runs)
    pitch_notes = False  # Generate notes from pitch contour instead of word timing
    lyrics_lookup = True  # Look up reference lyrics from LRCLIB and correct Whisper transcription
    disable_reference_lyrics = False  # Disable reference-lyrics-first pipeline (forced alignment with LRCLIB synced lyrics)
    keep_audio_in_video = False  # Keep full audio (vocals+instrumental) embedded in the output video
    write_settings_info = False  # Write ultrasinger_parameter.info with settings + score to output dir
    write_metadata_tags = True  # Write ID3/Vorbis metadata tags to output audio files

    # Refinement (reverse-scoring polish via ultrastar-score C++ ptAKF)
    refine_from_vocal = True  # Reverse-scoring refinement pass (enabled by default)
    refine_pitch = True  # Correct note pitches from vocal audio (when refine is on)
    refine_timing = True  # Correct note timing from vocal audio (when refine is on)
    refine_hit_ratio: float = 0.4  # Notes below this hit ratio are pitch-corrected (0.0-1.0)
    refine_timing_threshold: float = 30.0  # Milliseconds deviation before correcting

    # Process data Paths
    input_file_path = ""
    output_folder_path = ""
    
    language = None
    format_version = FormatVersion.V1_2_0

    # Vocal separation
    separator_backend = SeparatorBackend.AUDIO_SEPARATOR  # audio_separator (default, deterministic) | demucs
    demucs_model = DemucsModel.HTDEMUCS  # htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q|SIG
    audio_separator_model = AudioSeparatorModel.BS_ROFORMER  # BS-Roformer SDR 12.97 (default)

    # Whisper
    transcriber = "whisper"  # whisper
    whisper_model = WhisperModel.LARGE_V2  # Multilingual model tiny|base|small|medium|large-v1|large-v2|large-v3
    # English-only model tiny.en|base.en|small.en|medium.en
    whisper_align_model = None   # Model for other languages from huggingface.co e.g -> "gigant/romanian-wav2vec2"
    whisper_batch_size = 16   # reduce if low on GPU mem
    whisper_compute_type = None   # change to "int8" if low on GPU mem (may reduce accuracy)
    keep_numbers = False

    # VAD (Voice Activity Detection) — tuned for singing
    vad_onset: float = 0.35   # Speech activation threshold (WhisperX default: 0.5, lowered for singing)
    vad_offset: float = 0.20  # Speech deactivation threshold (WhisperX default: 0.363, lowered for singing)
    # ASR (Automatic Speech Recognition) — tuned for singing
    no_speech_threshold: float = 0.4  # No-speech probability threshold (WhisperX default: 0.6, lowered for singing)

    # Device
    pytorch_device = 'cpu'  # cpu|cuda
    force_cpu = False
    force_whisper_cpu = False

    # MuseScore
    musescore_path = None

    # yt-dlp
    cookiefile = None
    youtube_url: str | None = None  # For metadata when audio is pre-downloaded

    # LLM lyric correction
    llm_correct_lyrics = False  # Enable LLM-based lyric post-correction
    llm_api_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None  # API key (or set LLM_API_KEY env var)
    llm_model: str = "gpt-4o-mini"  # Default to cheap, fast model
    llm_retry_on_rate_limit = True  # Retry on HTTP 429 rate limit errors
    llm_retry_wait: int = 60  # Seconds to wait between retries
    llm_retry_max: int = 3  # Maximum retries per chunk

    # Growl/Scream detection — mark unpitchable passages as freestyle
    detect_growl = False  # Enable growl/scream/spoken detection (marks notes as freestyle)
    growl_confidence_threshold: float = 0.35  # SwiftF0 median confidence below this → suspect
    growl_pitch_stdev_threshold: float = 4.0  # Pitch stdev (semitones) above this → suspect
    growl_spectral_flatness_threshold: float = 0.25  # Spectral flatness above this → noisy
    growl_use_spectral: bool = True  # Enable Tier 2 spectral flatness analysis

    # Denoise
    denoise_noise_reduction = 20  # Noise reduction in dB (0.01-97, default: 20). Previous default was 70 which destroyed vocal nuances needed by Whisper.
    denoise_noise_floor = -80  # Noise floor in dB (-80 to -20, default: -80)
    denoise_track_noise = True  # Enable adaptive noise floor tracking

    # UltraSinger Evaluation Configuration
    test_songs_input_folder = None
    cache_override_path = None #"C:\\UltraSinger\\test_output"
    skip_cache_vocal_separation = False
    skip_cache_denoise_vocal_audio = False
    skip_cache_transcription = False
    skip_cache_pitch_detection = False
    calculate_score = True