"""Common Prints"""

from modules.console_colors import ULTRASINGER_HEAD, gold_highlighted, light_blue_highlighted


def print_help() -> None:
    """Print help text"""
    help_string = """
    UltraSinger.py [opt] [mode] [transcription] [post-processing] [output]
    
    [opt]
    -h      This help text.
    -i      Ultrastar.txt
            audio/video like .mp3, .mp4, .wav, youtube link
    -o      Output folder
    
    [mode]
    ## INPUT is audio ##
    default (Full Automatic Mode) - Creates all, depending on command line options
    --interactive - Interactive Mode. All options are asked at runtime for easier configuration
    
    # Single file creation selection is in progress, you currently getting all!
    (-u      Create ultrastar txt file) # In Progress
    (-m      Create midi file) # In Progress
    (-s      Create sheet file) # In Progress
    
    ## INPUT is ultrastar.txt ##
    default  Creates all


    [metadata]
    --youtube_url           YouTube URL for metadata lookup (artist, title) when the input (-i)
                            is a local audio/video file instead of a URL. This is used by the
                            GUI's browser-based download: the audio is pre-downloaded via the
                            embedded browser (bypassing bot detection), so the input is a local
                            file, but the YouTube URL is still needed for artist/title/thumbnail.
                            Not needed when -i is already a YouTube URL.

    [separation]
    # Default is htdemucs
    --demucs              Model name htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q >> ((default) is htdemucs)

    [transcription]
    # Default is whisper
    --whisper               Multilingual model > tiny|base|small|medium|large-v1|large-v2|large-v3  >> ((default) is large-v2)
                            English-only model > tiny.en|base.en|small.en|medium.en
    --whisper_align_model   Use other languages model for Whisper provided from huggingface.co
    --language              Override the language detected by whisper for alignment and hyphenation.
                            Default: auto-detect. WARNING: setting this for non-matching songs
                            will degrade alignment quality (e.g. --language en for German songs).
    --whisper_batch_size    Reduce if low on GPU mem >> ((default) is 16)
    --whisper_compute_type  Change to "int8" if low on GPU mem (may reduce accuracy) >> ((default) is "float16" for cuda devices, "int8" for cpu)
    --keep_numbers          Numbers will be transcribed as numerics instead of as words >> True|False >> ((default) is False)
    --vad_onset             VAD speech activation threshold (0.0-1.0). Lower values capture more vocal
                            segments including soft/breathy singing. >> ((default) is 0.35, WhisperX default: 0.5)
    --vad_offset            VAD speech deactivation threshold (0.0-1.0). Lower values keep segments active
                            longer during vocal dips. >> ((default) is 0.20, WhisperX default: 0.363)
    --no_speech_threshold   No-speech probability threshold (0.0-1.0). Lower values prevent Whisper from
                            classifying singing as silence. >> ((default) is 0.4, WhisperX default: 0.6)

    [post-processing]
    --bpm                   Override auto-detected BPM with a manual value (e.g., --bpm 340)
    --octave                Shift all notes by N octaves after pitch detection (e.g., --octave 1 for up, --octave -1 for down)
    --disable_hyphenation   Disable word hyphenation. Hyphenation is enabled by default.
    --disable_separation    Disable track separation. Track separation is enabled by default.
    --disable_karaoke       Disable creation of karaoke style txt file. Karaoke is enabled by default.
    --disable_onset_correction  Disable onset-based timing correction. Enabled by default.
    --disable_quantization  Disable key quantization. Key quantization is enabled by default and removes slides and out-of-key notes.
    --disable_vocal_center  Disable vocal-centre octave correction. Enabled by default.
    --syllable_split        Preserve syllable-level note splits at pitch changes (experimental). Disabled by default.
    --vocal_gap_fill        Fill un-transcribed vocal gaps with placeholder notes (experimental). Disabled by default.
    --pitch_change_split    Split notes at pitch change boundaries within a syllable (experimental). Disabled by default.
    --pitch_notes           Generate notes from pitch contour instead of word timing (experimental). Disabled by
                            default. Best for melismatic songs with runs, slides and ornaments where word-level
                            timing produces flat, unusable notes. Whisper lyrics are overlaid by time alignment.
    --disable_lyrics_lookup Disable LRCLIB lyrics lookup and correction. Lyrics lookup is enabled by default
                            and fetches verified reference lyrics to correct Whisper transcription errors.

    [refinement]
    --disable_refine            Disable the reverse-scoring refinement pass. Refinement is enabled by default
                                and uses the game's C++ ptAKF pitch detector to find and fix poorly-scoring notes.
    --refine_from_vocal         (legacy) Explicitly enable refinement (now the default)
    --disable_refine_pitch      Disable pitch refinement (enabled by default when refine is on)
    --disable_refine_timing     Disable timing refinement (enabled by default when refine is on)
    --refine_hit_ratio          Notes below this hit ratio are pitch-corrected (0.0-1.0) >> ((default) is 0.4)
    --refine_timing_threshold   Milliseconds threshold before correcting timing >> ((default) is 30)


    [llm lyric correction]
    --llm_correct           Enable LLM-based lyric correction (requires API key)
    --llm_api_base_url      OpenAI-compatible API base URL >> ((default) is https://api.openai.com/v1)
    --llm_api_key           API key for LLM service (or set LLM_API_KEY env var)
    --llm_model             LLM model name >> ((default) is gpt-4o-mini)
    --llm_no_retry          Disable automatic retry on rate limit (HTTP 429). Retry is enabled by default.
    --llm_retry_wait        Seconds to wait between retries >> ((default) is 60)
    --llm_retry_max         Maximum retries per text chunk >> ((default) is 3)

    [output]
    --format_version        0.3.0|1.0.0|1.1.0|1.2.0 >> ((default) is 1.2.0)
    --create_audio_chunks   Enable creation of audio chunks. Audio chunks are disabled by default.
    --keep_cache            Keep cache folder after creation. Cache folder is removed by default.
    --keep_audio_in_video   Keep full audio (vocals+instrumental) in the output video. Disabled by default.
    --write_settings_info   Write ultrasinger_parameter.info with all settings and scores to output dir. Disabled by default.
    --plot                  Enable creation of plots. Plots are disabled by default.

    [denoise]
    --denoise_nr            Noise reduction in dB (0.01-97). Lower preserves more vocal detail. >> ((default) is 20)
    --denoise_nf            Noise floor in dB (-80 to -20) >> ((default) is -80)
    --disable_denoise_track_noise  Disable adaptive noise floor tracking >> ((default) tracking is enabled)

    [yt-dlp]
    --cookiefile            File name where cookies should be read from and dumped to.

    [paths]
    --musescore_path        Path to MuseScore executable
    --ffmpeg                Path to ffmpeg and ffprobe executable

    [device]
    --force_cpu             Force all steps to be processed on CPU.
    --force_whisper_cpu     Force whisper transcription to be processed on CPU.
    """
    print(help_string)


def print_support() -> None:
    """Print support text"""
    print()
    print(
        f"{ULTRASINGER_HEAD} {gold_highlighted('Do you like UltraSinger? Want it to be even better? Then help with your')} {light_blue_highlighted('support')}{gold_highlighted('!')}"
    )
    print(
        f"{ULTRASINGER_HEAD} See project page -> https://github.com/rakuri255/UltraSinger"
    )
    print(
        f"{ULTRASINGER_HEAD} {gold_highlighted('This will help a lot to keep this project alive and improved.')}"
    )


def print_version(app_version: str) -> None:
    """Print version text"""
    versiontext = (f"UltraSinger Version: {app_version}")    
    starquant = "*" * len(versiontext) # set star number to length of version
    print()
    print(f"{ULTRASINGER_HEAD} {gold_highlighted(starquant)}")
    print(f"{ULTRASINGER_HEAD} {gold_highlighted('UltraSinger Version:')} {light_blue_highlighted(app_version)}")
    print(f"{ULTRASINGER_HEAD} {gold_highlighted(starquant)}")
