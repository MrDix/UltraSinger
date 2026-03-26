"""UltraSinger uses AI to automatically create UltraStar song files"""

import copy
import getopt
import os
import sys
import warnings

# Suppress noisy third-party warnings that fire at import time.
# Must run before any transitive imports of requests, torchaudio, pyannote, etc.
warnings.filterwarnings("ignore", module="requests")
warnings.filterwarnings("ignore", module="pyannote")
warnings.filterwarnings("ignore", message="In 2\\.9.*torchaudio\\.save_with_torchcodec")

# Reconfigure console streams to UTF-8 on Windows to prevent UnicodeEncodeError
# when printing non-ASCII characters (e.g. ♪ from Whisper transcriptions)
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

import librosa

from packaging import version

from modules import os_helper
from modules.init_interactive_mode import init_settings_interactive
from modules.Audio.denoise import denoise_vocal_audio
from modules.Audio.separation import separate_vocal_from_audio
from modules.Audio.vocal_chunks import (
    create_audio_chunks_from_transcribed_data,
    create_audio_chunks_from_ultrastar_data,
)
from modules.Audio.key_detector import detect_key_from_audio, get_allowed_notes_for_key
from modules.Audio.silence_processing import remove_silence_from_transcription_data, mute_no_singing_parts
from modules.Audio.separation import DemucsModel
from modules.Audio.convert_audio import convert_audio_to_mono_wav, convert_audio_format
from modules.Audio.youtube import (
    download_from_youtube,
)
from modules.Audio.bpm import get_bpm_from_file

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    gold_highlighted,
    red_highlighted,
    green_highlighted,
    cyan_highlighted,
    bright_green_highlighted,
)
from modules.Midi.midi_creator import (
    create_midi_segments_from_transcribed_data,
    create_repitched_midi_segments_from_ultrastar_txt,
    apply_octave_shift,
    correct_global_octave,
    correct_octave_outliers,
    correct_vocal_center,
    create_midi_file,
)
from modules.Midi.MidiSegment import MidiSegment
from modules.Midi.note_length_calculator import get_thirtytwo_note_second, get_sixteenth_note_second
from modules.Pitcher.pitcher import (
    get_pitch_with_file,
)
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitch_change_splitter import split_notes_at_pitch_changes
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.Speech_Recognition.hyphenation import (
    hyphenate_each_word,
)
from modules.Speech_Recognition.Whisper import transcribe_with_whisper
from modules.Ultrastar import (
    ultrastar_writer,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.Whisper import WhisperModel
from modules.Ultrastar.ultrastar_score_calculator import Score, calculate_score_points
from modules.Ultrastar.ultrastar_txt import FILE_ENCODING, FormatVersion
from modules.Ultrastar.coverter.ultrastar_txt_converter import from_ultrastar_txt, \
    create_ultrastar_txt_from_midi_segments, create_ultrastar_txt_from_automation
from modules.Ultrastar.ultrastar_parser import parse_ultrastar_txt
from modules.common_print import print_support, print_help, print_version
from modules.os_helper import check_file_exists, get_unused_song_output_dir
from modules.plot import create_plots
from modules.musicbrainz_client import search_musicbrainz
from modules.sheet import create_sheet
from modules.ProcessData import ProcessData, ProcessDataPaths, MediaInfo
from modules.DeviceDetection.device_detection import check_gpu_support
from modules.Image.image_helper import save_image
from modules.ffmpeg_helper import (
    is_ffmpeg_available,
    get_ffmpeg_and_ffprobe_paths,
    is_video_file,
    separate_audio_video,
    convert_to_ultrastar_format,
)

from Settings import Settings

settings = Settings()


def add_hyphen_to_data(
        transcribed_data: list[TranscribedData], hyphen_words: list[list[str]]
):
    """Add hyphen to transcribed data return new data list"""
    new_data = []

    for i, data in enumerate(transcribed_data):
        if not hyphen_words[i]:
            new_data.append(data)
        else:
            chunk_duration = data.end - data.start
            chunk_duration = chunk_duration / (len(hyphen_words[i]))

            next_start = data.start
            for j in enumerate(hyphen_words[i]):
                hyphenated_word_index = j[0]
                dup = copy.copy(data)
                dup.start = next_start
                next_start = data.end - chunk_duration * (
                        len(hyphen_words[i]) - 1 - hyphenated_word_index
                )
                dup.end = next_start
                dup.word = hyphen_words[i][hyphenated_word_index]
                dup.is_hyphen = True
                if hyphenated_word_index == len(hyphen_words[i]) - 1:
                    dup.is_word_end = True
                else:
                    dup.is_word_end = False
                new_data.append(dup)

    return new_data



def remove_unecessary_punctuations(transcribed_data: list[TranscribedData]) -> None:
    """Remove unecessary punctuations from transcribed data"""
    punctuation = ".,"
    for i, data in enumerate(transcribed_data):
        data.word = data.word.translate({ord(i): None for i in punctuation})


def run() -> tuple[str, Score, Score]:
    """The processing function of this program"""
    #List selected options (can add more later)
    if settings.keep_numbers:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Numbers will be transcribed as numerics (i.e. 1, 2, 3, etc.)')}")
    if settings.create_plot:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Plot will be created')}")
    if settings.keep_cache:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Cache folder will not be deleted')}")
    if settings.create_audio_chunks:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Audio chunks will be created')}")
    if not settings.create_karaoke:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Karaoke txt will not be created')}")
    if not settings.use_separated_vocal:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Vocals will not be separated')}")
    if not settings.hyphenation:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Hyphenation will not be applied')}")
    if settings.quantize_to_key:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Notes will be quantized to the detected musical key')}")
    if settings.bpm_override is not None:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted(f'BPM override: {settings.bpm_override}')}")
    if settings.octave_shift is not None:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted(f'Octave shift: {settings.octave_shift:+d}')}")
    if not settings.lyrics_lookup:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Lyrics lookup disabled')}")
    if settings.llm_correct_lyrics:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted(f'LLM lyric correction enabled (model: {settings.llm_model})')}")
    if settings.syllable_split:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Syllable-level note splitting enabled')}")
    if settings.vocal_gap_fill:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Vocal gap fill enabled')}")
    if settings.pitch_change_split:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Pitch-change split enabled')}")
    if settings.disable_reference_lyrics:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Reference-lyrics-first pipeline disabled')}")
    if settings.write_settings_info:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Settings info file will be written')}")

    process_data = InitProcessData()

    process_data.process_data_paths.cache_folder_path = (
        os.path.join(settings.output_folder_path, "cache")
        if settings.cache_override_path is None
        else settings.cache_override_path
    )

    # Create process audio — two separate paths:
    # whisper_audio_path: un-muted mono for Whisper, BPM, key detection
    # processing_audio_path: muted for pitch detection (SwiftF0)
    whisper_audio_path, pitch_audio_path = CreateProcessAudio(process_data)
    process_data.process_data_paths.whisper_audio_path = whisper_audio_path
    process_data.process_data_paths.processing_audio_path = pitch_audio_path

    # Get BPM — manual override takes precedence over auto-detection and .txt BPM
    if settings.bpm_override is not None:
        process_data.media_info.bpm = settings.bpm_override
        print(f"{ULTRASINGER_HEAD} Using manual BPM: {blue_highlighted(str(settings.bpm_override))}")
    elif not settings.input_file_is_ultrastar_txt:
        # Auto-detect from wav file (use un-muted audio — muting can distort librosa.tempo)
        process_data.media_info.bpm = get_bpm_from_file(process_data.process_data_paths.whisper_audio_path)

    # Detect key (use un-muted audio)
    detected_key, detected_mode = detect_key_from_audio(process_data.process_data_paths.whisper_audio_path)
    if process_data.media_info.music_key is None:
        process_data.media_info.music_key = f"{detected_key} {detected_mode}"

    # Audio transcription
    process_data.media_info.language = settings.language
    lyrics_lookup_result = None
    llm_result = None
    whisper_skipped = False

    # Early LRCLIB lookup: if synced lyrics are available, we can skip the
    # expensive Whisper transcription entirely (~2 min saved per song).
    # Language detection uses Whisper tiny (~2-3s) when not explicitly set.
    if (not settings.ignore_audio
            and settings.lyrics_lookup
            and not settings.disable_reference_lyrics
            and process_data.media_info.artist
            and process_data.media_info.title):
        try:
            from modules.lrclib_client import search_lyrics
            early_lyrics_info = search_lyrics(
                process_data.media_info.artist, process_data.media_info.title,
            )
            if early_lyrics_info is not None and early_lyrics_info.synced_lyrics:
                process_data.synced_lyrics = early_lyrics_info.synced_lyrics
                # Detect language if not explicitly set
                if process_data.media_info.language is None:
                    from modules.Speech_Recognition.Whisper import detect_language_from_audio
                    process_data.media_info.language = detect_language_from_audio(
                        process_data.process_data_paths.whisper_audio_path,
                        device=settings.pytorch_device,
                    )
                whisper_skipped = True
                print(
                    f"{ULTRASINGER_HEAD} "
                    f"{cyan_highlighted('Synced lyrics found — skipping Whisper transcription')}"
                )
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} Early lyrics lookup failed: {e}")

    if not settings.ignore_audio and not whisper_skipped:
        lyrics_lookup_result, llm_result = TranscribeAudio(process_data)

    # Onset correction — snap note starts to audio onsets for better timing
    if not settings.ignore_audio and not whisper_skipped and settings.onset_correction:
        try:
            from modules.Audio.onset_correction import detect_vocal_onsets, snap_to_onsets
            onset_times = detect_vocal_onsets(
                process_data.process_data_paths.whisper_audio_path
            )
            process_data.transcribed_data = snap_to_onsets(
                process_data.transcribed_data, onset_times
            )
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} Onset correction skipped: {e}")

    # Split syllables into segments (skip when Whisper was skipped — no transcribed_data yet)
    if not settings.ignore_audio and not whisper_skipped:
        process_data.transcribed_data = split_syllables_into_segments(process_data.transcribed_data,
                                                                  process_data.media_info.bpm)

    # Pitch audio
    process_data.pitched_data = pitch_audio(process_data.process_data_paths)

    # Fill vocal gaps (skip when Whisper was skipped — no transcribed_data yet)
    if not settings.ignore_audio and not whisper_skipped and settings.vocal_gap_fill:
        try:
            from modules.Audio.vocal_gap_fill import fill_vocal_gaps
            process_data.transcribed_data = fill_vocal_gaps(
                process_data.transcribed_data, process_data.pitched_data
            )
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} Vocal gap fill skipped: {e}")

    # Allowed keys for quantization
    allowed_notes_for_key = None
    if settings.quantize_to_key and not settings.ignore_audio:
        allowed_notes_for_key = get_allowed_notes_for_key(detected_key, detected_mode)

    # Create Midi_Segments
    reference_first_used = False
    if not settings.ignore_audio:
        # Reference-Lyrics-First pipeline: use LRCLIB synced lyrics + forced alignment
        # when available (produces dramatically better lyrics coverage and timing)
        if process_data.synced_lyrics and not settings.disable_reference_lyrics:
            try:
                from modules.Speech_Recognition.reference_lyrics_aligner import (
                    create_midi_segments_from_reference_lyrics,
                )
                ref_language = process_data.media_info.language
                if not ref_language:
                    ref_language = "en"
                    print(
                        f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} "
                        f"Language unknown — falling back to English alignment model"
                    )
                ref_segments = create_midi_segments_from_reference_lyrics(
                    synced_lyrics=process_data.synced_lyrics,
                    audio_path=process_data.process_data_paths.whisper_audio_path,
                    language=ref_language,
                    pitched_data=process_data.pitched_data,
                    device=settings.pytorch_device,
                    allowed_notes=allowed_notes_for_key,
                    melisma_split=settings.pitch_change_split,
                    align_model_name=settings.whisper_align_model,
                )
                if ref_segments:
                    process_data.midi_segments = ref_segments
                    reference_first_used = True
                    # Rebuild transcribed_data to match reference-derived segments
                    process_data.transcribed_data = [
                        TranscribedData(
                            word=seg.word,
                            start=seg.start,
                            end=seg.end,
                            confidence=1.0,
                            is_word_end=not seg.word.strip().startswith("~"),
                        )
                        for seg in process_data.midi_segments
                    ]
            except Exception as e:
                print(f"{ULTRASINGER_HEAD} Reference-first pipeline failed: {e}")
                print(f"{ULTRASINGER_HEAD} Falling back to standard pipeline")

        if not reference_first_used:
            # If Whisper was skipped but reference-first failed, run Whisper now
            if whisper_skipped:
                print(f"{ULTRASINGER_HEAD} Running Whisper as fallback")
                lyrics_lookup_result, llm_result = TranscribeAudio(process_data)
                whisper_skipped = False
                # Run the intermediate steps we skipped
                if settings.onset_correction:
                    try:
                        from modules.Audio.onset_correction import detect_vocal_onsets, snap_to_onsets
                        onset_times = detect_vocal_onsets(
                            process_data.process_data_paths.whisper_audio_path
                        )
                        process_data.transcribed_data = snap_to_onsets(
                            process_data.transcribed_data, onset_times
                        )
                    except Exception as e:
                        print(f"{ULTRASINGER_HEAD} Onset correction skipped: {e}")
                process_data.transcribed_data = split_syllables_into_segments(
                    process_data.transcribed_data, process_data.media_info.bpm
                )
                if settings.vocal_gap_fill:
                    try:
                        from modules.Audio.vocal_gap_fill import fill_vocal_gaps
                        process_data.transcribed_data = fill_vocal_gaps(
                            process_data.transcribed_data, process_data.pitched_data
                        )
                    except Exception as e:
                        print(f"{ULTRASINGER_HEAD} Vocal gap fill skipped: {e}")

            process_data.midi_segments = create_midi_segments_from_transcribed_data(
                process_data.transcribed_data,
                process_data.pitched_data,
                allowed_notes_for_key
            )

    else:
        process_data.midi_segments = create_repitched_midi_segments_from_ultrastar_txt(process_data.pitched_data,
                                                                                       process_data.parsed_file)

    # Create audio chunks after transcribed_data is finalized
    if settings.create_audio_chunks:
        create_audio_chunks(process_data)

    # Split notes at pitch change boundaries (melismas, runs)
    # (Skip when reference_first is active — it already handles pitch segmentation)
    if not settings.ignore_audio and settings.pitch_change_split and not reference_first_used:
        process_data.midi_segments = split_notes_at_pitch_changes(
            process_data.midi_segments, process_data.pitched_data
        )

    # Correct global octave shift (e.g. sub-harmonic detection)
    process_data.midi_segments = correct_global_octave(process_data.midi_segments)

    # Correct local octave outliers
    process_data.midi_segments = correct_octave_outliers(process_data.midi_segments)

    # Safety-net: shift notes toward vocal centre if still concentrated
    # outside the expected range (catches 100%-consistent wrong-octave)
    if settings.vocal_center_correction:
        process_data.midi_segments = correct_vocal_center(process_data.midi_segments)

    # Apply manual octave shift (after automatic correction, so user gets final say)
    if settings.octave_shift is not None:
        process_data.midi_segments = apply_octave_shift(process_data.midi_segments, settings.octave_shift)

    # Merge syllable segments
    # (Skip when reference_first is active — notes are already correctly segmented;
    # merging would undo that)
    if not settings.ignore_audio and not reference_first_used:
        process_data.midi_segments, process_data.transcribed_data = merge_syllable_segments(
            process_data.midi_segments,
            process_data.transcribed_data,
            process_data.media_info.bpm,
            preserve_syllables=settings.syllable_split,
        )

    # Reverse-scoring refinement pass
    if not settings.ignore_audio and settings.refine_from_vocal:
        from modules.Refinement.refine_from_vocal import refine_notes

        process_data.midi_segments = refine_notes(
            midi_segments=process_data.midi_segments,
            pitched_data=process_data.pitched_data,
            vocal_audio_path=process_data.process_data_paths.whisper_audio_path,
            bpm=process_data.media_info.bpm,
            refine_pitch_enabled=settings.refine_pitch,
            refine_timing_enabled=settings.refine_timing,
            timing_threshold_ms=settings.refine_timing_threshold,
            hit_ratio_threshold=settings.refine_hit_ratio,
        )

    # Create plot
    if settings.create_plot:
        create_plots(process_data, settings.output_folder_path)

    # Create Ultrastar txt
    accurate_score, simple_score, ultrastar_file_output = CreateUltraStarTxt(process_data)

    # Create Midi
    if settings.create_midi:
        create_midi_file(process_data.media_info.bpm, settings.output_folder_path, process_data.midi_segments,
                         process_data.basename)

    # Sheet music
    create_sheet(process_data.midi_segments, settings.output_folder_path,
                 process_data.process_data_paths.cache_folder_path, settings.musescore_path, process_data.basename,
                 process_data.media_info)

    # Write settings info file
    if settings.write_settings_info:
        _write_settings_info_file(
            settings.output_folder_path, simple_score, accurate_score,
            detected_language=process_data.media_info.language,
            lyrics_lookup_result=lyrics_lookup_result,
            llm_result=llm_result,
            reference_first_used=reference_first_used,
            whisper_skipped=whisper_skipped,
            has_synced_lyrics=process_data.synced_lyrics is not None,
        )

    # Cleanup
    if not settings.keep_cache:
        remove_cache_folder(process_data.process_data_paths.cache_folder_path)

    # Print Support
    print_support()
    return ultrastar_file_output, simple_score, accurate_score


def _write_settings_info_file(
        output_folder: str,
        simple_score: "Score | None",
        accurate_score: "Score | None",
        *,
        detected_language: str | None = None,
        lyrics_lookup_result=None,
        llm_result: "LLMResult | None" = None,
        reference_first_used: bool = False,
        whisper_skipped: bool = False,
        has_synced_lyrics: bool = False,
) -> None:
    """Write ultrasinger_parameter.info with all conversion settings and score results."""
    from datetime import datetime, timezone

    info_path = os.path.join(output_folder, "ultrasinger_parameter.info")
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"UltraSinger v{settings.APP_VERSION} — Conversion Settings\n")
            f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write("=" * 60 + "\n\n")

            # Input / Output
            f.write("[Input / Output]\n")
            f.write(f"  Input:                    {settings.input_file_path}\n")
            f.write(f"  Output folder:            {settings.output_folder_path}\n")
            f.write(f"  Format version:           {settings.format_version.value}\n")
            f.write("\n")

            # Transcription
            f.write("[Transcription]\n")
            f.write(f"  Whisper model:            {settings.whisper_model.value if hasattr(settings.whisper_model, 'value') else settings.whisper_model}\n")
            f.write(f"  Whisper batch size:       {settings.whisper_batch_size}\n")
            f.write(f"  Whisper compute type:     {settings.whisper_compute_type or 'auto'}\n")
            f.write(f"  Whisper align model:      {settings.whisper_align_model or '(default)'}\n")
            f.write(f"  Demucs model:             {settings.demucs_model.value if hasattr(settings.demucs_model, 'value') else settings.demucs_model}\n")
            lang_display = settings.language or "auto-detect"
            if not settings.language and detected_language:
                lang_display = f"auto-detect → {detected_language}"
            f.write(f"  Language:                 {lang_display}\n")
            f.write(f"  Keep numbers:             {settings.keep_numbers}\n")
            f.write(f"  VAD onset:                {settings.vad_onset}\n")
            f.write(f"  VAD offset:               {settings.vad_offset}\n")
            f.write(f"  No-speech threshold:      {settings.no_speech_threshold}\n")
            f.write("\n")

            # Pipeline
            f.write("[Pipeline]\n")
            if reference_first_used:
                f.write(f"  Pipeline:                 Reference-Lyrics-First\n")
                f.write(f"  LRCLIB synced lyrics:     found\n")
                f.write(f"  Whisper transcription:    skipped\n")
                f.write(f"  Alignment:                wav2vec2 CTC forced alignment\n")
            elif has_synced_lyrics:
                f.write(f"  Pipeline:                 Whisper (reference-first failed, fell back)\n")
                f.write(f"  LRCLIB synced lyrics:     found (but alignment failed)\n")
                f.write(f"  Whisper transcription:    full\n")
            elif whisper_skipped:
                f.write(f"  Pipeline:                 Whisper skipped (no audio)\n")
                f.write(f"  LRCLIB synced lyrics:     not found\n")
            else:
                f.write(f"  Pipeline:                 Standard Whisper\n")
                f.write(f"  LRCLIB synced lyrics:     not found\n")
                f.write(f"  Whisper transcription:    full\n")
                f.write(f"  Alignment:                WhisperX wav2vec2\n")
            if whisper_skipped and not reference_first_used:
                lang_method = "Whisper tiny (fast detection)"
            elif settings.language:
                lang_method = f"manual (--language {settings.language})"
            else:
                lang_method = "Whisper (full transcription)"
            f.write(f"  Language detection:        {lang_method}\n")
            f.write("\n")

            # Post-processing
            f.write("[Post-Processing]\n")
            f.write(f"  Hyphenation:              {settings.hyphenation}\n")
            f.write(f"  Vocal separation:         {settings.use_separated_vocal}\n")
            f.write(f"  Quantize to key:          {settings.quantize_to_key}\n")
            f.write(f"  Vocal center correction:  {settings.vocal_center_correction}\n")
            f.write(f"  Onset correction:         {settings.onset_correction}\n")
            f.write(f"  Syllable split:           {settings.syllable_split}\n")
            f.write(f"  Vocal gap fill:           {settings.vocal_gap_fill}\n")
            f.write(f"  Pitch-change split:       {settings.pitch_change_split}\n")
            f.write(f"  Reference lyrics:         {not settings.disable_reference_lyrics}\n")
            f.write(f"  Noise reduction:          {settings.denoise_noise_reduction} dB\n")
            f.write(f"  Noise floor:              {settings.denoise_noise_floor} dB\n")
            f.write(f"  Noise floor tracking:     {settings.denoise_track_noise}\n")
            f.write("\n")

            # Overrides
            f.write("[Overrides]\n")
            f.write(f"  BPM override:             {settings.bpm_override or '(none)'}\n")
            f.write(f"  Octave shift:             {settings.octave_shift if settings.octave_shift is not None else '(none)'}\n")
            f.write("\n")

            # Output options
            f.write("[Output Options]\n")
            f.write(f"  Create MIDI:              {settings.create_midi}\n")
            f.write(f"  Create plots:             {settings.create_plot}\n")
            f.write(f"  Create audio chunks:      {settings.create_audio_chunks}\n")
            f.write(f"  Create karaoke:           {settings.create_karaoke}\n")
            f.write(f"  Keep audio in video:      {settings.keep_audio_in_video}\n")
            f.write(f"  Keep cache:               {settings.keep_cache}\n")
            f.write("\n")

            # Device
            f.write("[Device]\n")
            f.write(f"  PyTorch device:           {settings.pytorch_device}\n")
            f.write(f"  Force CPU:                {settings.force_cpu}\n")
            f.write(f"  Force Whisper CPU:        {settings.force_whisper_cpu}\n")
            import torch
            f.write(f"  cuDNN deterministic:      {torch.backends.cudnn.deterministic}\n")
            f.write(f"  cuDNN benchmark:          {torch.backends.cudnn.benchmark}\n")
            f.write(f"  Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}\n")
            f.write(f"  CUBLAS_WORKSPACE_CONFIG:  {os.environ.get('CUBLAS_WORKSPACE_CONFIG', '(not set)')}\n")
            from modules.DeviceDetection.device_detection import nondeterministic_warnings
            if nondeterministic_warnings:
                f.write(f"  Non-deterministic ops:    {len(nondeterministic_warnings)} detected (would crash in strict mode)\n")
                for w in nondeterministic_warnings:
                    f.write(f"    - {w}\n")
            else:
                f.write(f"  Non-deterministic ops:    none detected\n")
            f.write("\n")

            # Refinement
            f.write("[Refinement]\n")
            f.write(f"  Enabled:                  {settings.refine_from_vocal}\n")
            if settings.refine_from_vocal:
                f.write(f"  Pitch refinement:         {settings.refine_pitch}\n")
                f.write(f"  Timing refinement:        {settings.refine_timing}\n")
                f.write(f"  Hit ratio threshold:      {settings.refine_hit_ratio}\n")
                f.write(f"  Timing threshold:         {settings.refine_timing_threshold} ms\n")
            f.write("\n")

            # Lyrics Lookup
            f.write("[Lyrics Lookup]\n")
            f.write(f"  Enabled:                  {settings.lyrics_lookup}\n")
            if lyrics_lookup_result is not None:
                f.write(f"  Words corrected:          {lyrics_lookup_result.words_corrected}\n")
                f.write(f"  Words kept:               {lyrics_lookup_result.words_kept}\n")
                f.write(f"  Words total:              {lyrics_lookup_result.words_total}\n")
                f.write(f"  Reference words:          {lyrics_lookup_result.reference_words}\n")
            f.write("\n")

            # LLM
            f.write("[LLM Lyric Correction]\n")
            f.write(f"  Enabled:                  {settings.llm_correct_lyrics}\n")
            if settings.llm_correct_lyrics:
                f.write(f"  API base URL:             {settings.llm_api_base_url}\n")
                f.write(f"  Model:                    {settings.llm_model}\n")
                retry_str = f"yes ({settings.llm_retry_max}x, {settings.llm_retry_wait}s wait)" if settings.llm_retry_on_rate_limit else "no"
                f.write(f"  Retry on rate limit:      {retry_str}\n")
                if llm_result is not None:
                    retry_info = f", {llm_result.retries} retries" if llm_result.retries > 0 else ""
                    if llm_result.errors == 0:
                        f.write(f"  Status:                   OK ({llm_result.chunks_ok}/{llm_result.chunks_total} chunks, {llm_result.corrections} corrections{retry_info})\n")
                    elif llm_result.chunks_ok > 0:
                        f.write(f"  Status:                   PARTIAL ({llm_result.chunks_ok}/{llm_result.chunks_total} chunks OK, {llm_result.errors} errors, {llm_result.corrections} corrections{retry_info})\n")
                    else:
                        f.write(f"  Status:                   FAILED (all {llm_result.chunks_total} chunks failed{retry_info})\n")
                    if llm_result.last_error:
                        f.write(f"  Last error:               {llm_result.last_error}\n")
            f.write("\n")

            # Score results
            if simple_score is not None or accurate_score is not None:
                f.write("=" * 60 + "\n")
                f.write("[Score Results]\n")
                if simple_score is not None:
                    pct = round(simple_score.score / 100, 2)
                    f.write(f"  Simple (octave-ignoring):\n")
                    f.write(f"    Total:                  {simple_score.score} ({pct}%)\n")
                    f.write(f"    Notes:                  {simple_score.notes}\n")
                    f.write(f"    Golden notes:           {simple_score.golden}\n")
                    f.write(f"    Line bonus:             {simple_score.line_bonus}\n")
                    f.write(f"    Max possible:           {simple_score.max_score}\n")
                if accurate_score is not None:
                    pct = round(accurate_score.score / 100, 2)
                    f.write(f"  Accurate (octave-matched):\n")
                    f.write(f"    Total:                  {accurate_score.score} ({pct}%)\n")
                    f.write(f"    Notes:                  {accurate_score.notes}\n")
                    f.write(f"    Golden notes:           {accurate_score.golden}\n")
                    f.write(f"    Line bonus:             {accurate_score.line_bonus}\n")
                    f.write(f"    Max possible:           {accurate_score.max_score}\n")

        print(f"{ULTRASINGER_HEAD} Settings info written to {blue_highlighted(info_path)}")
    except OSError as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Could not write settings info: {e}")


def split_syllables_into_segments(
        transcribed_data: list[TranscribedData],
        real_bpm: float) -> list[TranscribedData]:
    """Split every syllable into sub-segments

    This splits long syllables (including hyphenated ones) into smaller segments
    to allow detect for pitch changes within a syllable (e.g., singing a scale).
    """
    syllable_segment_size = get_sixteenth_note_second(real_bpm)
    thirtytwo_note = get_thirtytwo_note_second(real_bpm)

    segment_size_decimal_points = len(str(syllable_segment_size).split(".")[1])
    new_data = []

    for i, data in enumerate(transcribed_data):
        duration = data.end - data.start

        # If duration is less than or equal to a 16th note, don't split
        if duration <= syllable_segment_size:
            new_data.append(data)
            continue

        has_space = str(data.word).endswith(" ")
        first_segment = copy.deepcopy(data)
        filler_words_start = data.start + syllable_segment_size
        remainder = data.end - filler_words_start
        first_segment.end = filler_words_start
        if has_space:
            first_segment.word = first_segment.word[:-1]

        first_segment.is_word_end = False
        new_data.append(first_segment)

        full_segments, partial_segment = divmod(remainder, syllable_segment_size)

        if full_segments >= 1:
            first_segment.is_hyphen = True
            for i in range(int(full_segments)):
                segment = TranscribedData()
                segment.word = "~"
                segment.start = filler_words_start + round(
                    i * syllable_segment_size, segment_size_decimal_points
                )
                segment.end = segment.start + syllable_segment_size
                segment.is_hyphen = True
                segment.is_word_end = False
                new_data.append(segment)

        # Only add a partial_segment if it's at least as long as a 32nd note
        # Otherwise add it to the last note
        if partial_segment >= thirtytwo_note:
            first_segment.is_hyphen = True
            segment = TranscribedData()
            segment.word = "~"
            segment.start = filler_words_start + round(
                full_segments * syllable_segment_size, segment_size_decimal_points
            )
            segment.end = segment.start + partial_segment
            segment.is_hyphen = True
            segment.is_word_end = False
            new_data.append(segment)
        elif full_segments >= 1 or len(new_data) > 0:
            # Add remaining time to the last note
            new_data[-1].end += partial_segment

        if has_space:
            new_data[-1].word += " "
            new_data[-1].is_word_end = True
    return new_data


def merge_syllable_segments(midi_segments: list[MidiSegment],
                            transcribed_data: list[TranscribedData],
                            real_bpm: float,
                            preserve_syllables: bool = False) -> tuple[list[MidiSegment], list[TranscribedData]]:
    """Merge sub-segments of a syllable where the pitch is the same

    This function handles three cases:
    1. Merge consecutive ~ segments with the SAME pitch (same note held)
    2. Detect and merge SLIDES: short ~ segments with ±1-2 semitone jumps between syllables
    3. Merge ANY consecutive segments (including regular syllables) with the SAME pitch into one word

    When *preserve_syllables* is ``True``, case 3 is skipped so that
    hyphenated syllables stay as separate notes even when they share
    the same pitch.  After the main merge loop, any surviving ``~``
    markers are absorbed into adjacent real syllables.
    """

    sixteenth_note = get_sixteenth_note_second(real_bpm)
    # Slides are typically very short (1-2 16th notes)
    max_slide_duration = sixteenth_note * 2

    new_data = []
    new_midi_notes = []

    previous_data = None

    for i, data in enumerate(transcribed_data):
        # Check if previous element exists
        is_same_note = i > 0 and midi_segments[i].note == midi_segments[i - 1].note
        has_breath_pause = False

        if previous_data is not None:
            has_breath_pause = (data.start - previous_data.end) > sixteenth_note

        # Check if current word is a ~ segment (continuation marker)
        current_word_stripped = str(data.word).strip()
        is_tilde_segment = current_word_stripped == "~"

        # Slide detection: Detect short ~ segments with small pitch jumps
        is_potential_slide = False
        if i > 0 and is_tilde_segment:
            duration = data.end - data.start

            # Calculate pitch jump in semitones
            try:
                prev_midi = librosa.note_to_midi(midi_segments[i - 1].note)
                curr_midi = librosa.note_to_midi(midi_segments[i].note)
                semitone_diff = abs(curr_midi - prev_midi)

                # Slide: Short duration AND small pitch jump (1-2 semitones)
                is_potential_slide = (duration <= max_slide_duration and
                                     semitone_diff <= 2 and
                                     semitone_diff > 0)
            except:
                # Ignore slide detection on error
                pass

        # Check if current segment should be merged with previous due to same pitch
        should_merge_same_pitch = False
        if (i > 0 and
            previous_data is not None and
            not is_tilde_segment and  # Not a ~ segment
            is_same_note and
            not has_breath_pause and
            not previous_data.is_word_end and  # Don't merge across word boundaries
            not preserve_syllables):  # Skip when preserving syllables
            should_merge_same_pitch = True

        should_merge_tilde = (is_tilde_segment and
                             previous_data is not None and
                             (is_same_note or is_potential_slide) and
                             not has_breath_pause)

        if should_merge_tilde:
            new_data[-1].end = data.end
            new_midi_notes[-1].end = data.end

            # For slides: Keep the original note (not the transition note)
            if is_potential_slide and not is_same_note:
                new_midi_notes[-1].note = midi_segments[i - 1].note

            # Take over space and word_end flag from current segment
            # "~ " means end of word - add space to previous segment
            if str(data.word).endswith(" "):
                if not new_data[-1].word.endswith(" "):
                    new_data[-1].word += " "
                if not new_midi_notes[-1].word.endswith(" "):
                    new_midi_notes[-1].word += " "
                new_data[-1].is_word_end = True

        elif should_merge_same_pitch:
            # Merge regular syllable with previous syllable (same pitch)
            new_data[-1].end = data.end
            new_midi_notes[-1].end = data.end

            # Check if current word has space at end before stripping
            has_space = str(data.word).endswith(" ")

            # Concatenate the words (remove trailing space from previous, add current word)
            prev_word = new_data[-1].word.rstrip()
            curr_word = data.word.rstrip()

            # Remove ~ from BOTH previous and current word if present
            if prev_word == "~":
                prev_word = ""
            if curr_word.startswith("~"):
                curr_word = curr_word[1:]

            # Concatenate
            new_data[-1].word = prev_word + curr_word
            new_midi_notes[-1].word = prev_word + curr_word

            # Preserve space at end if current segment had it
            if has_space:
                new_data[-1].word += " "
                new_midi_notes[-1].word += " "

            if data.is_word_end:
                new_data[-1].is_word_end = True
                new_data[-1].is_hyphen = False
            else:
                # Keep hyphen status if either segment was hyphenated
                new_data[-1].is_hyphen = new_data[-1].is_hyphen or data.is_hyphen

        else:
            # Add as new segment
            new_data.append(data)
            new_midi_notes.append(midi_segments[i])

        previous_data = data

    # When preserving syllables, absorb any surviving ~ markers into
    # adjacent real segments so no tilde text appears in the output.
    if preserve_syllables:
        new_midi_notes, new_data = _absorb_tilde_segments(new_midi_notes, new_data)

    return new_midi_notes, new_data


def _absorb_tilde_segments(
    midi_segments: list[MidiSegment],
    transcribed_data: list[TranscribedData],
) -> tuple[list[MidiSegment], list[TranscribedData]]:
    """Absorb surviving ``~`` segments into the nearest real syllable.

    After the main merge loop with ``preserve_syllables=True``, some
    ``~`` markers may survive when they have a large pitch jump from
    the previous segment.  This function merges each ``~`` into the
    adjacent segment whose pitch matches best (preferring the next
    real syllable, falling back to the previous one).
    """
    if not transcribed_data:
        return midi_segments, transcribed_data

    result_data: list[TranscribedData] = []
    result_midi: list[MidiSegment] = []

    for i, data in enumerate(transcribed_data):
        is_tilde = str(data.word).strip() == "~"
        if not is_tilde:
            result_data.append(data)
            result_midi.append(midi_segments[i])
            continue

        # Try to absorb into previous segment
        if result_data:
            result_data[-1].end = data.end
            result_midi[-1].end = data.end
            # Carry over word_end and trailing space
            if str(data.word).endswith(" "):
                if not result_data[-1].word.endswith(" "):
                    result_data[-1].word += " "
                result_data[-1].is_word_end = True
        else:
            # No previous segment — keep as-is (shouldn't normally happen)
            result_data.append(data)
            result_midi.append(midi_segments[i])

    return result_midi, result_data


def create_audio_chunks(process_data):
    if not settings.ignore_audio:
        create_audio_chunks_from_transcribed_data(
            process_data.process_data_paths,
            process_data.transcribed_data)
    else:
        create_audio_chunks_from_ultrastar_data(
            process_data.process_data_paths,
            process_data.parsed_file
        )


def InitProcessData():
    settings.input_file_is_ultrastar_txt = settings.input_file_path.endswith(".txt")
    if settings.input_file_is_ultrastar_txt:
        # Parse Ultrastar txt
        (
            basename,
            settings.output_folder_path,
            audio_file_path,
            ultrastar_class,
            audio_extension,
        ) = parse_ultrastar_txt(settings.input_file_path, settings.output_folder_path)
        process_data = from_ultrastar_txt(ultrastar_class)
        process_data.basename = basename
        process_data.process_data_paths.audio_output_file_path = audio_file_path
        process_data.media_info.audio_extension = audio_extension
        # todo: ignore transcribe
        settings.ignore_audio = True

    elif settings.input_file_path.startswith("https:"):
        # Youtube
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Full Automatic Mode')}")
        process_data = ProcessData()
        (
            process_data.basename,
            settings.output_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            process_data.media_info
        ) = download_from_youtube(
            settings.input_file_path, settings.output_folder_path, settings.cookiefile,
            keep_audio_in_video=settings.keep_audio_in_video,
        )
    else:
        # Audio/Video File
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Full Automatic Mode')}")
        process_data = ProcessData()

        # If a YouTube URL was provided (e.g. from browser interceptor),
        # fetch metadata BEFORE deriving the local song identity so that
        # the output folder and basename use the correct artist/title
        # instead of the temporary intercepted filename.
        yt_artist, yt_title = None, None
        if settings.youtube_url:
            try:
                from modules.Audio.youtube import get_youtube_title
                yt_artist, yt_title, _video_title = get_youtube_title(
                    settings.youtube_url, settings.cookiefile
                )
                print(
                    f"{ULTRASINGER_HEAD} YouTube metadata: "
                    f"{yt_artist} - {yt_title}"
                )
            except Exception as e:
                print(
                    f"{ULTRASINGER_HEAD} Warning: YouTube metadata lookup "
                    f"failed: {e}"
                )

        (
            process_data.basename,
            settings.output_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            process_data.media_info,
        ) = infos_from_audio_video_input_file()

        # Override with YouTube metadata if available
        if yt_artist:
            process_data.media_info.artist = yt_artist
            process_data.basename = f"{yt_artist} - {yt_title}" if yt_title else yt_artist
        if yt_title:
            process_data.media_info.title = yt_title

    return process_data


def TranscribeAudio(process_data):
    # Use un-muted audio for Whisper — muted audio causes WhisperX's wav2vec2
    # forced alignment to lose sync at zero→audio transitions
    transcription_result = transcribe_audio(process_data.process_data_paths.cache_folder_path,
                                            process_data.process_data_paths.whisper_audio_path)

    if process_data.media_info.language is None:
        process_data.media_info.language = transcription_result.detected_language

    process_data.transcribed_data = transcription_result.transcribed_data

    # Lyrics lookup correction (before LLM — uses verified reference lyrics)
    lyrics_lookup_result = None
    if settings.lyrics_lookup and process_data.media_info.artist and process_data.media_info.title:
        try:
            from modules.lrclib_client import search_lyrics
            from modules.Speech_Recognition.lyrics_corrector import correct_transcription_from_lyrics
            lyrics_info = search_lyrics(process_data.media_info.artist, process_data.media_info.title)
            if lyrics_info is not None:
                # Save synced lyrics for reference-first pipeline (independent of plain lyrics)
                if lyrics_info.synced_lyrics:
                    process_data.synced_lyrics = lyrics_info.synced_lyrics
                if lyrics_info.plain_lyrics:
                    process_data.transcribed_data, lyrics_lookup_result = correct_transcription_from_lyrics(
                        process_data.transcribed_data, lyrics_info.plain_lyrics
                    )
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} Lyrics lookup correction skipped: {e}")

    # LLM lyric correction (optional, before punctuation removal for context)
    llm_result = None
    if settings.llm_correct_lyrics:
        try:
            from modules.Speech_Recognition.llm_corrector import correct_lyrics_with_llm, LLMConfig
            llm_config = LLMConfig(
                api_base_url=settings.llm_api_base_url,
                api_key=settings.llm_api_key or os.environ.get("LLM_API_KEY", ""),
                model=settings.llm_model,
                language=process_data.media_info.language,
                artist=process_data.media_info.artist,
                title=process_data.media_info.title,
                retry_on_rate_limit=settings.llm_retry_on_rate_limit,
                retry_wait=settings.llm_retry_wait,
                retry_max=settings.llm_retry_max,
            )
            process_data.transcribed_data, llm_result = correct_lyrics_with_llm(
                process_data.transcribed_data, llm_config
            )
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} LLM lyric correction skipped: {e}")

    # Hyphen
    # Todo: Is it really unnecessary?
    remove_unecessary_punctuations(process_data.transcribed_data)
    if settings.hyphenation:
        hyphen_words = hyphenate_each_word(process_data.media_info.language, process_data.transcribed_data)

        if hyphen_words is not None:
            process_data.transcribed_data = add_hyphen_to_data(process_data.transcribed_data, hyphen_words)

    # Use un-muted audio for silence detection — running on muted audio
    # detects more silence than actually exists in the original
    process_data.transcribed_data = remove_silence_from_transcription_data(
        process_data.process_data_paths.whisper_audio_path, process_data.transcribed_data
    )

    return lyrics_lookup_result, llm_result


def CreateUltraStarTxt(process_data: ProcessData):
    # Move instrumental and vocals
    if settings.create_karaoke and version.parse(settings.format_version.value) < version.parse(
            FormatVersion.V1_1_0.value):
        karaoke_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Karaoke]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.instrumental_audio_file_path, karaoke_output_path)

    if version.parse(settings.format_version.value) >= version.parse(FormatVersion.V1_1_0.value):
        instrumental_output_path = os.path.join(settings.output_folder_path,
                                                process_data.basename + " [Instrumental]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.instrumental_audio_file_path, instrumental_output_path)
        vocals_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Vocals]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.vocals_audio_file_path, vocals_output_path)

    # Create Ultrastar txt
    if not settings.ignore_audio:
        ultrastar_file_output = create_ultrastar_txt_from_automation(
            process_data.basename,
            settings.output_folder_path,
            process_data.midi_segments,
            process_data.media_info,
            settings.format_version,
            settings.create_karaoke,
            settings.APP_VERSION
        )
    else:
        ultrastar_file_output = create_ultrastar_txt_from_midi_segments(
            settings.output_folder_path, settings.input_file_path, process_data.media_info.title,
            process_data.midi_segments
        )

    # Calc Points
    simple_score = None
    accurate_score = None
    if settings.calculate_score:
        simple_score, accurate_score = calculate_score_points(process_data, ultrastar_file_output)

    # Add calculated score to Ultrastar txt
    #Todo: Missing Karaoke
    ultrastar_writer.add_score_to_ultrastar_txt(ultrastar_file_output, simple_score)
    return accurate_score, simple_score, ultrastar_file_output


def CreateProcessAudio(process_data) -> tuple[str, str]:
    """Create processed audio files.

    Returns two paths:
    - whisper_audio_path: denoised mono audio (no muting) for Whisper transcription,
      BPM detection, key detection, and silence-based transcription cleanup.
    - pitch_audio_path: muted audio for pitch detection (SwiftF0).

    Muting zeroes silent sections which helps pitch detection focus on singing,
    but destroys onset information that WhisperX's forced alignment needs.
    """
    os_helper.create_folder(process_data.process_data_paths.cache_folder_path)

    # Separate vocal from audio
    audio_separation_folder_path = separate_vocal_from_audio(
        process_data.process_data_paths.cache_folder_path,
        process_data.process_data_paths.audio_output_file_path,
        settings.use_separated_vocal,
        settings.create_karaoke,
        settings.pytorch_device,
        settings.demucs_model,
        settings.skip_cache_vocal_separation
    )
    process_data.process_data_paths.vocals_audio_file_path = os.path.join(audio_separation_folder_path, "vocals.wav")
    process_data.process_data_paths.instrumental_audio_file_path = os.path.join(audio_separation_folder_path,
                                                                                "no_vocals.wav")

    if settings.use_separated_vocal:
        input_path = process_data.process_data_paths.vocals_audio_file_path
    else:
        input_path = process_data.process_data_paths.audio_output_file_path

    # Denoise vocal audio
    # Include denoise parameters in cache filename so changed settings invalidate the cache
    denoise_config = f"nr{float(settings.denoise_noise_reduction):.1f}_nf{float(settings.denoise_noise_floor):.1f}_tn{int(settings.denoise_track_noise)}"
    denoised_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + f"_denoised_{denoise_config}.wav"
    )
    denoise_vocal_audio(input_path, denoised_output_path,
                        skip_cache=settings.skip_cache_denoise_vocal_audio,
                        noise_reduction=settings.denoise_noise_reduction,
                        noise_floor=settings.denoise_noise_floor,
                        track_noise=settings.denoise_track_noise)

    # Convert to mono audio — this is the Whisper path (no muting)
    mono_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_mono.wav"
    )
    convert_audio_to_mono_wav(denoised_output_path, mono_output_path)
    whisper_audio_path = mono_output_path

    # Mute silence sections — this is the pitch detection path
    mute_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_mute.wav"
    )
    mute_no_singing_parts(mono_output_path, mute_output_path)
    pitch_audio_path = mute_output_path

    return whisper_audio_path, pitch_audio_path


def transcribe_audio(cache_folder_path: str, audio_path: str) -> TranscriptionResult:
    """Transcribe audio with AI"""
    transcription_result = None
    whisper_align_model_string = None
    if settings.transcriber == "whisper":
        if settings.whisper_align_model is not None:
            whisper_align_model_string = settings.whisper_align_model.replace("/", "_")
        whisper_device = "cpu" if settings.force_whisper_cpu else settings.pytorch_device
        transcription_config = f"{settings.transcriber}_{settings.whisper_model.value}_{whisper_device}_{whisper_align_model_string}_{settings.whisper_batch_size}_{settings.whisper_compute_type}_{settings.language}_vad{settings.vad_onset}_{settings.vad_offset}_nst{settings.no_speech_threshold}_unmuted"
        transcription_path = os.path.join(cache_folder_path, f"{transcription_config}.json")
        cached_transcription_available = check_file_exists(transcription_path)
        if settings.skip_cache_transcription or not cached_transcription_available:
            transcription_result = transcribe_with_whisper(
                audio_path,
                settings.whisper_model,
                whisper_device,
                settings.whisper_align_model,
                settings.whisper_batch_size,
                settings.whisper_compute_type,
                settings.language,
                settings.keep_numbers,
                vad_onset=settings.vad_onset,
                vad_offset=settings.vad_offset,
                no_speech_threshold=settings.no_speech_threshold,
            )
            with open(transcription_path, "w", encoding=FILE_ENCODING) as file:
                file.write(transcription_result.to_json())
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached transcribed data")
            with open(transcription_path) as file:
                json = file.read()
                transcription_result = TranscriptionResult.from_json(json)
    else:
        raise NotImplementedError
    return transcription_result


def infos_from_audio_video_input_file() -> tuple[str, str, str, MediaInfo]:
    """Infos from audio/video input file"""
    basename = os.path.basename(settings.input_file_path)
    basename_without_ext = os.path.splitext(basename)[0]

    artist, title = None, None
    if " - " in basename_without_ext:
        artist, title = basename_without_ext.split(" - ", 1)
    else:
        title = basename_without_ext

    song_info = search_musicbrainz(title, artist)
    basename_without_ext = f"{song_info.artist} - {song_info.title}"

    song_folder_output_path = os.path.join(settings.output_folder_path, basename_without_ext)
    song_folder_output_path = get_unused_song_output_dir(song_folder_output_path)
    os_helper.create_folder(song_folder_output_path)

    extension = os.path.splitext(basename)[1]
    if is_video_file(settings.input_file_path):
        print(f"{ULTRASINGER_HEAD} Video file detected - separating audio and video")

        video_with_audio_basename = f"{basename_without_ext}{extension}"
        video_with_audio_path = os.path.join(song_folder_output_path, video_with_audio_basename)
        os_helper.copy(settings.input_file_path, video_with_audio_path)

        # Separate audio and video
        ultrastar_audio_input_path, final_video_path, audio_ext, video_ext = separate_audio_video(
            video_with_audio_path, basename_without_ext, song_folder_output_path,
            keep_audio_in_video=settings.keep_audio_in_video,
        )
    else:
        # Audio file
        basename_with_ext = f"{basename_without_ext}{extension}"
        audio_ext = extension.lstrip('.')
        video_ext = None
        os_helper.copy(settings.input_file_path, song_folder_output_path)
        os_helper.rename(
            os.path.join(song_folder_output_path, os.path.basename(settings.input_file_path)),
            os.path.join(song_folder_output_path, basename_with_ext),
        )
        ultrastar_audio_input_path = os.path.join(song_folder_output_path, basename_with_ext)

        # Convert to UltraStar-compatible format if needed
        ultrastar_audio_input_path, audio_ext = convert_to_ultrastar_format(
            ultrastar_audio_input_path, basename_without_ext, song_folder_output_path, audio_ext
        )

    # Todo: Read ID3 tags
    if song_info.cover_image_data is not None:
        save_image(song_info.cover_image_data, basename_without_ext, song_folder_output_path)

    return (
        basename_without_ext,
        song_folder_output_path,
        ultrastar_audio_input_path,
        MediaInfo(
            artist=song_info.artist,
            title=song_info.title,
            year=song_info.year,
            genre=song_info.genres,
            cover_url=song_info.cover_url,
            audio_extension=audio_ext,
            video_extension=video_ext
        ),
    )


def pitch_audio(
        process_data_paths: ProcessDataPaths) -> PitchedData:
    """Pitch audio"""

    pitching_config = f"swiftf0_{settings.ignore_audio}"
    pitched_data_path = os.path.join(process_data_paths.cache_folder_path, f"{pitching_config}.json")
    cache_available = check_file_exists(pitched_data_path)

    if settings.skip_cache_pitch_detection or not cache_available:
        pitched_data = get_pitch_with_file(
            process_data_paths.processing_audio_path
        )

        pitched_data_json = pitched_data.to_json()
        with open(pitched_data_path, "w", encoding=FILE_ENCODING) as file:
            file.write(pitched_data_json)
    else:
        print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached pitch data")
        with open(pitched_data_path) as file:
            json = file.read()
            pitched_data = PitchedData.from_json(json)

    return pitched_data


def main(argv: list[str]) -> None:
    """Main function"""
    print_version(settings.APP_VERSION)
    init_settings(argv)
    check_requirements()
    if settings.interactive_mode:
        init_settings_interactive(settings)
    run()
    sys.exit()


def check_requirements() -> None:
    if not settings.force_cpu:
        settings.pytorch_device = check_gpu_support()
    print(f"{ULTRASINGER_HEAD} ----------------------")

    if not is_ffmpeg_available(settings.user_ffmpeg_path):
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} {blue_highlighted('FFmpeg')} {red_highlighted('is not available. Provide --ffmpeg ‘path’ or install FFmpeg with PATH')}")
        sys.exit(1)
    else:
        ffmpeg_path, ffprobe_path = get_ffmpeg_and_ffprobe_paths()
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('FFmpeg')} - using {red_highlighted(ffmpeg_path)}")
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('FFprobe')} - using {red_highlighted(ffprobe_path)}")

    print(f"{ULTRASINGER_HEAD} ----------------------")

def remove_cache_folder(cache_folder_path: str) -> None:
    """Remove cache folder"""
    os_helper.remove_folder(cache_folder_path)


def init_settings(argv: list[str]) -> Settings:
    """Init settings"""
    long, short = arg_options()
    opts, args = getopt.getopt(argv, short, long)
    if len(opts) == 0:
        print_help()
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            print_help()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            settings.input_file_path = arg
        elif opt in ("-o", "--ofile"):
            settings.output_folder_path = arg
        elif opt == "--bpm":
            try:
                val = float(arg)
            except ValueError:
                print(f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} --bpm must be a positive number, got {arg}")
                sys.exit(1)
            if val <= 0:
                print(f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} --bpm must be a positive number, got {val}")
                sys.exit(1)
            settings.bpm_override = val
        elif opt == "--octave":
            try:
                val = int(arg)
            except ValueError:
                print(f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} --octave must be an integer, got {arg}")
                sys.exit(1)
            settings.octave_shift = val
        elif opt in ("--whisper"):
            settings.transcriber = "whisper"

            #Addition of whisper model choice. Added error handling for unknown models.
            try:
                settings.whisper_model = WhisperModel(arg)
            except ValueError as ve:
                print(f"{ULTRASINGER_HEAD} The model {arg} is not a valid whisper model selection. Please use one of the following models: {blue_highlighted(', '.join([m.value for m in WhisperModel]))}")
                sys.exit()
        elif opt in ("--whisper_align_model"):
            settings.whisper_align_model = arg
        elif opt in ("--whisper_batch_size"):
            settings.whisper_batch_size = int(arg)
        elif opt in ("--whisper_compute_type"):
            settings.whisper_compute_type = arg
        elif opt in ("--keep_numbers"):
            settings.keep_numbers = True
        elif opt in ("--vad_onset"):
            val = float(arg)
            if not 0.0 <= val <= 1.0:
                print(f"{ULTRASINGER_HEAD} Error: --vad_onset must be between 0.0 and 1.0, got {val}")
                sys.exit(1)
            settings.vad_onset = val
        elif opt in ("--vad_offset"):
            val = float(arg)
            if not 0.0 <= val <= 1.0:
                print(f"{ULTRASINGER_HEAD} Error: --vad_offset must be between 0.0 and 1.0, got {val}")
                sys.exit(1)
            settings.vad_offset = val
        elif opt in ("--no_speech_threshold"):
            val = float(arg)
            if not 0.0 <= val <= 1.0:
                print(f"{ULTRASINGER_HEAD} Error: --no_speech_threshold must be between 0.0 and 1.0, got {val}")
                sys.exit(1)
            settings.no_speech_threshold = val
        elif opt in ("--language"):
            settings.language = arg
        elif opt in ("--plot"):
            settings.create_plot = True
        elif opt in ("--midi"):
            settings.create_midi = True
        elif opt in ("--disable_hyphenation"):
            settings.hyphenation = False
        elif opt in ("--disable_separation"):
            settings.use_separated_vocal = False
        elif opt in ("--disable_karaoke"):
            settings.create_karaoke = False
        elif opt in ("--create_audio_chunks"):
            settings.create_audio_chunks = arg
        elif opt in ("--ignore_audio"):
            settings.ignore_audio = True
        elif opt in ("--force_cpu"):
            settings.force_cpu = True
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif opt in ("--force_whisper_cpu"):
            settings.force_whisper_cpu = True
        elif opt in ("--format_version"):
            if arg == FormatVersion.V0_3_0.value:
                settings.format_version = FormatVersion.V0_3_0
            elif arg == FormatVersion.V1_0_0.value:
                settings.format_version = FormatVersion.V1_0_0
            elif arg == FormatVersion.V1_1_0.value:
                settings.format_version = FormatVersion.V1_1_0
            elif arg == FormatVersion.V1_2_0.value:
                settings.format_version = FormatVersion.V1_2_0
            else:
                print(
                    f"{ULTRASINGER_HEAD} {red_highlighted('Error: Format version')} {blue_highlighted(arg)} {red_highlighted('is not supported.')}"
                )
                sys.exit(1)
        elif opt in ("--keep_cache"):
            settings.keep_cache = True
        elif opt in ("--musescore_path"):
            settings.musescore_path = arg
        #Addition of demucs model choice. Work seems to be needed to make sure syntax is same for models. Added error handling for unknown models
        elif opt in ("--demucs"):
            try:
                settings.demucs_model = DemucsModel(arg)
            except ValueError as ve:
                print(f"{ULTRASINGER_HEAD} The model {arg} is not a valid demucs model selection. Please use one of the following models: {blue_highlighted(', '.join([m.value for m in DemucsModel]))}")
                sys.exit()
        elif opt in ("--cookiefile"):
            settings.cookiefile = arg
        elif opt in ("--interactive"):
            settings.interactive_mode = True
        elif opt in ("--disable_quantization"):
            settings.quantize_to_key = False
        elif opt in ("--disable_vocal_center"):
            settings.vocal_center_correction = False
        elif opt in ("--disable_onset_correction"):
            settings.onset_correction = False
        elif opt in ("--syllable_split"):
            settings.syllable_split = True
        elif opt in ("--vocal_gap_fill"):
            settings.vocal_gap_fill = True
        elif opt in ("--pitch_change_split"):
            settings.pitch_change_split = True
        elif opt in ("--disable_lyrics_lookup"):
            settings.lyrics_lookup = False
        elif opt in ("--disable_reference_lyrics"):
            settings.disable_reference_lyrics = True
        elif opt in ("--ffmpeg"):
            settings.user_ffmpeg_path = arg
        elif opt in ("--denoise_nr"):
            val = float(arg)
            if not (0.01 <= val <= 97):
                print(f"Error: --denoise_nr must be between 0.01 and 97, got {val}")
                sys.exit(1)
            settings.denoise_noise_reduction = val
        elif opt in ("--denoise_nf"):
            val = float(arg)
            if not (-80 <= val <= -20):
                print(f"Error: --denoise_nf must be between -80 and -20, got {val}")
                sys.exit(1)
            settings.denoise_noise_floor = val
        elif opt in ("--disable_denoise_track_noise"):
            settings.denoise_track_noise = False
        elif opt in ("--keep_audio_in_video"):
            settings.keep_audio_in_video = True
        elif opt in ("--write_settings_info"):
            settings.write_settings_info = True
        elif opt in ("--llm_correct"):
            settings.llm_correct_lyrics = True
        elif opt in ("--llm_api_base_url"):
            settings.llm_api_base_url = arg
        elif opt in ("--llm_api_key"):
            settings.llm_api_key = arg
        elif opt in ("--llm_model"):
            settings.llm_model = arg
        elif opt in ("--llm_no_retry"):
            settings.llm_retry_on_rate_limit = False
        elif opt in ("--llm_retry_wait"):
            settings.llm_retry_wait = int(arg)
        elif opt in ("--llm_retry_max"):
            settings.llm_retry_max = int(arg)
        elif opt in ("--youtube_url"):
            settings.youtube_url = arg
        elif opt in ("--refine_from_vocal"):
            settings.refine_from_vocal = True
        elif opt in ("--disable_refine"):
            settings.refine_from_vocal = False
        elif opt in ("--disable_refine_pitch"):
            settings.refine_pitch = False
        elif opt in ("--disable_refine_timing"):
            settings.refine_timing = False
        elif opt in ("--refine_hit_ratio"):
            settings.refine_hit_ratio = float(arg)
        elif opt in ("--refine_timing_threshold"):
            settings.refine_timing_threshold = float(arg)
    if settings.output_folder_path == "":
        if settings.input_file_path.startswith("https:"):
            dirname = os.getcwd()
        else:
            dirname = os.path.dirname(settings.input_file_path)
        settings.output_folder_path = os.path.join(dirname, "output")

    return settings


#For convenience, made True/False options into noargs
def arg_options():
    short = "hi:o:amv:"
    long = [
        "ifile=",
        "ofile=",
        "bpm=",
        "octave=",
        "demucs=",
        "whisper=",
        "whisper_align_model=",
        "whisper_batch_size=",
        "whisper_compute_type=",
        "vad_onset=",
        "vad_offset=",
        "no_speech_threshold=",
        "language=",
        "plot",
        "midi",
        "disable_hyphenation",
        "disable_separation",
        "disable_karaoke",
        "create_audio_chunks",
        "ignore_audio",
        "force_cpu",
        "force_whisper_cpu",
        "format_version=",
        "keep_cache",
        "musescore_path=",
        "keep_numbers",
        "disable_quantization",
        "disable_vocal_center",
        "disable_onset_correction",
        "syllable_split",
        "vocal_gap_fill",
        "pitch_change_split",
        "disable_lyrics_lookup",
        "disable_reference_lyrics",
        "interactive",
        "cookiefile=",
        "ffmpeg=",
        "denoise_nr=",
        "denoise_nf=",
        "disable_denoise_track_noise",
        "keep_audio_in_video",
        "write_settings_info",
        "llm_correct",
        "llm_api_base_url=",
        "llm_api_key=",
        "llm_model=",
        "llm_no_retry",
        "llm_retry_wait=",
        "llm_retry_max=",
        "youtube_url=",
        "refine_from_vocal",
        "disable_refine",
        "disable_refine_pitch",
        "disable_refine_timing",
        "refine_hit_ratio=",
        "refine_timing_threshold=",
    ]
    return long, short

if __name__ == "__main__":
    main(sys.argv[1:])
