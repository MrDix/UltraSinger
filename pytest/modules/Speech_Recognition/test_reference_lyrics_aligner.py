"""Tests for reference_lyrics_aligner module."""

from types import SimpleNamespace

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from modules.Speech_Recognition.lyrics_alignment import lyrics_tokens
from modules.Speech_Recognition.reference_lyrics_aligner import (
    parse_lrc_synced_lyrics,
    _align_model_log_probs,
    _compute_note_for_word,
    _split_word_at_pitch_changes,
    _trim_word_to_voiced,
    create_midi_segments_from_reference_lyrics,
    align_lyrics_to_audio,
)
from modules.Pitcher.pitched_data import PitchedData
from modules.Midi.MidiSegment import MidiSegment


# ---------------------------------------------------------------------------
# LRC parsing
# ---------------------------------------------------------------------------


class TestParseLrcSyncedLyrics:
    """Tests for parse_lrc_synced_lyrics."""

    def test_basic_parsing(self):
        lrc = "[00:17.21] Hello world\n[00:23.25] Second line"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 2
        assert segs[0]["text"] == "Hello world"
        assert segs[0]["start"] == pytest.approx(17.21)
        assert segs[0]["end"] == pytest.approx(23.25)
        assert segs[1]["text"] == "Second line"
        assert segs[1]["start"] == pytest.approx(23.25)

    def test_last_segment_end(self):
        lrc = "[01:00.00] Only line"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 1
        assert segs[0]["end"] == pytest.approx(70.0)  # start + 10s (fallback)

    def test_last_segment_end_with_audio_duration(self):
        lrc = "[01:00.00] Only line"
        segs = parse_lrc_synced_lyrics(lrc, audio_duration=180.0)
        assert len(segs) == 1
        assert segs[0]["end"] == pytest.approx(180.0)  # actual audio duration

    def test_empty_lines_skipped(self):
        lrc = "[00:10.00] First\n[00:15.00] \n[00:20.00] Third"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 2
        assert segs[0]["text"] == "First"
        assert segs[1]["text"] == "Third"
        # First segment ends at third's start (empty line skipped)
        assert segs[0]["end"] == pytest.approx(20.0)

    def test_empty_input(self):
        assert parse_lrc_synced_lyrics("") == []
        assert parse_lrc_synced_lyrics("   ") == []

    def test_invalid_lines_ignored(self):
        lrc = "Not a valid line\n[00:10.00] Valid line\nAlso invalid"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 1
        assert segs[0]["text"] == "Valid line"

    def test_minute_conversion(self):
        lrc = "[02:30.50] Two minutes thirty"
        segs = parse_lrc_synced_lyrics(lrc)
        assert segs[0]["start"] == pytest.approx(150.50)

    def test_windows_line_endings(self):
        lrc = "[00:10.00] First\r\n[00:20.00] Second"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 2

    def test_sorted_output(self):
        # Even if input is out of order
        lrc = "[00:20.00] Second\n[00:10.00] First"
        segs = parse_lrc_synced_lyrics(lrc)
        assert segs[0]["start"] < segs[1]["start"]
        assert segs[0]["text"] == "First"

    def test_melismatic_text(self):
        lrc = "[01:55.98] Oh-oh-oh-oh-oh\n[02:00.03] Ooh, ooh, ooh-ooh-ooh"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 2
        assert segs[0]["text"] == "Oh-oh-oh-oh-oh"
        assert segs[1]["text"] == "Ooh, ooh, ooh-ooh-ooh"

    def test_line_with_several_times_is_sung_at_each(self):
        lrc = "[00:10.00][01:30.00] Chorus line\n[00:20.00] Verse line\n[01:40.00] Last line"
        segs = parse_lrc_synced_lyrics(lrc)
        assert [(s["start"], s["text"]) for s in segs] == [
            (10.0, "Chorus line"), (20.0, "Verse line"), (90.0, "Chorus line"), (100.0, "Last line")]
        assert segs[1]["end"] == pytest.approx(90.0)

    def test_word_times_of_enhanced_lrc_are_left_out(self):
        segs = parse_lrc_synced_lyrics("[00:12.00] <00:12.00>Hello <00:12.50>world <00:13.10>")
        assert segs[0]["text"] == "Hello world"

    def test_other_tags_are_not_sung(self):
        lrc = "[ar:Artist]\n[ti:Song Title]\n[length:03:20]\n[00:10.00][Chorus]\n" \
              "[00:12.00] [Verse 2] Hello (oh) world\n[Bridge]\n[00:14.00][Pre-Chorus x2] Again\n" \
              "[00:16.00] [Verse 1: Artist] [offset:+100] Last"
        segs = parse_lrc_synced_lyrics(lrc)
        assert [s["text"] for s in segs] == ["Hello \x00oh world", "Again", "Last"]
        assert segs[0]["start"] == pytest.approx(12.0)

    def test_backing_vocal_at_the_start_of_a_line_is_sung(self):
        lrc = "[00:12.00] [oh yeah] Hello world\n[00:14.00][Hey] come on\n[00:16.00] [Chorus] [ooh] la"
        segs = parse_lrc_synced_lyrics(lrc)
        assert [s["text"] for s in segs] == [
            "\x00oh \x00yeah Hello world", "\x00Hey come on", "\x00ooh la"]

    def test_labels_after_a_backing_vocal_are_not_sung(self):
        lrc = "[00:12.00] [oh yeah] [Chorus] text\n[00:14.00] (la) [Intro]\n" \
              "[00:16.00] [oh] [ar:Artist] [Verse 2] (ooh) go"
        segs = parse_lrc_synced_lyrics(lrc)
        assert [s["text"] for s in segs] == [
            "\x00oh \x00yeah text", "\x00la", "\x00oh \x00ooh go"]


# ---------------------------------------------------------------------------
# Pitch assignment
# ---------------------------------------------------------------------------


def _make_pitched_data(duration_s=10.0, base_freq=440.0, confidence=0.95):
    """Create synthetic PitchedData for testing."""
    # SwiftF0: 16kHz, hop=256 → ~62.5 fps → 16ms per frame
    fps = 62.5
    n_frames = int(duration_s * fps)
    times = [i / fps for i in range(n_frames)]
    frequencies = [base_freq] * n_frames
    confidences = [confidence] * n_frames
    return PitchedData(times=times, frequencies=frequencies, confidence=confidences)


class TestComputeNoteForWord:
    """Tests for _compute_note_for_word."""

    def test_basic_pitch(self):
        pd = _make_pitched_data(base_freq=440.0)  # A4
        note = _compute_note_for_word(1.0, 2.0, pd)
        assert note == "A4"

    def test_c4_pitch(self):
        pd = _make_pitched_data(base_freq=261.63)  # C4
        note = _compute_note_for_word(1.0, 2.0, pd)
        assert note in {"C4", "B3"}  # Exact match, rounding may yield B3

    def test_silent_region_fallback(self):
        pd = _make_pitched_data(base_freq=0.0, confidence=0.0)
        note = _compute_note_for_word(1.0, 2.0, pd)
        assert note == "C4"  # fallback

    def test_key_quantization(self):
        pd = _make_pitched_data(base_freq=440.0)  # A4
        # Allow only C major notes (no A)
        allowed = {"C", "D", "E", "F", "G", "B"}
        note = _compute_note_for_word(1.0, 2.0, pd, allowed_notes=allowed)
        # Should be quantized to nearest allowed note (no accidentals)
        pitch_class = note.rstrip("0123456789")
        assert pitch_class in allowed


# ---------------------------------------------------------------------------
# Melisma splitting
# ---------------------------------------------------------------------------


class TestSplitWordAtPitchChanges:
    """Tests for _split_word_at_pitch_changes."""

    def test_stable_pitch_single_note(self):
        pd = _make_pitched_data(base_freq=440.0)
        result = _split_word_at_pitch_changes("hello", 1.0, 2.0, pd)
        assert len(result) == 1
        assert result[0].word == "hello"

    def test_pitch_change_creates_split(self):
        """A large pitch change mid-word should split into 2+ segments."""
        fps = 62.5
        n_frames = int(5.0 * fps)  # 5 seconds
        times = [i / fps for i in range(n_frames)]
        frequencies = []
        confidences = [0.95] * n_frames
        mid = n_frames // 2
        for i in range(n_frames):
            if i < mid:
                frequencies.append(440.0)  # A4
            else:
                frequencies.append(880.0)  # A5 (12 semitones up)
        pd = PitchedData(times=times, frequencies=frequencies, confidence=confidences)

        result = _split_word_at_pitch_changes("oooh", 0.5, 4.5, pd, threshold_st=2.0)
        assert len(result) >= 2
        assert result[0].word == "oooh"
        assert result[1].word == "~ "

    def test_short_word_no_split(self):
        pd = _make_pitched_data(base_freq=440.0)
        # Very short word - not enough frames to split
        result = _split_word_at_pitch_changes("a", 1.0, 1.05, pd)
        assert len(result) == 1

    def test_silent_word(self):
        pd = _make_pitched_data(base_freq=0.0, confidence=0.0)
        result = _split_word_at_pitch_changes("hmm", 1.0, 2.0, pd)
        assert len(result) == 1

    def test_boundaries_preserved(self):
        """First segment should start at word start, last should end at word end."""
        fps = 62.5
        n_frames = int(5.0 * fps)
        times = [i / fps for i in range(n_frames)]
        frequencies = []
        for i in range(n_frames):
            if i < n_frames // 2:
                frequencies.append(440.0)
            else:
                frequencies.append(880.0)
        pd = PitchedData(times=times, frequencies=frequencies, confidence=[0.95] * n_frames)

        result = _split_word_at_pitch_changes("test", 0.5, 4.5, pd)
        assert result[0].start == pytest.approx(0.5)
        assert result[-1].end == pytest.approx(4.5)

    def test_word_kept_when_the_first_part_is_too_short(self):
        """A first pitch part shorter than min_note_ms is skipped, but the word is not lost."""
        fps = 62.5  # 16 ms frames
        times = [i / fps for i in range(64)]  # about one second
        # 5 frames (64 ms from first to last frame, under 80 ms) at A4, then F#4 and D#4
        frequencies = [440.0] * 5 + [370.0] * 26 + [311.0] * 33
        pd = PitchedData(times=times, frequencies=frequencies, confidence=[0.95] * 64)

        result = _split_word_at_pitch_changes("glide", 0.0, times[-1], pd)

        assert len(result) == 2
        assert [s.word for s in result] == ["glide", "~ "]
        assert result[0].start == pytest.approx(0.0)  # still starts at the word start


# ---------------------------------------------------------------------------
# Voiced-region trimming
# ---------------------------------------------------------------------------


class TestTrimWordToVoiced:
    """Tests for _trim_word_to_voiced."""

    def _word_with_trailing_silence(self):
        """A word whose second half is silence, so trimming kicks in."""
        fps = 62.5
        n_frames = int(4.0 * fps)
        times = [i / fps for i in range(n_frames)]
        freqs, confs = [], []
        for t in times:
            if 1.0 <= t <= 2.0:
                freqs.append(440.0)
                confs.append(0.95)
            else:
                freqs.append(0.0)
                confs.append(0.0)
        pd = PitchedData(times=times, frequencies=freqs, confidence=confs)
        # CTC stretched the word until 3.5s although singing ends at 2.0s
        word = {"word": "night", "start": 1.0, "end": 3.5,
                "backing": True, "line_end": True}
        return word, pd

    def test_trims_trailing_silence(self):
        word, pd = self._word_with_trailing_silence()
        trimmed = _trim_word_to_voiced(word, pd)
        assert trimmed["end"] < word["end"]

    def test_preserves_flags_when_trimming(self):
        """Regression: trimming used to rebuild the dict and drop the
        line_end/backing flags. CTC stretches precisely the last word of
        each LRC line, so this silently killed nearly every LRCLIB
        linebreak in the output file."""
        word, pd = self._word_with_trailing_silence()
        trimmed = _trim_word_to_voiced(word, pd)
        assert trimmed["end"] < word["end"]  # trimming actually happened
        assert trimmed["line_end"] is True
        assert trimmed["backing"] is True
        assert trimmed["word"] == "night"


# ---------------------------------------------------------------------------
# Integration (with mocked WhisperX)
# ---------------------------------------------------------------------------


class TestCreateMidiSegmentsFromReferenceLyrics:
    """Tests for create_midi_segments_from_reference_lyrics (mocked alignment)."""

    @patch("modules.Speech_Recognition.reference_lyrics_aligner.librosa")
    @patch("modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_basic_pipeline(self, mock_align, mock_librosa):
        """Test full pipeline with mocked alignment."""
        mock_librosa.get_duration.return_value = 5.0
        mock_align.return_value = [
            {"word": "Hello", "start": 1.0, "end": 1.5},
            {"word": "world", "start": 1.6, "end": 2.0},
        ]

        pd = _make_pitched_data(base_freq=440.0, duration_s=5.0)
        lrc = "[00:01.00] Hello world"

        result = create_midi_segments_from_reference_lyrics(
            synced_lyrics=lrc,
            audio_path="/fake/audio.wav",
            language="en",
            pitched_data=pd,
            melisma_split=False,
        )

        assert len(result) == 2
        assert result[0].word == "Hello "  # trailing space for word boundary
        assert result[0].start == pytest.approx(1.0)
        assert result[1].word == "world"  # last word has no trailing space

    @patch("modules.Speech_Recognition.reference_lyrics_aligner.librosa")
    @patch("modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_empty_alignment_returns_empty(self, mock_align, mock_librosa):
        mock_librosa.get_duration.return_value = 10.0
        mock_align.return_value = []
        pd = _make_pitched_data()
        result = create_midi_segments_from_reference_lyrics(
            "[00:01.00] Test", "/fake/audio.wav", "en", pd
        )
        assert result == []

    @patch("modules.Speech_Recognition.reference_lyrics_aligner.librosa")
    def test_empty_lrc_returns_empty(self, mock_librosa):
        mock_librosa.get_duration.return_value = 10.0
        pd = _make_pitched_data()
        result = create_midi_segments_from_reference_lyrics(
            "", "/fake/audio.wav", "en", pd
        )
        assert result == []

    @patch("modules.Speech_Recognition.reference_lyrics_aligner.librosa")
    @patch("modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_melisma_split_enabled(self, mock_align, mock_librosa):
        """With melisma split and a pitch change, should produce extra notes."""
        mock_librosa.get_duration.return_value = 5.0
        mock_align.return_value = [
            {"word": "oooh", "start": 0.5, "end": 4.5},
        ]

        # Create pitched data with a clear pitch change
        fps = 62.5
        n = int(5.0 * fps)
        times = [i / fps for i in range(n)]
        freqs = [440.0 if i < n // 2 else 880.0 for i in range(n)]
        pd = PitchedData(times=times, frequencies=freqs, confidence=[0.95] * n)

        result = create_midi_segments_from_reference_lyrics(
            "[00:00.50] oooh",
            "/fake/audio.wav",
            "en",
            pd,
            melisma_split=True,
        )

        assert len(result) >= 2
        assert result[0].word == "oooh"
        # Additional segments should be continuations
        for seg in result[1:]:
            assert seg.word == "~ "

    @patch("modules.Speech_Recognition.reference_lyrics_aligner.librosa")
    @patch("modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_no_melisma_split(self, mock_align, mock_librosa):
        """Without melisma split, even pitch changes produce single note."""
        mock_librosa.get_duration.return_value = 5.0
        mock_align.return_value = [
            {"word": "oooh", "start": 0.5, "end": 4.5},
        ]

        fps = 62.5
        n = int(5.0 * fps)
        times = [i / fps for i in range(n)]
        freqs = [440.0 if i < n // 2 else 880.0 for i in range(n)]
        pd = PitchedData(times=times, frequencies=freqs, confidence=[0.95] * n)

        result = create_midi_segments_from_reference_lyrics(
            "[00:00.50] oooh",
            "/fake/audio.wav",
            "en",
            pd,
            melisma_split=False,
        )

        assert len(result) == 1


# ---------------------------------------------------------------------------
# Forced alignment (align_lyrics_to_audio) with a fake model
# ---------------------------------------------------------------------------

LETTERS = "abcdefghijklmnopqrstuvwxyz'"
DICTIONARY = {"<pad>": 0, "|": 1, **{c: i + 2 for i, c in enumerate(LETTERS)}}
METADATA = {"language": "en", "dictionary": DICTIONARY, "type": "torchaudio"}


class FakeAlignModel:
    """A torchaudio-like CTC model for audio that holds its sample indices:
    the frames of any part of the audio are rows of one emission of the song."""

    def __init__(self, log_probs, out_of_memory_from=None, error=None):
        self.log_probs = log_probs
        self.out_of_memory_from = out_of_memory_from  # inputs this long run out of memory
        self.error = error or torch.cuda.OutOfMemoryError("CUDA out of memory.")
        self.inputs = []

    def __call__(self, waveform, lengths=None):
        samples = waveform[0].numpy()
        self.inputs.append(len(samples))
        if self.out_of_memory_from is not None and len(samples) >= self.out_of_memory_from:
            raise self.error
        n = max(1, (len(samples) - 400) // 320 + 1)
        frames = (samples[np.arange(n) * 320] / 320).astype(int)
        return torch.from_numpy(self.log_probs[frames]).unsqueeze(0), None


def fake_song(segments, sung_at, seconds, **model_args):
    """A fake model and audio in which the words of segment i are sung from
    sung_at[i] seconds on (None: not at all)."""
    n_samples = int(seconds * 16000)
    n_frames = (n_samples - 400) // 320 + 1
    frame_s = n_samples / 16000 / n_frames  # as whisperx times the frames
    lines = [[w.replace("\x00", "") for w in seg["text"].split()] for seg in segments]
    tokens = lyrics_tokens(lines, DICTIONARY, "en")
    em = np.full((n_frames, len(DICTIONARY)), -10.0, dtype=np.float32)
    em[:, 0] = -0.01
    for i, t in enumerate(sung_at):
        if t is None:
            continue
        for n, k in enumerate(np.flatnonzero(tokens.word_line[tokens.word] == i)):
            f = round(t / frame_s) + 2 * n
            em[f, 0], em[f, tokens.ids[k]] = -10.0, -0.01
    em -= np.logaddexp.reduce(em, axis=1, keepdims=True)  # log-probabilities
    return FakeAlignModel(em, **model_args), np.arange(n_samples, dtype=np.float32)


def lrc_segments(n_lines):
    """LRC lines 3 s apart, two words of a letter of their own each."""
    return [{"text": f"{LETTERS[i] * 2} {LETTERS[i] * 3}", "start": 2.0 + 3 * i, "end": 5.0 + 3 * i}
            for i in range(n_lines)]


def align_with(model, audio, segments, language="en", **kwargs):
    with patch("whisperx.load_align_model", return_value=(model, METADATA)) as load_model, \
            patch("whisperx.load_audio", return_value=audio):
        words = align_lyrics_to_audio(segments, "/fake/audio.wav", language, device="cpu", **kwargs)
    return words, load_model


class TestAlignLyricsToAudio:

    def test_words_with_their_flags(self, capsys):
        segments = lrc_segments(20)
        segments[0]["text"] = "aa \x00aaa"
        model, audio = fake_song(segments, [s["start"] for s in segments], 64.0)

        words, _ = align_with(model, audio, segments)

        assert len(words) == 40
        assert all(set(w) == {"word", "start", "end", "backing", "line_end"} for w in words)
        assert all(isinstance(w["start"], float) and isinstance(w["end"], float) for w in words)
        assert [(w["word"], w["backing"], w["line_end"]) for w in words[:3]] == [
            ("aa", False, False), ("aaa", True, True), ("bb", False, False)]
        # frames of about 20 ms: "aa" up to the space after it, "aaa" from 3 frames on
        assert (words[0]["start"], words[0]["end"]) == pytest.approx((2.0, 2.08), abs=0.011)
        assert (words[1]["start"], words[1]["end"]) == pytest.approx((2.12, 2.24), abs=0.011)
        assert [w["line_end"] for w in words] == [False, True] * 20
        assert [w["start"] for w in words[::2]] == pytest.approx([s["start"] for s in segments], abs=0.011)
        out = capsys.readouterr().out
        assert "Aligning" in out and "40" in out
        assert "Lyrics alignment:" in out and "of" in out and "(whole song, free end)" in out
        assert "Reference alignment:" in out

    def test_pieces_first_then_the_whole_song(self):
        segments = lrc_segments(20)
        model, audio = fake_song(segments, [s["start"] for s in segments], 70.0)
        align_with(model, audio, segments)
        # 30-s pieces with 5 s of context on both sides, then the whole 70 s
        assert model.inputs == [35 * 16000, 40 * 16000, 15 * 16000, 70 * 16000]

    def test_words_of_lines_the_recording_does_not_have_are_left_out(self, capsys):
        segments = lrc_segments(22)
        sung = [s["start"] if i < 20 else None for i, s in enumerate(segments)]
        model, audio = fake_song(segments, sung, segments[19]["start"] + 4.0)

        words, _ = align_with(model, audio, segments)

        assert len(words) == 40
        assert {w["word"] for w in words}.isdisjoint({"uu", "uuu", "vv", "vvv"})
        assert words[-1]["word"] == "ttt" and words[-1]["line_end"]
        assert "Left out" in capsys.readouterr().out

    def test_plain_lyrics_as_one_segment(self, capsys):
        segments = [{"text": "aa aaa bb bbb", "start": 0.0, "end": 10.0}]
        model, audio = fake_song(segments, [1.0], 10.0)

        words, _ = align_with(model, audio, segments)

        # the whole song only, no line check
        assert model.inputs == [len(audio)]
        # tokens 2 frames apart: "a a | a a a | b b | b b b"
        assert [w["start"] for w in words] == pytest.approx([1.0, 1.12, 1.28, 1.40], abs=0.011)
        assert [w["line_end"] for w in words] == [False, False, False, True]
        assert "Lyrics alignment:" not in capsys.readouterr().out

    def test_whole_song_out_of_memory_uses_the_pieces(self, capsys):
        segments = lrc_segments(20)
        model, audio = fake_song(segments, [s["start"] for s in segments], 64.0,
                                 out_of_memory_from=64 * 16000)

        words, _ = align_with(model, audio, segments)

        assert len(words) == 40
        assert [w["start"] for w in words[::2]] == pytest.approx([s["start"] for s in segments], abs=0.011)
        out = capsys.readouterr().out
        assert "Not enough memory" in out
        assert "(30-s pieces," in out

    def test_out_of_memory_runtime_error_of_plain_lyrics(self, capsys):
        segments = [{"text": "aa aaa bb bbb", "start": 0.0, "end": 50.0}]
        model, audio = fake_song(segments, [41.0], 50.0, out_of_memory_from=50 * 16000,
                                 error=RuntimeError("CUDA error: out of memory"))

        words, _ = align_with(model, audio, segments)

        # the whole song first, then the pieces
        assert model.inputs == [len(audio), 35 * 16000, 25 * 16000]
        assert [w["start"] for w in words] == pytest.approx([41.0, 41.12, 41.28, 41.40], abs=0.011)
        assert "Not enough memory" in capsys.readouterr().out

    def test_cpu_out_of_memory_uses_the_pieces(self, capsys):
        # the CPU allocator reports "not enough memory"
        segments = lrc_segments(20)
        model, audio = fake_song(segments, [s["start"] for s in segments], 64.0, out_of_memory_from=64 * 16000,
                                 error=RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate"))

        words, _ = align_with(model, audio, segments)

        assert len(words) == 40
        assert "(30-s pieces," in capsys.readouterr().out

    def test_other_runtime_errors_are_raised(self):
        segments = lrc_segments(20)
        model, audio = fake_song(segments, [s["start"] for s in segments], 64.0,
                                 out_of_memory_from=64 * 16000, error=RuntimeError("shape mismatch"))
        with pytest.raises(RuntimeError, match="shape mismatch"):
            align_with(model, audio, segments)

    def test_model_with_another_hop_aligns_with_the_whole_song(self):
        segments = lrc_segments(20)

        class Model:
            def __call__(self, waveform, lengths=None):
                return torch.zeros(1, waveform.shape[-1] // 160, len(DICTIONARY)), None

        with patch("modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics",
                   return_value=None) as align:
            align_with(Model(), np.zeros(64 * 16000, dtype=np.float32), segments)
        assert [label for label, _ in align.call_args.args[1]] == ["whole song"]

    def test_custom_align_model_passed_through(self):
        segments = lrc_segments(20)
        model, audio = fake_song(segments, [s["start"] for s in segments], 64.0)

        words, load_model = align_with(model, audio, segments, align_model_name="org/custom-model")

        load_model.assert_called_once_with(language_code="en", device="cpu", model_name="org/custom-model")
        assert len(words) == 40

    def test_no_words_without_alignable_characters(self):
        segments = [{"text": "|||", "start": 0.0, "end": 5.0}]
        model, audio = fake_song([], [], 5.0)
        metadata = {**METADATA, "dictionary": {"<pad>": 0, "a": 1}}
        with patch("whisperx.load_align_model", return_value=(model, metadata)), \
                patch("whisperx.load_audio", return_value=audio):
            assert align_lyrics_to_audio(segments, "/fake/audio.wav", "en") == []
        assert model.inputs == []


class TestAlignModelLogProbs:

    def test_short_input_padded_as_whisperx(self):
        calls = []

        class TorchaudioModel:
            def __call__(self, waveform, lengths=None):
                calls.append((waveform.shape[-1], None if lengths is None else lengths.tolist()))
                return torch.zeros(1, 1, 5), None

        class HuggingfaceModel:
            def __call__(self, waveform):
                calls.append((waveform.shape[-1],))
                return SimpleNamespace(logits=torch.zeros(1, 1, 5))

        out = _align_model_log_probs(TorchaudioModel(), "torchaudio", np.zeros(100, np.float32), "cpu")
        _align_model_log_probs(HuggingfaceModel(), "huggingface", np.zeros(100, np.float32), "cpu")
        _align_model_log_probs(TorchaudioModel(), "torchaudio", np.zeros(1000, np.float32), "cpu")

        assert calls == [(400, [100]), (400,), (1000, None)]
        assert out.shape == (1, 5)
        assert out == pytest.approx(np.full((1, 5), -np.log(5)))

    def test_unknown_model_type(self):
        with pytest.raises(NotImplementedError):
            _align_model_log_probs(MagicMock(), "other", np.zeros(1000, np.float32), "cpu")


# ---------------------------------------------------------------------------
# Language-aware alignment model selection (align_lyrics_to_audio)
# ---------------------------------------------------------------------------


class TestAlignLyricsToAudioLanguageModel:
    """Tests that align_lyrics_to_audio threads the resolved language into
    whisperx.load_align_model, and degrades gracefully (instead of crashing)
    when WhisperX has no default wav2vec2 model for that language."""

    def test_language_passed_to_align_model(self):
        segments = [{"text": "hallo welt", "start": 0.0, "end": 5.0}]
        model, audio = fake_song(segments, [0.5], 5.0)

        words, load_model = align_with(model, audio, segments, language="de")

        load_model.assert_called_once_with(language_code="de", device="cpu", model_name=None)
        assert [w["word"] for w in words] == ["hallo", "welt"]

    @patch("whisperx.load_audio")
    def test_only_the_model_load_may_unpickle(self, mock_load_audio, tmp_path):
        """The alignment model loads as before PyTorch 2.6; afterwards torch.load
        refuses to unpickle objects again (weights_only)."""
        import pickle
        from fractions import Fraction

        path = tmp_path / "model.bin"
        torch.save({"value": Fraction(1, 3)}, path)  # weights_only=True refuses a Fraction
        segments = [{"text": "hello", "start": 0.0, "end": 5.0}]
        model, audio = fake_song(segments, [0.5], 5.0)
        mock_load_audio.return_value = audio
        loaded = []

        def load_align_model(*args, **kwargs):
            loaded.append(torch.load(path))
            return model, METADATA

        with patch("whisperx.load_align_model", new=load_align_model):
            align_lyrics_to_audio(segments, "/fake/audio.wav", "en", device="cpu")

        assert loaded == [{"value": Fraction(1, 3)}]
        assert torch.load is torch.serialization.load
        with pytest.raises(pickle.UnpicklingError):
            torch.load(path)

    @patch("whisperx.load_audio")
    @patch("whisperx.load_align_model")
    def test_unsupported_language_falls_back_to_english(self, mock_load_align_model, mock_load_audio):
        segments = [{"text": "hello", "start": 0.0, "end": 5.0}]
        model, audio = fake_song(segments, [0.5], 5.0)
        mock_load_audio.return_value = audio
        mock_load_align_model.side_effect = [
            ValueError("No default align-model for language: xx"),
            (model, METADATA),
        ]

        words = align_lyrics_to_audio(segments, "/fake/audio.wav", "xx", device="cpu")

        assert mock_load_align_model.call_count == 2
        first_call, second_call = mock_load_align_model.call_args_list
        assert first_call.kwargs == {"language_code": "xx", "device": "cpu", "model_name": None}
        assert second_call.kwargs == {"language_code": "en", "device": "cpu", "model_name": None}
        assert len(words) == 1

    @patch("whisperx.load_align_model")
    def test_custom_align_model_error_is_not_swallowed(self, mock_load_align_model):
        """A user-supplied --whisper_align_model that fails must raise --
        we must not silently substitute English for an explicit choice."""
        mock_load_align_model.side_effect = ValueError("bad custom model")

        segments = [{"text": "hello", "start": 0.0, "end": 5.0}]
        with pytest.raises(ValueError):
            align_lyrics_to_audio(
                segments, "/fake/audio.wav", "xx", device="cpu",
                align_model_name="some/custom-model",
            )

        mock_load_align_model.assert_called_once_with(
            language_code="xx", device="cpu", model_name="some/custom-model"
        )

    @patch("whisperx.load_audio")
    def test_model_dropped_although_the_frame_is_kept(self, mock_load_audio):
        """The alignment model is dropped when the alignment is done, also when
        a library keeps the call's frame alive (as a stored traceback of a
        failed optional import does)."""
        import sys
        import weakref

        kept_frames, models = [], []
        segments = lrc_segments(20)
        song_model, audio = fake_song(segments, [s["start"] for s in segments], 64.0)
        mock_load_audio.return_value = audio

        def load_align_model(*args, **kwargs):
            frame = sys._getframe(1)
            while frame.f_code.co_name != "align_lyrics_to_audio":
                frame = frame.f_back
            kept_frames.append(frame)
            model = FakeAlignModel(song_model.log_probs)
            models.append(weakref.ref(model))
            return model, METADATA

        with patch("whisperx.load_align_model", new=load_align_model):
            words = align_lyrics_to_audio(segments, "/fake/audio.wav", "en")

        assert len(words) == 40
        assert kept_frames[0].f_locals["align_model"] is None
        assert models[0]() is None
