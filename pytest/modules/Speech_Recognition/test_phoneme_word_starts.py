"""Tests for phoneme_word_starts."""

import warnings
from unittest.mock import patch

import numpy as np
import pytest

from modules.Midi.MidiSegment import MidiSegment
from modules.Speech_Recognition import phoneme_word_starts as pws
from modules.Speech_Recognition.phoneme_word_starts import (
    PhonemeAligner,
    correct_word_starts,
    earlier_starts,
    lines_of_words,
    move_word_starts,
)


def seg(word, start, end, note="C4", line_break=False, note_type=":"):
    return MidiSegment(note, start, end, word, note_type=note_type, line_break_after=line_break)


def held_note_line():
    """ "your" held from 1.0 to 4.0 in three notes, then "eyes", then a new line."""
    return [
        seg("your ", 1.0, 2.0, "C4"),
        seg("~", 2.0, 3.0, "D4"),
        seg("~", 3.0, 4.0, "E4"),
        seg("eyes ", 4.1, 4.5, "E4", line_break=True),
        seg("Next ", 6.0, 6.5),
    ]


class TestLinesOfWords:
    def test_words_and_lines(self):
        lines = lines_of_words(held_note_line())
        assert [[(w.first, w.last, w.text) for w in line] for line in lines] == [
            [(0, 2, "your"), (3, 3, "eyes")],
            [(4, 4, "Next")],
        ]

    def test_text_less_note_continues_the_word(self):
        lines = lines_of_words([seg("Hey ", 0, 1), seg("", 1, 2), seg("you", 2, 3)])
        assert [(w.first, w.last) for w in lines[0]] == [(0, 1), (2, 2)]


class TestMoveWordStarts:
    def test_notes_after_the_new_start_go_to_the_word(self):
        out = move_word_starts(held_note_line(), {3: 1.5})
        assert [(s.word, s.start, s.end, s.note) for s in out] == [
            ("your ", 1.0, 1.5, "C4"),
            ("eyes ", 1.5, 2.0, "C4"),
            ("~", 2.0, 3.0, "D4"),
            ("~", 3.0, 4.0, "E4"),
            ("~ ", 4.1, 4.5, "E4"),
            ("Next ", 6.0, 6.5, "C4"),
        ]
        # the line still ends after the word
        assert [s.line_break_after for s in out] == [False, False, False, False, True, False]

    def test_new_start_on_a_note_start_splits_nothing(self):
        out = move_word_starts(held_note_line(), {3: 2.0})
        assert [(s.word, s.start, s.end) for s in out[:3]] == [
            ("your ", 1.0, 2.0), ("eyes ", 2.0, 3.0), ("~", 3.0, 4.0)]
        assert len(out) == 5

    def test_moved_notes_take_the_note_type_of_the_word(self):
        segments = held_note_line()
        segments[3] = seg("(oh) ", 4.1, 4.5, note_type="F", line_break=True)
        out = move_word_starts(segments, {3: 3.0})
        assert [s.note_type for s in out[:4]] == [":", ":", "F", "F"]

    def test_continuations_keep_the_word_separator(self):
        segments = [seg("your ", 1.0, 2.0), seg("~ ", 2.0, 3.0), seg("eyes ", 3.2, 3.5), seg("are ", 3.5, 4.0)]
        out = move_word_starts(segments, {2: 1.5})
        assert [s.word for s in out] == ["your ", "eyes ", "~ ", "~ ", "are "]

    def test_start_near_a_note_boundary_takes_it(self):
        out = move_word_starts(held_note_line(), {3: 2.03})
        assert [(s.word, s.start, s.end) for s in out[:3]] == [
            ("your ", 1.0, 2.0), ("eyes ", 2.0, 3.0), ("~", 3.0, 4.0)]

    def test_no_short_note_is_split_off_at_a_boundary_between_notes(self):
        # 2.08 is 0.08 s after the boundary at 2.0: no note of 0.08 s for "your"
        out = move_word_starts(held_note_line(), {3: 2.08})
        assert [(s.word, s.start, s.end) for s in out[:3]] == [
            ("your ", 1.0, 2.0), ("eyes ", 2.0, 3.0), ("~", 3.0, 4.0)]
        out = move_word_starts(held_note_line(), {3: 2.12})
        assert [(s.word, s.start, s.end) for s in out[:3]] == [
            ("your ", 1.0, 2.0), ("~", 2.0, 2.12), ("eyes ", 2.12, 3.0)]

    def test_start_close_to_the_end_of_the_previous_word_ends_it(self):
        # 4.0 - 3.95 is too little to take: "your" ends at 3.95, "eyes" starts there
        out = move_word_starts(held_note_line(), {3: 3.92})
        assert [(s.word, s.start, s.end) for s in out] == [
            ("your ", 1.0, 2.0), ("~", 2.0, 3.0), ("~", 3.0, 3.92), ("eyes ", 3.92, 4.5), ("Next ", 6.0, 6.5)]

    def test_new_start_in_the_pause_moves_the_first_note(self):
        segments = held_note_line()
        out = move_word_starts(segments, {3: 4.07})
        assert [(s.word, s.start, s.end) for s in out] == [
            ("your ", 1.0, 2.0), ("~", 2.0, 3.0), ("~", 3.0, 4.0), ("eyes ", 4.07, 4.5), ("Next ", 6.0, 6.5)]
        assert out[3].line_break_after

    def test_a_start_before_the_previous_word_moves_nothing(self):
        segments = held_note_line()
        assert move_word_starts(segments, {3: 0.5}) == segments

    def test_first_word_of_a_line_is_not_moved(self):
        segments = held_note_line()
        assert move_word_starts(segments, {4: 4.6}) == segments

    def test_a_word_after_a_moved_word(self):
        segments = [seg("a ", 0.0, 2.0), seg("b ", 2.0, 4.0), seg("c ", 4.0, 4.2)]
        out = move_word_starts(segments, {1: 1.0, 2: 3.0})
        assert [(s.word, s.start, s.end) for s in out] == [
            ("a ", 0.0, 1.0), ("b ", 1.0, 2.0), ("~ ", 2.0, 3.0), ("c ", 3.0, 4.0), ("~ ", 4.0, 4.2)]


class FakeAligner:
    """Word starts and phoneme scores given in frames of 20 ms."""

    def __init__(self, first, scores, owner):
        self.first, self.scores, self.owner = first, scores, owner

    def log_probs(self, samples):
        return np.zeros((int(round(len(samples) / pws.SAMPLE_RATE / 0.02)), 4))

    def word_starts(self, log_probs, words):
        return self.first, self.scores, self.owner


def line_of_two(frames=300):
    # line 1.0 .. 4.5 s, audio window 0.5 .. 5.0 s (the line padded by 0.5 s)
    segments = held_note_line()[:4] + [seg("Next ", 6.0, 6.5)]
    samples = np.zeros(int(7.0 * pws.SAMPLE_RATE), dtype=np.float32)
    return segments, samples


def scores_for(frames, eyes_from, own, other):
    """Two phonemes: "your" (column 0) and "eyes" (column 1)."""
    scores = np.full((frames, 2), -10.0, dtype=np.float32)
    scores[:eyes_from, 0] = own
    scores[eyes_from:, 1] = own
    scores[eyes_from:, 0] = other
    return scores


class TestEarlierStarts:
    def test_word_heard_inside_the_held_note_moves(self):
        segments, samples = line_of_two()
        # window starts at 0.5 s: frame 50 = 1.5 s
        aligner = FakeAligner([25, 50], scores_for(225, 50, -0.5, -3.0), [0, 1])
        moves = earlier_starts(aligner, segments, samples)
        assert list(moves) == [3]
        assert moves[3] == pytest.approx(1.5, abs=0.03)

    def test_sounds_that_fit_both_words_move_nothing(self):
        segments, samples = line_of_two()
        aligner = FakeAligner([25, 50], scores_for(225, 50, -0.5, -0.8), [0, 1])
        assert earlier_starts(aligner, segments, samples) == {}

    def test_small_lead_moves_nothing(self):
        segments, samples = line_of_two()
        # heard at 3.9 s, 0.2 s before the start at 4.1 s
        aligner = FakeAligner([25, 170], scores_for(225, 170, -0.5, -3.0), [0, 1])
        assert earlier_starts(aligner, segments, samples) == {}

    def test_lead_beyond_the_longest_moves_nothing(self):
        segments = [seg("a ", 0.0, 8.0), seg("b ", 8.5, 9.0, line_break=True), seg("c ", 12.0, 12.5)]
        samples = np.zeros(int(13.0 * pws.SAMPLE_RATE), dtype=np.float32)
        # window 0 .. 9.5 s: heard at 2.0 s, 6.5 s before the start of "b"
        aligner = FakeAligner([0, 100], scores_for(475, 100, -0.5, -3.0), [0, 1])
        assert earlier_starts(aligner, segments, samples) == {}
        aligner = FakeAligner([0, 200], scores_for(475, 200, -0.5, -3.0), [0, 1])
        assert earlier_starts(aligner, segments, samples) == {1: pytest.approx(4.0, abs=0.03)}

    def test_heard_at_the_start_of_the_previous_word_moves_nothing(self):
        segments, samples = line_of_two()
        aligner = FakeAligner([25, 26], scores_for(225, 26, -0.5, -3.0), [0, 1])
        assert earlier_starts(aligner, segments, samples) == {}


class TestPhonemeAlignment:
    def make(self):
        aligner = PhonemeAligner.__new__(PhonemeAligner)
        aligner.vocab = {"<pad>": 0, "j": 1, "ɔː": 2, "ɹ": 3, "aɪ": 4, "z": 5, "a": 6, "ɑ": 7}
        aligner._phonemes = {"your": ["j", "ɔː", "ɹ"], "eyes": ["aɪ", "z"]}
        aligner._alternatives = {}
        aligner._vowels = {"ɔː", "aɪ", "a", "ɑ"}
        return aligner

    def test_held_vowel_of_the_next_word(self):
        """ "your" sung shortly, then a long /a/: the start of "eyes", whose
        first sound is /aɪ/, not the end of "your"."""
        aligner = self.make()
        frames = ["j"] * 2 + ["ɔː"] * 3 + ["ɹ"] * 2 + ["a"] * 30 + ["z"] * 2
        log_probs = np.full((len(frames), 8), np.log(0.01), dtype=np.float32)
        for f, token in enumerate(frames):
            log_probs[f, aligner.vocab[token]] = np.log(0.9)
        first, scores, owner = aligner.word_starts(log_probs, ["your", "eyes"])
        assert first == [0, 7]
        assert owner == [0, 0, 0, 1, 1]
        assert scores.shape == (len(frames), 5)

    def test_word_without_phonemes(self):
        aligner = self.make()
        aligner._phonemes["--"] = []
        log_probs = np.log(np.full((10, 8), 1 / 8, dtype=np.float32))
        first, _, _ = aligner.word_starts(log_probs, ["--"])
        assert first == [None]


class TestCtcPath:
    @staticmethod
    def path_score(scores, path):
        return float(sum(scores[t, label] for t, label in enumerate(path)))

    def test_labels_in_order_once_each(self):
        rng = np.random.default_rng(1)
        for n_frames, n_labels in ((5, 5), (40, 7), (120, 30)):
            scores = np.log(rng.dirichlet(np.ones(n_labels + 1), size=n_frames)).astype(np.float32)
            path = pws.ctc_path(scores)
            labels = [label for k, label in enumerate(path) if label and (k == 0 or path[k - 1] != label)]
            assert labels == list(range(1, n_labels + 1))

    def test_best_path_as_the_torchaudio_one(self):
        torch = pytest.importorskip("torch")
        functional = pytest.importorskip("torchaudio.functional")
        if not hasattr(functional, "forced_align"):
            pytest.skip("torchaudio without forced_align")
        rng = np.random.default_rng(2)
        for n_frames, n_labels in ((30, 6), (200, 25)):
            scores = np.log(rng.dirichlet(np.ones(n_labels + 1), size=n_frames)).astype(np.float32)
            ours = pws.ctc_path(scores)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # deprecated in torchaudio 2.8
                theirs, _ = functional.forced_align(torch.from_numpy(scores)[None],
                                                    torch.arange(1, n_labels + 1, dtype=torch.int32)[None], blank=0)
            assert self.path_score(scores, ours) == pytest.approx(
                self.path_score(scores, theirs[0].tolist()), abs=1e-3)


class TestPhonemes:
    def test_language_switch_marks_are_no_phonemes(self):
        aligner = PhonemeAligner.__new__(PhonemeAligner)
        aligner.vocab = {"<pad>": 0, "b": 1, "eɪ": 2, "i": 3, "e": 4, "n": 5, "d": 6}
        aligner._phonemes = {}

        class Backend:
            def phonemize(self, texts, separator, strip):
                return ["(en) b eɪ b i (de)"]

        aligner.backend = Backend()
        pytest.importorskip("phonemizer")
        assert aligner.phonemes("baby ") == ["b", "eɪ", "b", "i"]


class TestCorrectWordStarts:
    def test_language_without_phonemes_keeps_the_segments(self):
        segments = held_note_line()
        with patch.object(pws, "espeak_voice", return_value=None):
            out, result = correct_word_starts(segments, "vocals.wav", "xx")
        assert out is segments
        assert result == "skipped, no phonemes for language xx"

    def test_failure_keeps_the_segments(self):
        segments = held_note_line()
        with patch.object(pws, "espeak_voice", return_value="en-us"), \
                patch("librosa.load", side_effect=OSError("no such file")):
            out, result = correct_word_starts(segments, "vocals.wav", "en")
        assert out is segments
        assert result.startswith("skipped after an error")

    def test_moves_from_the_aligner(self):
        segments = held_note_line()
        with patch.object(pws, "espeak_voice", return_value="en-us"), \
                patch("librosa.load", return_value=(np.zeros(16000), 16000)), \
                patch.object(pws, "PhonemeAligner"), \
                patch.object(pws, "earlier_starts", return_value={3: 1.5}):
            out, result = correct_word_starts(segments, "vocals.wav", "en")
        assert [s.word for s in out[:2]] == ["your ", "eyes "]
        assert result == "1 word starts earlier"


class TestEspeakVoice:
    def test_voices(self):
        pytest.importorskip("phonemizer")
        pytest.importorskip("espeakng_loader")
        assert pws.espeak_voice("en") == "en-us"
        assert pws.espeak_voice("de") == "de"
        assert pws.espeak_voice("xx") is None
        assert pws.espeak_voice(None) is None
