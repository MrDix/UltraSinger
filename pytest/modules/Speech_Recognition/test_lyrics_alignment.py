"""Tests for lyrics_alignment (synthetic emissions, no model)."""

from unittest.mock import patch

import numpy as np
import pytest

from modules.Speech_Recognition import lyrics_alignment
from modules.Speech_Recognition.lyrics_alignment import (
    END_COST,
    LANGUAGES_WITHOUT_SPACES,
    OVERLAP_S,
    WORD_S,
    _Emission,
    _local,
    _moves,
    _Placement,
    align_lyrics,
    align_path,
    backtrack,
    blank_index,
    emission_in_pieces,
    end_row,
    line_times_from_words,
    lyrics_tokens,
    trellis,
    with_wildcard,
)

LETTERS = "abcdefghijklmnopqrstuvwxyz'"
DICTIONARY = {"<pad>": 0, "|": 1, **{c: i + 2 for i, c in enumerate(LETTERS)}}
VOCAB = len(DICTIONARY)
BLANK = 0
FRAME_S = 0.02
WHOLE = "whole song"
PIECES = "30-s pieces"


def frame(t: float) -> int:
    return round(t / FRAME_S)


def line_words(i: int) -> list[str]:
    """Two words of a letter of their own per line: no line fits another one's spikes."""
    return [LETTERS[i] * 2, LETTERS[i] * 3]


def line_tokens(tokens, i: int) -> np.ndarray:
    return np.flatnonzero(tokens.word_line[tokens.word] == i)


def spike_song(sung: list, n_frames: int):
    """Tokens of len(sung) lines and an emission where line i is sung from
    sung[i] seconds on (its tokens 2 frames apart) or not at all (None)."""
    tokens = lyrics_tokens([line_words(i) for i in range(len(sung))], DICTIONARY, "en")
    em = np.full((n_frames, VOCAB), -10.0, dtype=np.float32)
    em[:, BLANK] = -0.01
    for i, t in enumerate(sung):
        if t is None:
            continue
        for n, k in enumerate(line_tokens(tokens, i)):
            f = frame(t) + 2 * n
            em[f, BLANK] = -10.0
            em[f, tokens.ids[k]] = -0.01
    return tokens, em


def lrc_times(n: int) -> list[float]:
    return [2.0 + 3.0 * i for i in range(n)]


def starts_of_line(result, tokens, i: int) -> list[float]:
    return [s for w, s, _ in result.words if tokens.word_line[w] == i]


def placement(start: float, end: float, score: float) -> _Placement:
    return _Placement(start, end, score, None)


# ---------------------------------------------------------------------------
# Tokens, as whisperx cleans the text
# ---------------------------------------------------------------------------


def whisperx_tokens(text: str, dictionary: dict, language: str) -> list[int]:
    """The tokens whisperx.align builds for a segment with this text (-1 for the wildcard)."""
    import torch
    import whisperx.alignment as wa

    captured = {}
    real_get_trellis = wa.get_trellis

    def spy(emission, tokens, blank_id=0):
        captured["tokens"] = list(tokens)
        return real_get_trellis(emission, tokens, blank_id)

    vocab = max(dictionary.values()) + 1

    class Model:
        def __call__(self, x, lengths=None):
            return torch.zeros(1, 200, vocab), None

    class Splitter:
        def span_tokenize(self, text):
            return [(0, len(text))]

    with patch.object(wa, "get_trellis", spy), patch.object(wa, "nltk_load", lambda *a, **k: Splitter()):
        wa.align([{"text": text, "start": 0.0, "end": 4.0}], Model(),
                 {"dictionary": dictionary, "language": language, "type": "torchaudio"},
                 np.zeros(4 * 16000, dtype=np.float32), "cpu")
    return [t if t < vocab else -1 for t in captured["tokens"]]


class TestLyricsTokens:

    @pytest.mark.parametrize("text", [
        "We'll MEET at noon",              # upper case, apostrophe
        "Bus 42 leaves at 7, OK?",        # digits and punctuation: the wildcard
        "  hello world  ",                # spaces around the text
        "a|b and İzmir",                  # "|" in a word; a character lower() makes two of
        "it's 'quoted'",
    ])
    def test_tokens_as_whisperx(self, text):
        tokens = lyrics_tokens([text.split()], DICTIONARY, "en")
        assert tokens.ids.tolist() == whisperx_tokens(text, DICTIONARY, "en")

    def test_tokens_of_several_lines_as_one_segment(self):
        lines = [["Hello", "there"], ["good", "Morning!"]]
        text = " ".join(w for line in lines for w in line)
        tokens = lyrics_tokens(lines, DICTIONARY, "en")
        assert tokens.ids.tolist() == whisperx_tokens(text, DICTIONARY, "en")
        assert tokens.words == ["Hello", "there", "good", "Morning!"]
        assert tokens.word_line.tolist() == [0, 0, 1, 1]
        assert tokens.word_source.tolist() == [0, 1, 2, 3]

    def test_language_without_spaces(self):
        dictionary = {"<pad>": 0, "|": 1, "あ": 2, "い": 3, "う": 4}
        tokens = lyrics_tokens([["あい", "うえ"]], dictionary, "ja")
        assert tokens.ids.tolist() == whisperx_tokens("あい うえ", dictionary, "ja") == [2, 3, 4, -1]
        # each character is a word, as whisperx gives them
        assert tokens.words == ["あ", "い", "う", "え"]
        assert tokens.word_source.tolist() == [0, 0, 1, 1]

    def test_languages_without_spaces_as_whisperx(self):
        from whisperx.alignment import LANGUAGES_WITHOUT_SPACES as whisperx_languages
        assert sorted(LANGUAGES_WITHOUT_SPACES) == sorted(whisperx_languages)

    def test_space_belongs_to_the_word_before_it(self):
        tokens = lyrics_tokens([["ab", "c"], ["d"]], DICTIONARY, "en")
        sep = DICTIONARY["|"]
        assert tokens.ids.tolist() == [2, 3, sep, 4, sep, 5]
        assert tokens.word.tolist() == [0, 0, 0, 1, 1, 2]
        assert tokens.letter.tolist() == [True, True, False, True, False, True]
        # the space between the lines belongs to the first line
        assert tokens.word_line[tokens.word].tolist() == [0, 0, 0, 0, 0, 1]

    def test_no_separator_token_without_one_in_the_dictionary(self):
        dictionary = {k: v for k, v in DICTIONARY.items() if k != "|"}
        tokens = lyrics_tokens([["ab", "c"]], dictionary, "en")
        assert tokens.ids.tolist() == whisperx_tokens("ab c", dictionary, "en") == [2, 3, 4]
        assert tokens.letter.all()

    def test_blank_index(self):
        assert blank_index({"<pad>": 3, "a": 1}) == 3
        assert blank_index({"a": 0, "[pad]": 7}) == 7
        assert blank_index({"-": 0, "|": 1, "a": 2}) == 0


# ---------------------------------------------------------------------------
# Trellis and backtrack, as whisperx
# ---------------------------------------------------------------------------


class TestPathAsWhisperx:

    @pytest.mark.parametrize("n_frames,n_tokens", [(5, 5), (6, 5), (40, 7), (300, 60), (1200, 150)])
    def test_trellis_and_backtrack_as_whisperx(self, n_frames, n_tokens):
        import torch
        from whisperx.alignment import backtrack as whisperx_backtrack
        from whisperx.alignment import get_trellis

        rng = np.random.default_rng(n_frames)
        vocab, blank_id = 12, 3
        log_probs = np.log(rng.dirichlet(np.ones(vocab), size=n_frames)).astype(np.float32)
        ids = rng.choice([i for i in range(vocab) if i != blank_id], n_tokens)
        ids[::4] = -1  # characters the model does not know
        emission, cols = with_wildcard(log_probs, ids, blank_id)

        # the wildcard column as whisperx adds it
        em = torch.from_numpy(log_probs)
        mask = torch.ones(vocab, dtype=torch.bool)
        mask[blank_id] = False
        expected_emission = torch.cat([em, em[:, mask].max(dim=1).values.unsqueeze(1)], dim=1)
        assert np.array_equal(emission, expected_emission.numpy())
        assert (cols[ids < 0] == vocab).all()

        expected = get_trellis(expected_emission, cols.tolist(), blank_id)
        tr = trellis(emission, cols, blank_id)
        assert np.array_equal(tr, expected.numpy())

        path = whisperx_backtrack(expected, expected_emission, cols.tolist(), blank_id)
        expected_frames = [min(p.time_index for p in path if p.token_index == j) for j in range(n_tokens)]
        t_end = end_row(tr, emission[:, blank_id])
        assert t_end == path[-1].time_index + 1
        assert backtrack(tr, emission, cols, blank_id, t_end).tolist() == expected_frames

    def test_no_path_with_fewer_frames_than_tokens(self):
        emission = np.zeros((3, 4), dtype=np.float32)
        assert align_path(emission, np.array([1, 2, 3, 1]), 0) is None
        assert align_path(emission, np.array([], dtype=np.int64), 0) is None


# ---------------------------------------------------------------------------
# Path end: free or capped
# ---------------------------------------------------------------------------


def end_emission(trailing_blank: float) -> np.ndarray:
    """Tokens 1 and 2 clearly at frames 2 and 6. Tokens 3 and 4 fit poorly at
    frames 8 and 9 and well at frames 30 and 34, after sung frames (blank
    unlikely); the frames after 34 have the given blank log-probability."""
    em = np.full((40, 5), -8.0, dtype=np.float32)
    em[:, 0] = -0.25
    for f, tok in ((2, 1), (6, 2), (30, 3), (34, 4)):
        em[f, 0], em[f, tok] = -8.0, -0.25
    em[8, 3] = em[9, 4] = -4.0
    em[8:10, 0] = -1.0
    em[10:30, 0] = -1.0
    em[31:34, 0] = -1.0
    em[35:, 0] = trailing_blank
    return em


class TestPathEnd:
    tokens = np.array([1, 2, 3, 4])

    def test_capped_end_goes_on_to_sung_frames(self):
        em = end_emission(trailing_blank=-1.0)
        free_frames, free_end = align_path(em, self.tokens, 0)
        capped_frames, capped_end = align_path(em, self.tokens, 0, END_COST)
        # the free end squeezes the last tokens in right after the first ones
        assert free_frames.tolist() == [2, 6, 8, 9]
        assert free_end == 10
        assert capped_frames.tolist() == [2, 6, 30, 34]
        assert capped_end == 35

    def test_capped_end_stops_where_the_free_end_does_before_silence(self):
        em = np.full((40, 5), -8.0, dtype=np.float32)
        em[:, 0] = -0.25
        for f, tok in ((2, 1), (6, 2), (10, 3), (14, 4)):
            em[f, 0], em[f, tok] = -8.0, -0.25
        free = align_path(em, self.tokens, 0)
        capped = align_path(em, self.tokens, 0, END_COST)
        assert free[0].tolist() == capped[0].tolist() == [2, 6, 10, 14]
        assert free[1] == capped[1] == 15

    def test_end_cost_zero_is_the_free_end(self):
        em = end_emission(trailing_blank=-1.0)
        tr = trellis(em, self.tokens, 0)
        assert end_row(tr, em[:, 0], 0.0) == end_row(tr, em[:, 0]) == 10

    def test_unlimited_end_cost_pays_for_all_frames(self):
        em = end_emission(trailing_blank=-1.0)
        tr = trellis(em, self.tokens, 0)
        t_end = end_row(tr, em[:, 0], -np.inf)
        # as good as the path that runs to the last frame
        assert tr[t_end, 4] + em[t_end:, 0].sum() == tr[len(em), 4]
        assert backtrack(tr, em, self.tokens, 0, t_end).tolist() == [2, 6, 30, 34]


# ---------------------------------------------------------------------------
# Emission in pieces
# ---------------------------------------------------------------------------


def wav2vec2_frames(n_samples: int) -> int:
    """Frames of a wav2vec2 model: a 400-sample window every 320 samples
    (shorter inputs are padded to one window)."""
    return max(1, (n_samples - 400) // 320 + 1)


def frame_number_model(calls: list):
    """A fake model for audio that holds its sample indices: per frame, the
    song frame it starts at and the length of the input."""
    def run(samples):
        calls.append((int(samples[0]), len(samples)))
        n = wav2vec2_frames(len(samples))
        out = np.zeros((n, 2), dtype=np.float32)
        out[:, 0] = samples[np.minimum(np.arange(n) * 320, len(samples) - 1)] / 320
        out[:, 1] = len(samples)
        return out
    return run


class TestEmissionInPieces:

    @pytest.mark.parametrize("seconds", [0.5, 29.0, 30.0, 30.0 + 100 / 16000, 30.0 + 1000 / 16000, 61.3, 95.0])
    def test_frames_stitched_at_their_song_frame(self, seconds):
        audio = np.arange(round(seconds * 16000), dtype=np.float32)
        calls = []
        em = emission_in_pieces(frame_number_model(calls), audio)
        # as many frames as the whole song, each from its own place
        assert len(em) == wav2vec2_frames(len(audio)) == len(frame_number_model([])(audio))
        assert em[:, 0].tolist() == list(range(len(em)))
        # pieces of 30 s with up to 5 s of context on both sides
        piece, context = 30 * 16000, 5 * 16000
        expected = []
        for start in range(0, len(audio), piece):
            a, b = max(0, start - context), min(len(audio), start + piece + context)
            expected.append((a, b - a))
        assert calls == expected

    def test_each_frame_comes_from_the_piece_it_lies_in(self):
        audio = np.arange(70 * 16000, dtype=np.float32)
        em = emission_in_pieces(frame_number_model([]), audio)
        lengths = em[:, 1]
        piece = 30 * 16000 // 320
        assert (lengths[:piece] == 35 * 16000).all()             # first piece: context on the right only
        assert (lengths[piece:2 * piece] == 40 * 16000).all()    # context on both sides
        assert (lengths[2 * piece:] == 15 * 16000).all()         # last piece: 10 s and the context on the left

    def test_short_audio_is_one_piece(self):
        audio = np.arange(300, dtype=np.float32)  # shorter than one window: the model pads it
        calls = []
        em = emission_in_pieces(frame_number_model(calls), audio)
        assert calls == [(0, 300)]
        assert em.shape == (1, 2)

    def test_given_number_of_frames(self):
        audio = np.arange(31 * 16000, dtype=np.float32)
        em = emission_in_pieces(frame_number_model([]), audio, n_frames=1000)
        assert em[:, 0].tolist() == list(range(1000))

    def test_frame_arithmetic_of_wav2vec2(self):
        """The fake model's frames are those of a real wav2vec2 feature extractor."""
        import torch
        import torchaudio

        extractor = torchaudio.models.wav2vec2_base().feature_extractor
        for n in (400, 401, 719, 720, 1039, 16000, 16001, 48333):
            with torch.inference_mode():
                features, _ = extractor(torch.zeros(1, n), None)
            assert features.shape[1] == wav2vec2_frames(n)


# ---------------------------------------------------------------------------
# Choice of the alignment
# ---------------------------------------------------------------------------


def misplaced_line_song(n_lines: int = 20, line: int = 10):
    """Lines sung at their LRC times; the given line quietly (its tokens less
    likely), while just before its LRC window the frames are sung (blank
    unlikely) and fit its tokens poorly, so the path puts the line there."""
    times = lrc_times(n_lines)
    tokens, em = spike_song(times, frame(times[-1]) + 300)
    ks = line_tokens(tokens, line)
    for n, k in enumerate(ks):
        f = frame(times[line]) + 2 * n
        em[f, BLANK], em[f, tokens.ids[k]] = -0.01, -1.0
    x0, x1 = frame(times[line] - 2.6), frame(times[line] - 2.1)
    em[x0:x1, BLANK] = -5.0
    em[x0:x1, np.unique(tokens.ids[ks])] = -4.0
    return tokens, em, times


class TestChoice:

    def test_more_lines_at_their_lrc_time_wins(self):
        tokens, misplaced, times = misplaced_line_song()
        _, right = spike_song(times, len(misplaced))
        result = align_lyrics(tokens, [(WHOLE, misplaced), (PIECES, right)], BLANK, FRAME_S, times)
        assert result.emission == PIECES
        assert not result.capped
        assert result.fit == result.lines == 20

    def test_tie_keeps_the_whole_song_and_the_free_end(self):
        times = lrc_times(20)
        tokens, em = spike_song(times, frame(times[-1]) + 300)
        result = align_lyrics(tokens, [(WHOLE, em), (PIECES, em.copy())], BLANK, FRAME_S, times)
        assert result.emission == WHOLE
        assert not result.capped
        assert result.fit == 20

    def test_capped_end_wins_when_the_free_end_squeezes_the_last_line(self):
        times = lrc_times(20)
        tokens, em = spike_song(times, frame(times[-1]) + 100)
        # sung frames (blank unlikely) between the last two lines, where the
        # last line's tokens also fit, but poorly
        last = line_tokens(tokens, 19)
        gap0, gap1 = frame(times[18] + 0.3), frame(times[19])
        em[gap0:gap1, BLANK] = -1.0
        for n, k in enumerate(last):
            em[gap0 + 1 + n, tokens.ids[k]] = -3.0
        free = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, None)
        assert starts_of_line(free, tokens, 19)[0] < times[19] - 1.5
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        assert result.capped
        assert result.fit == 20
        assert starts_of_line(result, tokens, 19)[0] == pytest.approx(times[19])


# ---------------------------------------------------------------------------
# Line check
# ---------------------------------------------------------------------------


class TestLineCheck:

    def test_misplaced_line_moves_to_its_lrc_window(self, capsys):
        tokens, em, times = misplaced_line_song()
        free = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, None)
        assert starts_of_line(free, tokens, 10)[0] < times[10] - 2  # the path puts it before its window
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        assert result.fit == 19
        assert result.moved == 1
        assert result.dropped == []
        assert starts_of_line(result, tokens, 10) == pytest.approx([times[10], times[10] + 0.12])
        # the other lines stay where the path put them
        for i in (9, 11):
            assert starts_of_line(result, tokens, i)[0] == pytest.approx(times[i])

    def test_moved_line_keeps_the_order(self):
        glob = {0: placement(0.0, 2.0, -0.1), 1: placement(3.0, 5.0, -6.0), 2: placement(6.0, 8.0, -0.1)}
        # better in its window, but it would start before the previous line ends (minus 0.3 s)
        local = {0: None, 1: placement(1.6, 2.6, -0.5), 2: None}
        assert _moves([0, 1, 2], glob, local) == {0: 0, 1: 0, 2: 0}
        local[1] = placement(1.8, 2.6, -0.5)
        assert _moves([0, 1, 2], glob, local) == {0: 0, 1: 1, 2: 0}

    def test_moves_that_overlap_each_other_take_the_bigger_gain(self):
        glob = {0: placement(0.0, 2.0, -6.0), 1: placement(10.0, 12.0, -6.0), 2: placement(30.0, 32.0, -6.0)}
        # each move alone keeps the order, both together do not
        local = {0: None, 1: placement(14.0, 21.0, -1.0), 2: placement(20.0, 22.0, -3.0)}
        assert _moves([0, 1, 2], glob, local) == {0: 0, 1: 1, 2: 0}
        local[2] = placement(20.0, 22.0, -0.5)
        assert _moves([0, 1, 2], glob, local) == {0: 0, 1: 0, 2: 1}

    def test_no_move_when_the_local_place_is_close_or_not_clearly_better(self):
        glob = {0: placement(0.0, 2.0, -0.1), 1: placement(5.0, 7.0, -3.0), 2: placement(9.0, 10.0, -0.1)}
        local = {0: None, 1: placement(5.9, 7.5, -0.5), 2: None}   # within 1 s
        assert _moves([0, 1, 2], glob, local)[1] == 0
        local[1] = placement(3.0, 4.0, -2.8)                          # gain of 0.2 only
        assert _moves([0, 1, 2], glob, local)[1] == 0

    def test_lines_move_also_with_an_unreliable_offset(self):
        tokens, em, times = misplaced_line_song()
        # most lines are 3 s before or after their LRC time: the offset is not
        # reliable, which only stops lines from being left out
        jittered = [t + (3.0 if i % 2 else -3.0) if i not in (9, 10, 11) and i % 3 else t
                    for i, t in enumerate(times)]
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, jittered)
        assert result.moved == 1
        assert starts_of_line(result, tokens, 10)[0] == pytest.approx(times[10])

    def test_words_of_a_moved_line_overlap_the_previous_line_at_most_a_little(self):
        tokens, em, times = misplaced_line_song()
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        assert result.moved == 1
        by_line = {}
        for w, start, end in result.words:
            by_line.setdefault(int(tokens.word_line[w]), []).append((start, end))
        for i in range(1, 20):
            assert by_line[i][0][0] >= by_line[i - 1][-1][1] - OVERLAP_S

    def test_two_lines_are_not_checked(self):
        # a short LRC: too few lines for an offset, but the choice still runs
        tokens, em = spike_song([2.0, 5.0], 400)
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, [2.0, 5.0])
        assert (result.fit, result.moved, result.dropped) == (2, 0, [])
        assert starts_of_line(result, tokens, 1)[0] == pytest.approx(5.0)

    def test_no_local_place_past_the_audio_or_in_a_too_short_window(self):
        times = lrc_times(20)
        tokens, em = spike_song(times, frame(times[-1]) + 100)
        emission = _Emission(em, tokens.ids, BLANK, FRAME_S)
        rng = (int(line_tokens(tokens, 19)[0]), int(line_tokens(tokens, 19)[-1]) + 1)
        assert _local(emission, 19, rng, 0.0, times) is not None
        # an offset that puts the line after the end of the audio
        assert _local(emission, 19, rng, 30.0, times) is None
        # a window that the end of the audio cuts to fewer frames than the line has tokens
        audio_end = len(em) * FRAME_S
        offset = audio_end - 0.06 + lyrics_alignment.MARGIN_S - times[19]
        assert _local(emission, 19, rng, offset, times) is None

    def test_lines_are_checked_from_five_well_placed_lines_on(self):
        # five lines at least as well placed as the median line give the offset
        tokens, em, times = misplaced_line_song(n_lines=6, line=3)
        assert align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times).moved == 1
        # four do not: no line check
        tokens, em, times = misplaced_line_song(n_lines=5, line=3)
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        assert result.moved == 0
        assert starts_of_line(result, tokens, 3)[0] < times[3] - 2


class TestTailDrop:

    def song(self, sung_lines: int = 20, n_lines: int = 22, extra=None):
        times = lrc_times(n_lines)
        sung = [t if i < sung_lines else None for i, t in enumerate(times)]
        for i in extra or ():
            sung[i] = None
        n_frames = frame(times[sung_lines - 1]) + 200  # 4 s after the last sung line
        tokens, em = spike_song(sung, n_frames)
        return tokens, em, times

    def test_lines_past_the_audio_end_are_left_out(self):
        tokens, em, times = self.song()
        real_lines = lyrics_alignment._Emission.lines
        calls = []

        def lines_spy(self, ranges, lines, end_cost):
            calls.append(list(lines))
            return real_lines(self, ranges, lines, end_cost)

        with patch.object(lyrics_alignment._Emission, "lines", lines_spy):
            result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        assert result.dropped == [20, 21]
        assert {int(tokens.word_line[w]) for w, _, _ in result.words} == set(range(20))
        # the path runs once more over the kept lines
        assert calls[-1] == list(range(20))
        assert starts_of_line(result, tokens, 19)[0] == pytest.approx(times[19])

    def test_poor_line_in_the_middle_is_kept(self):
        tokens, em, times = self.song(extra=[10])
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        assert result.dropped == [20, 21]
        assert len(starts_of_line(result, tokens, 10)) == 2

    def test_poor_line_at_the_end_within_the_audio_is_kept(self):
        tokens, em, times = self.song(extra=[19], sung_lines=20)
        # line 19 is not sung, but the LRC puts it before the end of the audio
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        assert result.dropped == [20, 21]
        assert len(starts_of_line(result, tokens, 19)) == 2

    def test_unreliable_offset_gives_no_drops(self):
        tokens, em, times = self.song()
        jittered = [t + (3.0 if i % 2 else -3.0) if i < 18 and i % 3 else t for i, t in enumerate(times)]
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, jittered)
        assert result.dropped == []
        assert {int(tokens.word_line[w]) for w, _, _ in result.words} == set(range(22))

    def test_never_all_lines(self):
        # ten lines sung poorly (tokens unlikely) at their LRC times, all in the
        # last 2 s of the audio
        times = [10.2 + 0.18 * i for i in range(10)]
        tokens = lyrics_tokens([line_words(i) for i in range(10)], DICTIONARY, "en")
        em = np.full((600, VOCAB), -10.0, dtype=np.float32)
        em[:, BLANK] = -0.01
        for i, t in enumerate(times):
            for n, k in enumerate(line_tokens(tokens, i)):
                em[frame(t) + n, BLANK], em[frame(t) + n, tokens.ids[k]] = -10.0, -5.0
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        assert result.fit == 10
        assert result.dropped == []
        assert len(result.words) == 20


# ---------------------------------------------------------------------------
# Plain lyrics and word times
# ---------------------------------------------------------------------------


class TestPlainLyrics:

    def test_whole_song_free_end_without_line_check(self):
        tokens, em, times = misplaced_line_song()
        _, right = spike_song(times, len(em))
        result = align_lyrics(tokens, [(WHOLE, em), (PIECES, right)], BLANK, FRAME_S, None)
        assert (result.emission, result.capped, result.fit, result.moved, result.dropped) == (WHOLE, False, None, 0, [])
        # line 10 stays where the path put it
        assert starts_of_line(result, tokens, 10)[0] < times[10] - 2
        assert len(result.words) == 40

    def test_blank_in_the_first_column_without_a_pad_label(self):
        # torchaudio's models: the blank is the first label ("-"), not "<pad>"
        dictionary = {"-": 0, **{c: i for c, i in DICTIONARY.items() if c != "<pad>"}}
        assert blank_index(dictionary) == 0
        times = lrc_times(20)
        tokens, em = spike_song(times, frame(times[-1]) + 100)
        dash_tokens = lyrics_tokens([line_words(i) for i in range(20)], dictionary, "en")
        expected = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        result = align_lyrics(dash_tokens, [(WHOLE, em)], blank_index(dictionary), FRAME_S, times)
        assert result.words == expected.words

    def test_unusable_frames_count_as_silence(self):
        # one NaN frame would make every later frame of the trellis NaN
        times = lrc_times(20)
        tokens, em = spike_song(times, frame(times[-1]) + 100)
        expected = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        broken = em.copy()
        broken[frame(times[5] + 1.5)] = np.nan
        result = align_lyrics(tokens, [(WHOLE, broken)], BLANK, FRAME_S, times)
        assert result.words == expected.words
        assert np.isnan(broken).any()  # the caller's emission is left as it was

    def test_no_tokens(self):
        tokens = lyrics_tokens([], DICTIONARY, "en")
        assert align_lyrics(tokens, [(WHOLE, np.zeros((10, VOCAB), dtype=np.float32))], BLANK, FRAME_S) is None


class TestWordTimes:

    def test_word_times_as_whisperx_forms_them(self):
        tokens, em = spike_song([1.0, 3.0], 250)
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, None)
        # "aa": letters at frames 50, 52, the space after it at 54; "aaa" up to its space
        # at 62; the last word ends where the path ends, after its last letter
        assert result.words == [
            (0, 1.0, 1.08), (1, 1.12, 1.24),
            (2, 3.0, 3.08), (3, 3.12, 3.22),
        ]

    def test_word_after_a_moved_line(self):
        tokens, em, times = misplaced_line_song()
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        words = [(w, s, e) for w, s, e in result.words if tokens.word_line[w] == 10]
        # the moved line's words end at the frames of its own spaces
        assert words == [(20, pytest.approx(times[10]), pytest.approx(times[10] + 0.08)),
                         (21, pytest.approx(times[10] + 0.12), pytest.approx(times[10] + 0.24))]

    def test_words_in_lyrics_order(self):
        tokens, em, times = misplaced_line_song()
        result = align_lyrics(tokens, [(WHOLE, em)], BLANK, FRAME_S, times)
        assert [w for w, _, _ in result.words] == list(range(40))


# ---------------------------------------------------------------------------
# Line times of plain lyrics from the words of a transcription
# ---------------------------------------------------------------------------


def heard(*words) -> list[tuple[str, float, float]]:
    """Transcribed words (text, start): each lasts 0.3 s."""
    return [(text, start, start + 0.3) for text, start in words]


def steady_song(n_lines: int, step: float, first: float = 10.0):
    """Lines of three words, every word heard ``step`` s after the one before it."""
    lines = [[f"{LETTERS[i] * 2}{k}" for k in range(3)] for i in range(n_lines)]
    words = [(w, first + step * n, first + step * n + 0.2) for n, w in enumerate(w for line in lines for w in line)]
    return lines, words


class TestLineTimesFromWords:

    def test_line_starts_at_its_first_paired_word(self):
        lines = [["hello", "dear", "world"], ["come", "on", "now"]]
        # the first word of the first line was not heard: its time is that of
        # "dear" minus one word (0.35 s with too few steps to measure it)
        times, anchored = line_times_from_words(lines, heard(("dear", 10.0), ("world", 10.4), ("come", 15.0)))
        assert times == pytest.approx([10.0 - WORD_S, 15.0])
        assert anchored == 2

    def test_words_paired_without_case_accents_and_punctuation(self):
        lines = [["Don't", "go"], ["CAFÉ,", "now!"]]
        times, anchored = line_times_from_words(lines, heard(("dont ", 3.0), ("cafe", 8.0)))
        assert times == pytest.approx([3.0, 8.0])
        assert anchored == 2

    def test_words_paired_in_order(self):
        # a chorus sung twice: each line pairs with its own occurrence
        lines = [["la", "di", "da"], ["verse", "words"], ["la", "di", "da"]]
        words = heard(("la", 5.0), ("di", 5.4), ("da", 5.8), ("verse", 10.0), ("words", 10.4),
                      ("la", 20.0), ("di", 20.4), ("da", 20.8))
        times, anchored = line_times_from_words(lines, words)
        assert times == pytest.approx([5.0, 10.0, 20.0])
        assert anchored == 3

    def test_frequent_words_are_paired_in_a_long_text(self):
        # a longest common subsequence pairs them all; a matcher that skips the
        # frequent elements of long sequences (difflib's) pairs none of them here
        lines = [["la"] * 5 for _ in range(60)]
        words = heard(("oh", 1.0), *[("la", 2.0 + 0.4 * n) for n in range(300)])
        times, anchored = line_times_from_words(lines, words)
        assert anchored == 60
        assert times == pytest.approx([2.0 + 2.0 * i for i in range(60)])

    def test_seconds_per_word_from_the_steps_between_paired_words(self):
        # ten steps of 0.5 s between consecutive words of a line give 0.5 s per word
        lines, words = steady_song(6, 0.5)
        lines[5] = ["unheard", "unheard"] + lines[5]
        times, _ = line_times_from_words(lines, words)
        # the last line's first paired word is its third word
        assert times[5] == pytest.approx(words[15][1] - 2 * 0.5)

    def test_fewer_steps_give_the_default(self):
        # five lines: ten steps within lines, but the first line's word is missed
        lines, words = steady_song(5, 0.5)
        del words[1]  # its two steps are gone: eight are left
        lines.append(["unheard", lines[0][0]])
        words.append((lines[0][0], 30.0, 30.2))
        times, _ = line_times_from_words(lines, words)
        assert times[5] == pytest.approx(30.0 - WORD_S)

    @pytest.mark.parametrize("step,per_word", [(2.0, 1.0), (0.1, 0.15)])
    def test_seconds_per_word_are_limited(self, step, per_word):
        lines, words = steady_song(6, step)
        lines.append(["unheard", "late"])
        words.append(("late", 200.0, 200.2))
        times, _ = line_times_from_words(lines, words)
        assert times[6] == pytest.approx(200.0 - per_word)

    def test_steps_across_lines_or_missed_words_do_not_count(self):
        # one word per line: no step lies within a line, so the default holds
        lines = [[f"w{i}"] for i in range(12)] + [["unheard", "last"]]
        words = heard(*[(f"w{i}", 1.0 * i) for i in range(12)], ("last", 50.0))
        times, _ = line_times_from_words(lines, words)
        assert times[12] == pytest.approx(50.0 - WORD_S)

    def test_long_transcribed_word_gives_no_line_time(self):
        lines = [["one", "two", "three"]]
        # "one" lasts 4 s: its start is that of its piece, "two" places the line
        words = [("one", 0.0, 4.0), ("two", 4.2, 4.5), ("three", 4.6, 4.9)]
        times, anchored = line_times_from_words(lines, words)
        assert times == pytest.approx([4.2 - WORD_S])
        assert anchored == 1

    def test_anchors_out_of_time_order_are_dropped(self):
        lines = [["aa", "x"], ["bb", "x"], ["cc", "x"], ["dd", "x"]]
        # the second line was heard before the first one
        times, anchored = line_times_from_words(lines, heard(("aa", 10.0), ("bb", 5.0), ("cc", 20.0), ("dd", 30.0)))
        assert anchored == 3
        # it lies between the first and the third line, by its words
        assert times == pytest.approx([10.0, 15.0, 20.0, 30.0])

    def test_lines_between_anchors_are_spread_by_their_words(self):
        lines = [["aa", "x"], ["y", "y", "y"], ["z"], ["bb", "x"]]
        times, anchored = line_times_from_words(lines, heard(("aa", 10.0), ("bb", 22.0)))
        assert anchored == 2
        # words before each line: 0, 2, 5, 6
        assert times == pytest.approx([10.0, 14.0, 20.0, 22.0])

    def test_lines_before_and_after_the_anchors_at_the_seconds_per_word(self):
        lines = [["x"], ["x", "x"], ["aa", "x", "x"], ["y"], ["five", "words", "in", "this", "line"], ["z"]]
        times, anchored = line_times_from_words(lines, heard(("aa", 1.0)))
        assert anchored == 1
        # words before each line: 0, 1, 3, 6, 7, 12; the first line would start before 0
        assert times == pytest.approx([0.0, 1.0 - 2 * WORD_S, 1.0, 1.0 + 3 * WORD_S, 1.0 + 4 * WORD_S,
                                       1.0 + 9 * WORD_S])

    def test_no_line_time_before_zero(self):
        lines = [["x", "x", "x", "aa"]]
        times, _ = line_times_from_words(lines, heard(("aa", 0.5)))
        assert times == [0.0]

    def test_no_anchor(self):
        lines = [["hello", "world"]]
        assert line_times_from_words(lines, []) == (None, 0)
        assert line_times_from_words(lines, heard(("other", 1.0))) == (None, 0)
        assert line_times_from_words([], heard(("hello", 1.0))) == (None, 0)

    def test_words_without_letters_are_not_paired(self):
        lines = [["...", "-", "go"]]
        times, anchored = line_times_from_words(lines, heard(("...", 1.0), ("go", 2.0)))
        assert anchored == 1
        assert times == pytest.approx([2.0 - 2 * WORD_S])
