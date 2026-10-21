"""Tests for the conversion pipeline in UltraSinger.py (run, InitProcessData,
TranscribeAudio, CreateUltraStarTxt), with separation, transcription, lyrics
lookup, alignment and pitch detection replaced by fakes."""

import contextlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import librosa

import src.UltraSinger as us
from modules.Speech_Recognition.reference_lyrics_aligner import (
    LyricsOfOtherRecording,
    PlainLyricsPlacement,
    SyncedLyricsCheck,
)
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.Speech_Recognition.remote_stt import RemoteTranscription
from src.modules.Ultrastar.ultrastar_txt import FormatVersion


def _pitch_track(regions, length=2.0):
    """A pitch track with a frame every 16 ms: the (start, end, note) regions
    are sung, silence elsewhere."""
    times, frequencies, confidence = [], [], []
    for i in range(int(length / 0.016) + 1):
        t = round(i * 0.016, 3)
        hz, conf = 0.0, 0.0
        for start, end, note in regions:
            if start <= t <= end:
                hz, conf = float(librosa.note_to_hz(note)), 0.9
        times.append(t)
        frequencies.append(hz)
        confidence.append(conf)
    return us.PitchedData(times, frequencies, confidence)


def _syllable(word, start, end):
    return us.TranscribedData(word=word, start=start, end=end, confidence=0.9)


def _process_data(language=None):
    process_data = us.ProcessData()
    process_data.basename = "Artist - Song Title"
    process_data.media_info = us.MediaInfo(title="Song Title", artist="Artist", language=language,
                                           audio_extension="mp3")
    return process_data


def _transcription(words, language):
    """A fake TranscribeAudio: the transcription found ``words`` in ``language``."""
    def transcribe(process_data):
        process_data.transcribed_data = [_syllable(*word) for word in words]
        if us.settings.language is None:
            process_data.media_info.language = language
        return None, None
    return transcribe


def _run(process_data, pitched_data, transcribe, *, early_lyrics=None, fast_language=("en", 0.95),
         align=None, plain_align=None, remote=None, lyrics_check=None, mocks=None,
         audio_paths=("vocals.wav", "vocals_mute.wav"), **settings_values):
    """Run run() with fakes. Returns the notes it wrote, its output and the output folder's files.

    ``align``/``plain_align``: fake alignment of synced/plain lyrics (the remote text is aligned
    as plain lyrics), ``remote``: fake remote speech-to-text (returns a RemoteTranscription),
    ``mocks``: a dict that receives the mocks of TranscribeAudio, the plain-lyrics alignment and
    the remote speech-to-text."""
    written = {}
    mocks = {} if mocks is None else mocks

    def create_txt(data):
        written["notes"] = [(n.word, n.start, n.end, n.note) for n in data.midi_segments]
        return None, None, os.path.join(us.settings.output_folder_path, "song.txt")

    values = dict(
        ignore_audio=False, language=None, bpm_override=120.0, lyrics_lookup=True,
        disable_reference_lyrics=False, onset_correction=False, vocal_gap_fill=False, quantize_to_key=False,
        pitch_change_split=True, pitch_notes=False, syllable_split=False, segmentation_model=None,
        segmentation_model_repo=None, create_audio_chunks=False, octave_snap=False, octave_consistency=False,
        vocal_center_correction=False, octave_shift=None, detect_growl=False, refine_from_vocal=False,
        ptakf_refit=False, calculate_score=False, golden_notes=False, create_plot=False, create_midi=False,
        write_settings_info=False, remote_stt=False, llm_correct_lyrics=False, keep_numbers=False,
        cache_folder=None, keep_cache=False, use_separated_vocal=True, create_karaoke=True,
        check_synced_lyrics=False,
    )
    values.update(settings_values)
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
        values.setdefault("output_folder_path", folder)
        for name, value in values.items():
            stack.enter_context(patch.object(us.settings, name, value))
        for name, fake in (
                ("InitProcessData", lambda: process_data),
                ("setup_cache_folder", lambda: folder),
                ("CreateProcessAudio", lambda data: audio_paths),
                ("detect_key_from_audio", lambda path: ("C", "major")),
                ("pitch_audio", lambda paths: pitched_data),
                ("TranscribeAudio", transcribe),
                ("check_lyrics_language", lyrics_check or (lambda lyrics, language, confident: (True, None))),
                ("CreateUltraStarTxt", create_txt),
                ("create_sheet", lambda *args: None),
                ("finish_cache_folder", lambda path: None),
                ("print_support", lambda: None),
        ):
            mocks[name] = stack.enter_context(patch.object(us, name, side_effect=fake))
        stack.enter_context(patch("modules.lrclib_client.search_lyrics", return_value=early_lyrics))
        mocks["detect_confident"] = stack.enter_context(patch(
            "modules.Speech_Recognition.Whisper.detect_language_with_confidence", return_value=fast_language))
        mocks["detect"] = stack.enter_context(patch(
            "modules.Speech_Recognition.Whisper.detect_language_from_audio", return_value=fast_language[0]))
        stack.enter_context(patch(
            "modules.Speech_Recognition.reference_lyrics_aligner.create_midi_segments_from_reference_lyrics",
            side_effect=align))
        mocks["plain_align"] = stack.enter_context(patch(
            "modules.Speech_Recognition.reference_lyrics_aligner.create_midi_segments_from_plain_lyrics",
            side_effect=plain_align or (lambda **kwargs: ([], PlainLyricsPlacement()))))
        mocks["remote"] = stack.enter_context(patch(
            "modules.Speech_Recognition.remote_stt.transcribe_remote",
            side_effect=remote or (lambda **kwargs: None)))
        stack.enter_context(contextlib.redirect_stdout(out))
        try:
            us.run()
        finally:
            files = {name: open(os.path.join(folder, name), encoding="utf-8").read()
                     for name in os.listdir(folder) if name.endswith(".info")}
    return written.get("notes"), out.getvalue(), files


class TestPitchChangeSplitKeepsWords(unittest.TestCase):
    """Transcription path: a note split at a pitch change keeps all words and notes paired."""

    def test_all_words_to_the_end(self):
        # BPM 60: a 16th note is 0.25 s, so the 0.24 s syllables are not cut into pieces
        words = [("one ", 0.0, 0.24), ("two ", 0.40, 0.64), ("three ", 0.80, 1.04), ("four ", 1.20, 1.44)]
        pitched = _pitch_track([(0.0, 0.24, "C4"), (0.40, 0.52, "C4"), (0.528, 0.64, "G4"),
                                (0.80, 1.04, "E4"), (1.20, 1.44, "A4")])
        notes, _, _ = _run(_process_data(), pitched, _transcription(words, "en"), bpm_override=60.0)

        self.assertEqual([n[0] for n in notes], ["one ", "two", "~ ", "three ", "four "])
        self.assertEqual(notes[-1][1:3], (1.20, 1.44))
        self.assertEqual([n[3] for n in notes], ["C4", "C4", "G4", "E4", "A4"])


class TestEmptyTranscription(unittest.TestCase):
    def test_stops_with_a_clear_message(self):
        with self.assertRaises(SystemExit) as stop:
            _run(_process_data(), _pitch_track([]), _transcription([], "en"))
        self.assertEqual(stop.exception.code, 1)


class TestReferenceRecoveredAfterLanguageCorrection(unittest.TestCase):
    """The notes of the reference pipeline retried with the corrected language are kept."""

    def test_recovered_notes_are_written(self):
        reference = [us.MidiSegment("C4", 1.0, 1.4, "hello "), us.MidiSegment("D4", 1.5, 2.0, "world ")]
        aligned = iter([[], reference])  # fast-detected language fails, the corrected one works
        notes, out, _ = _run(
            _process_data(), _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5),
            _transcription([("hello ", 1.0, 1.4), ("world ", 1.5, 2.0)], "en"),
            early_lyrics=SimpleNamespace(duration=None, synced_lyrics="[00:01.00] hello world", plain_lyrics=None),
            fast_language=("cy", 0.95), align=lambda **kwargs: next(aligned),
        )
        self.assertIn("Reference pipeline recovered after language correction", out)
        self.assertEqual(notes, [("hello ", 1.0, 1.4, "C4"), ("world ", 1.5, 2.0, "D4")])


class TestLanguageSourceInInfoFile(unittest.TestCase):
    """The info file names the language the transcription detected, not the metadata it replaced."""

    def test_transcription_replaces_platform_metadata(self):
        _, _, files = _run(
            _process_data(language="en"), _pitch_track([(0.0, 0.24, "C4")]),
            _transcription([("eins ", 0.0, 0.24)], "de"), write_settings_info=True,
        )
        info = files["ultrasinger_parameter.info"]
        self.assertIn("Language detection:        full transcription -> de", info)
        self.assertIn("Transcriber language hint: no (auto-detect)", info)


class TestPipelineInInfoFile(unittest.TestCase):
    """The [Pipeline] section of the info file names what made the notes."""

    def _info(self, ignore_audio=False, disable_reference_lyrics=False, llm_correct_lyrics=False, **kwargs):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(us.settings, "ignore_audio", ignore_audio), \
                patch.object(us.settings, "disable_reference_lyrics", disable_reference_lyrics), \
                patch.object(us.settings, "llm_correct_lyrics", llm_correct_lyrics), \
                patch.object(us.settings, "language", None), \
                contextlib.redirect_stdout(io.StringIO()):
            us._write_settings_info_file(folder, None, None, **kwargs)
            with open(os.path.join(folder, "ultrasinger_parameter.info"), encoding="utf-8") as f:
                return f.read()

    def test_remote_stt_after_failed_synced_alignment(self):
        info = self._info(reference_first_used=True, has_synced_lyrics=True, remote_stt_used=True,
                          whisper_skipped=True)
        self.assertIn("Pipeline:                 Remote STT (text) + local alignment", info)
        self.assertIn("LRCLIB synced lyrics:     found (but alignment failed)", info)
        self.assertIn("Transcription:            skipped", info)
        self.assertNotIn("Reference-Lyrics-First", info)

    def test_plain_lyrics_after_failed_synced_alignment(self):
        info = self._info(reference_first_used=True, has_synced_lyrics=True, plain_lyrics_used=True)
        self.assertIn("Pipeline:                 Reference-Lyrics-First (plain)", info)
        self.assertIn("LRCLIB synced lyrics:     found (but alignment failed)", info)
        self.assertIn("Transcription:            full (its text replaced by the plain lyrics)", info)

    def test_how_the_plain_lyrics_were_placed(self):
        for placement, text in (
                (PlainLyricsPlacement(48, 45, True, True),
                 "line by line (45 of 48 lines anchored by the transcription's words)"),
                (PlainLyricsPlacement(48, 20, True, False),
                 "as one block (20 of 48 lines anchored by the transcription's words, too few)"),
                (PlainLyricsPlacement(48, 0, False, False), "as one block (no transcription words)")):
            with self.subTest(text=text):
                info = self._info(reference_first_used=True, plain_lyrics_used=True, plain_lyrics_placement=placement)
                self.assertIn(f"  Plain lyrics placed:      {text}\n", info)
        # not for the remote text
        info = self._info(reference_first_used=True, remote_stt_used=True,
                          plain_lyrics_placement=PlainLyricsPlacement(1, 0, False, False))
        self.assertNotIn("Plain lyrics placed", info)

    def test_synced(self):
        info = self._info(reference_first_used=True, has_synced_lyrics=True, whisper_skipped=True)
        self.assertIn("Pipeline:                 Reference-Lyrics-First (synced)", info)
        self.assertIn("  LRCLIB synced lyrics:     found\n", info)

    def test_lyrics_of_the_second_database(self):
        info = self._info(reference_first_used=True, has_synced_lyrics=True, whisper_skipped=True,
                          lyrics_source="music.163.com")
        self.assertIn("  music.163.com synced lyrics: found\n", info)
        self.assertNotIn("LRCLIB", info)
        info = self._info(reference_first_used=True, has_synced_lyrics=True, plain_lyrics_used=True,
                          lyrics_source="music.163.com")
        self.assertIn("  music.163.com plain lyrics: found\n", info)

    def test_reference_lyrics_disabled(self):
        # The lookup after the transcription still finds synced lyrics; they were never aligned
        info = self._info(disable_reference_lyrics=True, has_synced_lyrics=True)
        self.assertIn("Pipeline:                 Standard transcription (reference lyrics disabled)", info)
        self.assertIn("LRCLIB synced lyrics:     found (not aligned: reference lyrics disabled)", info)
        self.assertNotIn("alignment failed", info)

    def test_repitch(self):
        info = self._info(ignore_audio=True, llm_correct_lyrics=True)
        self.assertIn("Pipeline:                 Re-pitch (notes and lyrics of the UltraStar txt)", info)
        self.assertNotIn("Standard transcription", info)
        # no transcription ran: nothing about what it was given or corrected
        self.assertNotIn("Transcriber language hint", info)
        self.assertIn("Status:                   not run (no local transcription to correct)", info)

    def test_language_hint_only_with_language_option(self):
        info = self._info(lang_source="video_platform", detected_language="en")
        self.assertIn("Transcriber language hint: no (auto-detect)", info)

    def test_rejected_synced_lyrics(self):
        info = self._info(rejected_synced_language="de")
        self.assertIn("LRCLIB synced lyrics:     found but rejected (in de, another language than the singing)",
                      info)
        self.assertNotIn("not found", info)


_ALIGNED = [us.MidiSegment("C4", 1.0, 1.4, "hello "), us.MidiSegment("D4", 1.5, 2.0, "world ")]
_ALIGNED_NOTES = [("hello ", 1.0, 1.4, "C4"), ("world ", 1.5, 2.0, "D4")]


def _aligned(**kwargs):
    """A fake alignment of plain lyrics or of the remote text: line by line
    when it gets words of a transcription."""
    words = bool(kwargs.get("transcription_words"))
    placement = PlainLyricsPlacement(lines=2, anchored=2 if words else 0, with_words=words, line_by_line=words,
                                     words_from=kwargs.get("words_from", "transcription"))
    return [us.MidiSegment(n.note, n.start, n.end, n.word) for n in _ALIGNED], placement


_HEARD = RemoteTranscription("other words here", words=[("other", 1.0, 1.2), ("words", 1.3, 1.5), ("here", 1.6, 1.9)],
                              segments=[" other words", " here"])


class TestRemoteSpeechToText(unittest.TestCase):
    """--remote_stt replaces the local transcription; that only runs when nothing could be aligned."""

    def _remote_run(self, heard, early_lyrics=None, plain_align=_aligned, **settings_values):
        mocks = {}
        notes, out, files = _run(
            _process_data(), _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5),
            _transcription([("one ", 1.0, 1.24), ("two ", 1.5, 1.74)], "en"),
            early_lyrics=early_lyrics, plain_align=plain_align, remote=lambda **kwargs: heard,
            mocks=mocks, remote_stt=True, write_settings_info=True, bpm_override=60.0, **settings_values,
        )
        return notes, files["ultrasinger_parameter.info"], mocks

    def test_remote_text_without_local_transcription(self):
        notes, info, mocks = self._remote_run(RemoteTranscription("hello world"), llm_correct_lyrics=True)
        mocks["TranscribeAudio"].assert_not_called()
        mocks["remote"].assert_called_once()
        call = mocks["plain_align"].call_args.kwargs
        self.assertEqual((call["plain_lyrics"], call["transcription_words"]), ("hello world", []))
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("Pipeline:                 Remote STT (text) + local alignment", info)
        self.assertIn("Transcription:            skipped", info)
        self.assertIn("Status:                   not run (no local transcription to correct)", info)
        self.assertIn("Used this run:            yes (its text made the notes)", info)

    def test_remote_segments_are_lines_placed_by_its_words(self):
        notes, info, mocks = self._remote_run(_HEARD)
        call = mocks["plain_align"].call_args.kwargs
        self.assertEqual(call["plain_lyrics"], "other words\nhere")
        self.assertEqual(call["transcription_words"], _HEARD.words)
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("Pipeline:                 Remote STT (text) + local alignment", info)
        # not the plain lyrics' placement
        self.assertNotIn("Plain lyrics placed", info)

    def test_remote_failure_falls_back_to_the_local_transcription(self):
        notes, info, mocks = self._remote_run(None)
        mocks["remote"].assert_called_once()
        mocks["TranscribeAudio"].assert_called_once()
        self.assertEqual([n[0] for n in notes], ["one ", "two "])
        self.assertIn("Pipeline:                 Standard transcription", info)
        self.assertIn("Transcription:            full", info)
        self.assertIn("Used this run:            asked, no usable answer (see the console)", info)

    def test_remote_words_place_the_plain_lyrics(self):
        lyrics = SimpleNamespace(duration=None, synced_lyrics=None, plain_lyrics="hello world")
        notes, info, mocks = self._remote_run(_HEARD, early_lyrics=lyrics)
        mocks["remote"].assert_called_once()
        mocks["TranscribeAudio"].assert_not_called()
        # the plain lyrics make the notes, the remote words give their lines times
        mocks["plain_align"].assert_called_once()
        call = mocks["plain_align"].call_args.kwargs
        self.assertEqual((call["plain_lyrics"], call["transcription_words"]), ("hello world", _HEARD.words))
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("Pipeline:                 Reference-Lyrics-First (plain)", info)
        self.assertIn("Plain lyrics placed:      line by line (2 of 2 lines anchored by the remote transcription's "
                      "words)", info)
        self.assertIn("Transcription:            skipped", info)
        self.assertIn("Used this run:            yes (its words placed the lines of the plain lyrics)", info)

    def test_plain_lyrics_without_remote_words(self):
        # a service that returned no times: the plain lyrics are placed as one block
        lyrics = SimpleNamespace(duration=None, synced_lyrics=None, plain_lyrics="hello world")
        _, info, mocks = self._remote_run(RemoteTranscription("other words"), early_lyrics=lyrics)
        self.assertEqual(mocks["plain_align"].call_args.kwargs["transcription_words"], [])
        self.assertIn("Plain lyrics placed:      as one block (no remote transcription words)", info)
        self.assertIn("Used this run:            asked, its answer not used", info)

    def test_synced_lyrics_that_fail_to_align(self):
        # the remote service is asked then, and its words place the lines of the plain lyrics
        lyrics = SimpleNamespace(duration=None, synced_lyrics="[00:01.00] hello world", plain_lyrics="hello world")
        notes, info, mocks = self._remote_run(_HEARD, early_lyrics=lyrics, align=lambda **kwargs: [])
        mocks["remote"].assert_called_once()
        mocks["TranscribeAudio"].assert_not_called()
        call = mocks["plain_align"].call_args.kwargs
        self.assertEqual((call["plain_lyrics"], call["transcription_words"], call["words_from"]),
                         ("hello world", _HEARD.words, "remote transcription"))
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("LRCLIB synced lyrics:     found (but alignment failed)", info)
        self.assertIn("Used this run:            yes (its words placed the lines of the plain lyrics)", info)

    def test_remote_is_asked_once(self):
        # the plain lyrics fail to align: the remote text makes the notes, from the same answer
        lyrics = SimpleNamespace(duration=None, synced_lyrics=None, plain_lyrics="hello world")
        calls = []

        def plain_align(**kwargs):
            calls.append(kwargs)
            return ([], PlainLyricsPlacement()) if len(calls) == 1 else _aligned(**kwargs)

        notes, info, mocks = self._remote_run(_HEARD, early_lyrics=lyrics, plain_align=plain_align)
        mocks["remote"].assert_called_once()
        mocks["TranscribeAudio"].assert_not_called()
        self.assertEqual([(c["plain_lyrics"], c["transcription_words"]) for c in calls],
                         [("hello world", _HEARD.words), ("other words\nhere", _HEARD.words)])
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("Pipeline:                 Remote STT (text) + local alignment", info)

    def test_synced_lyrics_send_no_audio(self):
        lyrics = SimpleNamespace(duration=None, synced_lyrics="[00:01.00] hello world", plain_lyrics="hello world")
        mocks = {}
        notes, _, _ = _run(
            _process_data(), _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5),
            _transcription([("one ", 1.0, 1.24)], "en"), early_lyrics=lyrics,
            align=lambda **kwargs: [us.MidiSegment(n.note, n.start, n.end, n.word) for n in _ALIGNED],
            remote=lambda **kwargs: _HEARD, mocks=mocks, remote_stt=True,
        )
        mocks["remote"].assert_not_called()
        mocks["plain_align"].assert_not_called()
        self.assertEqual(notes, _ALIGNED_NOTES)

    def test_platform_language_is_not_trusted(self):
        # The metadata is often a default ("en"); no transcription replaces it now
        for early_lyrics, aligner in ((None, "remote"),
                                      (SimpleNamespace(duration=None, synced_lyrics=None, plain_lyrics="hallo welt"), "plain_align")):
            with self.subTest(aligner=aligner):
                mocks = {}
                _, out, files = _run(
                    _process_data(language="en"), _pitch_track([(1.0, 1.4, "C4")], length=2.5),
                    _transcription([("eins ", 1.0, 1.24)], "de"), early_lyrics=early_lyrics,
                    fast_language=("de", 0.95), plain_align=_aligned,
                    remote=lambda **kwargs: RemoteTranscription("hallo welt"),
                    mocks=mocks, remote_stt=True, write_settings_info=True,
                )
                self.assertEqual(mocks[aligner].call_args.kwargs["language"], "de")
                self.assertEqual(mocks["detect_confident"].call_count + mocks["detect"].call_count, 1)
                # (the GUI reads a language change from a "Using video platform language metadata" line)
                self.assertNotIn("Using video platform language metadata", out)
                self.assertIn("Language detection:        Whisper tiny (fast) -> de",
                              files["ultrasinger_parameter.info"])

    def test_without_remote_option_the_transcription_runs_first(self):
        mocks = {}
        _run(_process_data(), _pitch_track([(1.0, 1.24, "C4")], length=2.5),
             _transcription([("one ", 1.0, 1.24)], "en"), mocks=mocks, bpm_override=60.0)
        mocks["TranscribeAudio"].assert_called_once()
        mocks["remote"].assert_not_called()


class TestPlainLyricsAfterFailedSyncedAlignment(unittest.TestCase):
    """Synced lyrics found before the transcription fail to align: their plain lyrics are tried next."""

    def test_plain_lyrics_make_the_notes(self):
        mocks = {}
        lyrics = SimpleNamespace(duration=None, synced_lyrics="[00:01.00] hello world", plain_lyrics="hello world")
        notes, _, files = _run(
            _process_data(), _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5),
            _transcription([("one ", 1.0, 1.24)], "en"), early_lyrics=lyrics,
            align=lambda **kwargs: [], plain_align=_aligned, mocks=mocks, write_settings_info=True,
        )
        mocks["TranscribeAudio"].assert_not_called()
        mocks["remote"].assert_not_called()  # without --remote_stt no audio is sent
        self.assertEqual(notes, _ALIGNED_NOTES)
        # no transcription ran: no words to give the lines times
        self.assertEqual(mocks["plain_align"].call_args.kwargs["transcription_words"], [])
        info = files["ultrasinger_parameter.info"]
        self.assertIn("Pipeline:                 Reference-Lyrics-First (plain)", info)
        self.assertIn("LRCLIB synced lyrics:     found (but alignment failed)", info)
        self.assertIn("Plain lyrics placed:      as one block (no transcription words)", info)
        self.assertIn("Transcription:            skipped", info)


class TestSyncedLyricsCheckedByTranscription(unittest.TestCase):
    """Synced lyrics found before the transcription: a transcription runs only to check their times
    (from the cache when there is one); the full transcription stays skipped."""

    _LYRICS = SimpleNamespace(duration=None, synced_lyrics="[00:01.00] hello world", plain_lyrics="hello world")
    _HEARD = [("hello", 1.0, 1.4), ("world", 1.5, 2.0)]

    def _run_with(self, transcribe=None, checked=None, **kwargs):
        mocks = {}
        calls = {"reference": 0}

        def reference(**kw):
            calls["reference"] += 1
            return [us.MidiSegment(n.note, n.start, n.end, n.word) for n in _ALIGNED]

        result = TranscriptionResult([us.TranscribedData(word=w, start=a, end=b) for w, a, b in self._HEARD], "en")
        check = SyncedLyricsCheck(lines=1, checked=True, confirmed=1, moved=1)
        aligned = lambda **kw: ([us.MidiSegment(n.note, n.start, n.end, n.word) for n in _ALIGNED], check)  # noqa: E731
        with patch.object(us, "transcribe_audio", side_effect=transcribe or (lambda *args: result)) as fake, \
                patch("modules.Speech_Recognition.reference_lyrics_aligner."
                      "create_midi_segments_from_checked_synced_lyrics",
                      side_effect=checked or aligned) as checked_align:
            notes, out, files = _run(
                _process_data(), _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5),
                _transcription([("one ", 1.0, 1.24)], "en"), early_lyrics=self._LYRICS, align=reference,
                mocks=mocks, write_settings_info=True, **kwargs)
        return notes, out, files, mocks, fake, checked_align, calls

    def test_the_transcription_checks_the_lines(self):
        notes, out, files, mocks, transcribe, checked, calls = self._run_with(check_synced_lyrics=True)
        transcribe.assert_called_once()
        mocks["TranscribeAudio"].assert_not_called()
        self.assertEqual(checked.call_args.kwargs["transcription_words"], self._HEARD)
        self.assertEqual(calls["reference"], 0)
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("it may run to check their times", out)
        info = files["ultrasinger_parameter.info"]
        self.assertIn("Pipeline:                 Reference-Lyrics-First (synced)", info)
        self.assertIn("Synced lyrics checked:    1 of 1 lines confirmed by the transcription's words; "
                      "1 moved to where they are sung", info)
        self.assertIn("Transcription:            for the check only", info)

    def test_check_turned_off(self):
        notes, _, files, _, transcribe, checked, calls = self._run_with(check_synced_lyrics=False)
        transcribe.assert_not_called()
        checked.assert_not_called()
        self.assertEqual(calls["reference"], 1)
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("Synced lyrics checked:    no (--disable_synced_lyrics_check)", files["ultrasinger_parameter.info"])

    def test_not_with_remote_speech_to_text(self):
        _, out, files, _, transcribe, checked, calls = self._run_with(check_synced_lyrics=True, remote_stt=True)
        self.assertIn("Synced lyrics checked:    no (not with --remote_stt)", files["ultrasinger_parameter.info"])
        transcribe.assert_not_called()
        checked.assert_not_called()
        self.assertEqual(calls["reference"], 1)
        self.assertNotIn("it may run to check their times", out)

    def test_failed_transcription_keeps_the_lyrics_times(self):
        def fail(*args):
            raise RuntimeError("CUDA out of memory")
        notes, out, files, _, _, checked, calls = self._run_with(transcribe=fail, check_synced_lyrics=True)
        info = files["ultrasinger_parameter.info"]
        self.assertIn("Synced lyrics checked:    no (the transcription failed)", info)
        self.assertNotIn("for the check only", info)
        checked.assert_not_called()
        self.assertEqual(calls["reference"], 1)
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("the transcription failed (CUDA out of memory): the synced lyrics keep their times", out)

    def test_failed_alignment_leaves_the_words_to_the_plain_lyrics(self):
        mocks = {}
        notes, _, files, mocks, _, _, _ = self._run_with(
            checked=lambda **kw: ([], SyncedLyricsCheck(lines=1)), check_synced_lyrics=True, plain_align=_aligned)
        self.assertEqual(mocks["plain_align"].call_args.kwargs["transcription_words"], self._HEARD)
        self.assertEqual(notes, _ALIGNED_NOTES)
        info = files["ultrasinger_parameter.info"]
        self.assertIn("Pipeline:                 Reference-Lyrics-First (plain)", info)
        self.assertIn("Transcription:            for the check of the synced lyrics only", info)


class TestSyncedLyricsOfAnotherRecording(unittest.TestCase):
    """Synced lyrics of a recording much longer or shorter than the audio whose lines mostly miss
    their LRC times: their times are not used, the transcription runs, and its words place the
    plain lyrics (or the text of the synced lyrics)."""

    _HEARD = [("hello ", 1.0, 1.4), ("world ", 1.5, 2.0)]

    def _transcribe(self, process_data):
        process_data.transcription_words = list(self._HEARD)
        process_data.transcribed_data = [_syllable(*word) for word in self._HEARD]
        process_data.media_info.language = "en"
        return None, None

    @staticmethod
    def _align(calls):
        def align(**kwargs):
            calls.append(kwargs)
            if kwargs["other_recording"]:
                raise LyricsOfOtherRecording(7, 62)
            return [us.MidiSegment(n.note, n.start, n.end, n.word) for n in _ALIGNED]
        return align

    def _run_with(self, lyrics, audio_s=391.0, **kwargs):
        with patch.object(us, "audio_length", return_value=audio_s):
            return _run(_process_data(), _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5),
                        self._transcribe, early_lyrics=lyrics, plain_align=_aligned, write_settings_info=True,
                        **kwargs)

    def test_plain_lyrics_placed_by_the_transcription(self):
        calls, mocks = [], {}
        lyrics = SimpleNamespace(duration=236.0, synced_lyrics="[00:01.00] hello world", plain_lyrics="hello world")
        notes, out, files = self._run_with(lyrics, align=self._align(calls), mocks=mocks)
        self.assertEqual([c["other_recording"] for c in calls], [True])
        mocks["TranscribeAudio"].assert_called_once()
        mocks["plain_align"].assert_called_once()
        call = mocks["plain_align"].call_args.kwargs
        self.assertEqual((call["plain_lyrics"], call["transcription_words"]), ("hello world", self._HEARD))
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("the found synced lyrics are of another recording (236 s long, the audio 391 s, "
                      "only 7 of 62 lines at their LRC time)", out)
        self.assertIn("Aligning the text of the lyrics with the words of the transcription", out)
        self.assertNotIn("recovered after language correction", out)
        info = files["ultrasinger_parameter.info"]
        self.assertIn("Pipeline:                 Reference-Lyrics-First (plain)", info)
        self.assertIn("LRCLIB synced lyrics:     found, of another recording (236 s long, the audio 391 s, "
                      "only 7 of 62 lines at their LRC time): their times were not used", info)
        self.assertIn("LRCLIB plain lyrics:      found", info)
        self.assertIn("Transcription:            full (its text replaced by the plain lyrics)", info)

    def test_text_of_the_synced_lyrics_without_plain_lyrics(self):
        mocks = {}
        lyrics = SimpleNamespace(duration=236.0, synced_lyrics="[00:01.00] hello world\n[00:01.50] again",
                                 plain_lyrics=None)
        notes, _, files = self._run_with(lyrics, align=self._align([]), mocks=mocks)
        call = mocks["plain_align"].call_args.kwargs
        self.assertEqual((call["plain_lyrics"], call["transcription_words"]), ("hello world\nagain", self._HEARD))
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("LRCLIB plain lyrics:      the text of the synced lyrics", files["ultrasinger_parameter.info"])

    def test_lyrics_of_about_the_length_of_the_audio(self):
        calls, mocks = [], {}
        lyrics = SimpleNamespace(duration=380.0, synced_lyrics="[00:01.00] hello world", plain_lyrics="hello world")
        notes, _, files = self._run_with(lyrics, align=self._align(calls), mocks=mocks)
        self.assertEqual([c["other_recording"] for c in calls], [False])
        mocks["TranscribeAudio"].assert_not_called()
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("Pipeline:                 Reference-Lyrics-First (synced)", files["ultrasinger_parameter.info"])

    def test_remote_words_place_the_text_at_once(self):
        mocks = {}
        lyrics = SimpleNamespace(duration=236.0, synced_lyrics="[00:01.00] hello world", plain_lyrics="hello world")
        notes, _, _ = self._run_with(lyrics, align=self._align([]), mocks=mocks, remote=lambda **kwargs: _HEARD,
                                     remote_stt=True)
        mocks["TranscribeAudio"].assert_not_called()
        call = mocks["plain_align"].call_args.kwargs
        self.assertEqual((call["plain_lyrics"], call["transcription_words"]), ("hello world", _HEARD.words))
        self.assertEqual(notes, _ALIGNED_NOTES)


class TestLyricsAlignedAgainAfterLanguageCorrection(unittest.TestCase):
    """Nothing aligned in the quickly detected language: after the transcription corrected it, the
    plain lyrics (with its words) or the remote text are aligned again, without asking the remote
    service again."""

    _HEARD_LOCALLY = [("hello ", 1.0, 1.4), ("world ", 1.5, 2.0)]

    def _transcribe(self, process_data):
        process_data.transcription_words = list(self._HEARD_LOCALLY)
        process_data.transcribed_data = [_syllable(*word) for word in self._HEARD_LOCALLY]
        process_data.media_info.language = "en"
        return None, None

    @staticmethod
    def _plain_align(calls):
        def align(**kwargs):
            calls.append(kwargs)
            return _aligned(**kwargs) if kwargs["language"] == "en" else ([], PlainLyricsPlacement())
        return align

    def test_plain_lyrics_with_the_words_of_the_transcription(self):
        calls, mocks = [], {}
        lyrics = SimpleNamespace(duration=None, synced_lyrics="[00:01.00] hello world", plain_lyrics="hello world")
        notes, out, files = _run(
            _process_data(), _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5), self._transcribe,
            early_lyrics=lyrics, fast_language=("cy", 0.95), align=lambda **kwargs: [],
            plain_align=self._plain_align(calls), mocks=mocks, write_settings_info=True,
        )
        mocks["TranscribeAudio"].assert_called_once()
        self.assertEqual([(c["language"], c["transcription_words"]) for c in calls],
                         [("cy", []), ("en", self._HEARD_LOCALLY)])
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("Lyrics alignment recovered after language correction", out)
        # one "Language corrected" line, from the synced retry
        self.assertEqual(out.count("Language corrected"), 1)
        self.assertIn("Aligning the plain lyrics in the corrected language", out)
        info = files["ultrasinger_parameter.info"]
        self.assertIn("Pipeline:                 Reference-Lyrics-First (plain)", info)
        self.assertIn("Reference recovery:        yes (retried with corrected language)", info)

    def test_remote_text_without_a_second_upload(self):
        calls, mocks = [], {}
        notes, out, files = _run(
            _process_data(), _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5), self._transcribe,
            fast_language=("cy", 0.95), plain_align=self._plain_align(calls), remote=lambda **kwargs: _HEARD,
            mocks=mocks, remote_stt=True, write_settings_info=True,
        )
        mocks["remote"].assert_called_once()
        mocks["TranscribeAudio"].assert_called_once()
        self.assertEqual([(c["plain_lyrics"], c["language"], c["transcription_words"]) for c in calls],
                         [("other words\nhere", "cy", _HEARD.words), ("other words\nhere", "en", _HEARD.words)])
        self.assertEqual(notes, _ALIGNED_NOTES)
        info = files["ultrasinger_parameter.info"]
        self.assertIn("Pipeline:                 Remote STT (text) + local alignment", info)
        self.assertIn("Transcription:            full (for language correction)", info)
        self.assertIn("Used this run:            yes (its text made the notes)", info)

    def test_same_language_stays_on_the_transcription(self):
        calls = []
        lyrics = SimpleNamespace(duration=None, synced_lyrics="[00:01.00] hello world", plain_lyrics="hello world")
        notes, out, _ = _run(
            _process_data(), _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5), self._transcribe,
            early_lyrics=lyrics, fast_language=("en", 0.95), align=lambda **kwargs: [],
            plain_align=lambda **kwargs: (calls.append(kwargs), ([], PlainLyricsPlacement()))[1],
            bpm_override=60.0,
        )
        self.assertEqual(len(calls), 1)
        self.assertNotIn("Aligning the plain lyrics in the corrected language", out)
        self.assertEqual([n[0] for n in notes], ["hello ", "world "])


class TestPlainLyricsLineTimes(unittest.TestCase):
    """The transcription's words, as it placed them, give the lines of the plain lyrics their times."""

    def test_transcription_words_reach_the_plain_lyrics(self):
        heard = [("helo ", 1.0, 1.3), ("world ", 1.5, 1.8)]

        def transcribe(process_data):
            # the lyrics lookup of the transcription found plain lyrics and corrected its words
            process_data.transcription_words = list(heard)
            process_data.transcribed_data = [_syllable("hello ", 1.0, 1.3), _syllable("world ", 1.5, 1.8)]
            process_data.plain_lyrics = "hello world"
            return None, None

        mocks = {}
        notes, out, files = _run(
            _process_data(), _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5), transcribe,
            plain_align=_aligned, mocks=mocks, write_settings_info=True,
        )
        mocks["TranscribeAudio"].assert_called_once()
        call = mocks["plain_align"].call_args.kwargs
        self.assertEqual((call["plain_lyrics"], call["transcription_words"]), ("hello world", heard))
        self.assertEqual(notes, _ALIGNED_NOTES)
        info = files["ultrasinger_parameter.info"]
        self.assertIn("Pipeline:                 Reference-Lyrics-First (plain)", info)
        self.assertIn("Plain lyrics placed:      line by line (2 of 2 lines anchored by the transcription's words)",
                      info)
        self.assertIn("Transcription:            full (its text replaced by the plain lyrics)", info)


class TestTranscriptionWordsAreKept(unittest.TestCase):
    """TranscribeAudio keeps the words where the transcriber placed them, before the corrections change them."""

    def test_words_before_the_corrections(self):
        process_data = SimpleNamespace(
            process_data_paths=SimpleNamespace(cache_folder_path="cache", whisper_audio_path="audio.wav"),
            media_info=SimpleNamespace(artist="Artist", title="Song Title", language="en"),
            transcribed_data=[], synced_lyrics=None, plain_lyrics=None, check_transcription=None,
            lyrics_heard=None, lyrics_instead=None, lyrics_source=None,
        )
        heard = [_syllable("helo, ", 1.0, 1.3), _syllable("world ", 1.5, 1.8)]
        transcription = SimpleNamespace(detected_language="en", transcribed_data=heard)

        def correct(data, lyrics):
            # as the lyrics correction: the words are rewritten in place
            for word in data:
                word.word = word.word.replace("helo", "hello")
            return data, None

        lyrics = SimpleNamespace(duration=None, synced_lyrics=None, plain_lyrics="hello, world")
        with patch.object(us, "transcribe_audio", return_value=transcription), \
                patch.object(us, "check_lyrics_language", return_value=(True, None)), \
                patch("modules.Speech_Recognition.lyrics_corrector.correct_transcription_from_lyrics",
                      side_effect=correct), \
                patch.object(us, "remove_silence_from_transcription_data", side_effect=lambda path, data: data[1:]), \
                patch("modules.lrclib_client.search_lyrics", return_value=lyrics), \
                patch.object(us.settings, "lyrics_lookup", True), \
                patch.object(us.settings, "llm_correct_lyrics", False), \
                patch.object(us.settings, "hyphenation", False), \
                patch.object(us.settings, "language", None), \
                contextlib.redirect_stdout(io.StringIO()):
            us.TranscribeAudio(process_data)

        # corrected, without punctuation, and the first word removed as silence
        self.assertEqual([w.word for w in process_data.transcribed_data], ["world "])
        self.assertEqual(heard[0].word, "hello ")
        self.assertEqual(process_data.transcription_words, [("helo, ", 1.0, 1.3), ("world ", 1.5, 1.8)])

    def test_words_with_nothing_to_sing_are_left_out(self):
        """A transcription cached by an earlier version may hold the music notes Whisper writes for an
        instrumental part, and punctuation standing alone: they are no notes and no anchors."""
        process_data = SimpleNamespace(
            process_data_paths=SimpleNamespace(cache_folder_path="cache", whisper_audio_path="audio.wav"),
            media_info=SimpleNamespace(artist="Artist", title="Song Title", language="fr"),
            transcribed_data=[], synced_lyrics=None, plain_lyrics=None, check_transcription=None,
            lyrics_heard=None, lyrics_instead=None, lyrics_source=None,
        )
        cached = [_syllable("♪ ", 0.0, 0.1), _syllable("Pourquoi ", 1.0, 1.5), _syllable("? ", 1.6, 1.7),
                  _syllable("- ", 1.8, 1.9), _syllable("toi ", 2.0, 2.4), _syllable("♪ ", 2.5, 2.6)]
        transcription = SimpleNamespace(detected_language="fr", transcribed_data=cached)
        with patch.object(us, "transcribe_audio", return_value=transcription), \
                patch.object(us, "remove_silence_from_transcription_data", side_effect=lambda path, data: data), \
                patch.object(us.settings, "lyrics_lookup", False), \
                patch.object(us.settings, "llm_correct_lyrics", False), \
                patch.object(us.settings, "hyphenation", False), \
                patch.object(us.settings, "language", None), \
                contextlib.redirect_stdout(io.StringIO()):
            us.TranscribeAudio(process_data)

        self.assertEqual([w.word for w in process_data.transcribed_data], ["Pourquoi? ", "toi "])
        self.assertEqual(process_data.transcription_words, [("Pourquoi? ", 1.0, 1.5), ("toi ", 2.0, 2.4)])


    def test_the_transcription_made_to_check_synced_lyrics_is_reused(self):
        heard = [_syllable("hello ", 1.0, 1.3)]
        process_data = SimpleNamespace(
            process_data_paths=SimpleNamespace(cache_folder_path="cache", whisper_audio_path="audio.wav"),
            media_info=SimpleNamespace(artist="Artist", title="Song Title", language="en"),
            transcribed_data=[], synced_lyrics=None, plain_lyrics=None,
            lyrics_heard=None, lyrics_instead=None, lyrics_source=None,
            check_transcription=SimpleNamespace(detected_language="en", transcribed_data=heard),
        )
        with patch.object(us, "transcribe_audio") as transcribe, \
                patch.object(us, "remove_silence_from_transcription_data", side_effect=lambda path, data: data), \
                patch.object(us.settings, "lyrics_lookup", False), \
                patch.object(us.settings, "llm_correct_lyrics", False), \
                patch.object(us.settings, "hyphenation", False), \
                patch.object(us.settings, "language", None), \
                contextlib.redirect_stdout(io.StringIO()):
            us.TranscribeAudio(process_data)

        transcribe.assert_not_called()
        self.assertEqual(process_data.transcription_words, [("hello ", 1.0, 1.3)])

class TestRejectedSyncedLyricsInRun(unittest.TestCase):
    def test_info_file_says_found_but_rejected(self):
        def lyrics_check(lyrics, language, confident):
            return False, "de"

        _, _, files = _run(
            _process_data(), _pitch_track([(1.0, 1.24, "C4")], length=2.5),
            _transcription([("one ", 1.0, 1.24)], "en"),
            early_lyrics=SimpleNamespace(duration=None, synced_lyrics="[00:01.00] eins zwei", plain_lyrics=None),
            lyrics_check=lyrics_check, write_settings_info=True, bpm_override=60.0,
        )
        self.assertIn("LRCLIB synced lyrics:     found but rejected (in de, another language than the singing)",
                      files["ultrasinger_parameter.info"])


class TestRejectedSyncedLyricsAreDropped(unittest.TestCase):
    """Synced lyrics the full transcription's language rejects are not kept from the early lookup."""

    ENGLISH = ("[00:01.00] we are here today and we sing, because it is a day on which the sun is "
               "shining and we are not alone, and it is good that you are here, and we stay until the "
               "night comes and it is dark, and then we go home to the place where we are from")

    def test_early_synced_lyrics_are_cleared(self):
        process_data = self._transcribe(SimpleNamespace(duration=None, synced_lyrics=self.ENGLISH, plain_lyrics=None),
                                        synced_lyrics=self.ENGLISH)
        self.assertIsNone(process_data.synced_lyrics)
        self.assertEqual(process_data.synced_lyrics_other_language, "en")

    def test_early_plain_lyrics_are_cleared(self):
        plain = self.ENGLISH.replace("[00:01.00] ", "")
        process_data = self._transcribe(SimpleNamespace(duration=None, synced_lyrics=None, plain_lyrics=plain), plain_lyrics=plain)
        self.assertIsNone(process_data.plain_lyrics)

    def _transcribe(self, lyrics, synced_lyrics=None, plain_lyrics=None):
        """TranscribeAudio with lyrics kept by the early lookup; the transcription detects German."""
        process_data = SimpleNamespace(
            process_data_paths=SimpleNamespace(cache_folder_path="cache", whisper_audio_path="audio.wav"),
            media_info=SimpleNamespace(artist="Artist", title="Song Title", language="en"),
            transcribed_data=[], synced_lyrics=synced_lyrics, plain_lyrics=plain_lyrics, check_transcription=None,
            lyrics_heard=None, lyrics_instead=None, lyrics_source=None,
        )
        transcription = SimpleNamespace(detected_language="de", transcribed_data=[])
        with patch.object(us, "transcribe_audio", return_value=transcription), \
                patch.object(us, "remove_silence_from_transcription_data", side_effect=lambda path, data: data), \
                patch("modules.lrclib_client.search_lyrics", return_value=lyrics), \
                patch.object(us.settings, "lyrics_lookup", True), \
                patch.object(us.settings, "llm_correct_lyrics", False), \
                patch.object(us.settings, "hyphenation", False), \
                patch.object(us.settings, "language", None), \
                contextlib.redirect_stdout(io.StringIO()):
            us.TranscribeAudio(process_data)
        return process_data


class TestInputKinds(unittest.TestCase):
    """InitProcessData: what the input (-i) is and what --ignore_audio needs."""

    def _init(self, input_name, ignore_audio=False, create=True):
        """InitProcessData for the input ``input_name``: a link, or a file made in a temporary folder."""
        def must_not_run(*args, **kwargs):
            raise AssertionError("this input must not be handled as audio or video")

        parsed = ("Artist - Song Title", "out", "out/Artist - Song Title.mp3", object(), "mp3")
        downloaded = ("Artist - Song Title", "out", "out/Artist - Song Title.mp3",
                      us.MediaInfo(title="Song Title", artist="Artist"))
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as folder:
            input_path = input_name
            if not us.is_url(input_name):
                input_path = os.path.join(folder, input_name)
                if create:
                    open(input_path, "wb").close()
            with patch.object(us.settings, "input_file_path", input_path), \
                    patch.object(us.settings, "ignore_audio", ignore_audio), \
                    patch.object(us.settings, "input_file_is_ultrastar_txt", False), \
                    patch.object(us.settings, "output_folder_path", "out"), \
                    patch.object(us.settings, "video_url", None), \
                    patch.object(us, "parse_ultrastar_txt", return_value=parsed), \
                    patch.object(us, "from_ultrastar_txt", return_value=_process_data()), \
                    patch.object(us, "download_from_video_platform", return_value=downloaded) as download, \
                    patch.object(us, "search_musicbrainz", side_effect=must_not_run), \
                    contextlib.redirect_stdout(out):
                try:
                    process_data = us.InitProcessData()
                    return process_data, us.settings.ignore_audio, download, out.getvalue()
                finally:
                    self.output = out.getvalue()

    def test_txt_in_upper_case_is_an_ultrastar_txt(self):
        process_data, ignore_audio, _, _ = self._init("SONG.TXT")
        self.assertTrue(ignore_audio)
        self.assertEqual(process_data.basename, "Artist - Song Title")

    def test_http_link_is_downloaded(self):
        _, _, download, _ = self._init("http://example.invalid/watch?v=1")
        download.assert_called_once()

    def test_http_link_is_downloaded_within_the_limits(self):
        with patch.object(us.settings, "download_rate_limit", 500), \
                patch.object(us.settings, "max_video_resolution", 720), \
                patch.object(us.settings, "max_video_fps", 30):
            _, _, download, _ = self._init("http://example.invalid/watch?v=1")
        self.assertEqual(download.call_args.kwargs["limits"], us.DownloadLimits(500, 720, 30))

    def test_ignore_audio_needs_a_txt(self):
        for name in ("song.mp3", "https://example.invalid/watch?v=1"):
            with self.subTest(name=name), self.assertRaises(SystemExit) as stop:
                self._init(name, ignore_audio=True)
            self.assertEqual(stop.exception.code, 1)
            self.assertIn("--ignore_audio needs an UltraStar .txt", self.output)

    def test_missing_input_file(self):
        for name in ("song.mp3", "song.txt"):
            with self.subTest(name=name), self.assertRaises(SystemExit) as stop:
                self._init(name, create=False)
            self.assertEqual(stop.exception.code, 1)
            self.assertIn("does not exist", self.output)


class TestVideoUrlNaming(unittest.TestCase):
    """--video_url: the song is looked up and named by the video, not by the temporary file name."""

    _URL = "https://example.invalid/watch?v=abc"

    def _init(self, folder, info, file_length=None):
        """InitProcessData for the file "_intercepted_abc.mp3" of ``folder`` and the --video_url
        video, whose extraction gives ``info``; the file is ``file_length`` seconds long. Returns the
        process data, the song folder, the song lookup, the extraction and the file's length lookup."""
        from modules.musicbrainz_client import SongInfo

        source = os.path.join(folder, "_intercepted_abc.mp3")
        with open(source, "wb") as file:
            file.write(b"audio")
        out_root = os.path.join(folder, "out")
        os.makedirs(out_root)
        with patch.object(us.settings, "input_file_path", source), \
                patch.object(us.settings, "output_folder_path", out_root), \
                patch.object(us.settings, "ignore_audio", False), \
                patch.object(us.settings, "input_file_is_ultrastar_txt", False), \
                patch.object(us.settings, "video_url", self._URL), \
                patch("yt_dlp.YoutubeDL") as ydl, \
                patch.object(us, "get_duration", return_value=file_length) as length_of_file, \
                patch.object(us, "search_musicbrainz",
                             side_effect=lambda title, artist, duration=None: SongInfo(title=title, artist=artist)) \
                as lookup, \
                patch.object(us, "is_video_file", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            extract_info = ydl.return_value.__enter__.return_value.extract_info
            extract_info.return_value = info
            process_data = us.InitProcessData()
            return process_data, us.settings.output_folder_path, lookup, extract_info, length_of_file

    def test_folder_files_and_lookup_follow_the_video(self):
        info = {"title": "Artist - Song: Title? (Official Video)", "artist": "Artist", "track": "Song: Title?",
                "language": "en", "duration": 238}
        url = self._URL
        with tempfile.TemporaryDirectory() as folder:
            process_data, song_folder, lookup, extract_info, length_of_file = self._init(folder, info)

            # The title and the length of the video come from one extraction
            extract_info.assert_called_once_with(url, download=False)
            lookup.assert_called_once_with("Song: Title?", "Artist", duration=238)
            length_of_file.assert_not_called()
            basename = us.os_helper.sanitize_filename("Artist - Song: Title?")
            self.assertEqual(process_data.basename, basename)
            self.assertEqual(song_folder, os.path.join(folder, "out", basename))
            audio = os.path.join(song_folder, f"{basename}.mp3")
            self.assertEqual(process_data.process_data_paths.audio_output_file_path, audio)
            self.assertTrue(os.path.isfile(audio))
            self.assertEqual((process_data.media_info.artist, process_data.media_info.title),
                             ("Artist", "Song: Title?"))
            self.assertEqual((process_data.media_info.language, process_data.media_info.video_url), ("en", url))

    def test_the_length_of_the_file_without_that_of_the_video(self):
        for info in ({"title": "Artist - Title - Chapter 2"}, {"title": "Artist - Title - Chapter 2", "duration": 0}):
            with self.subTest(info=info), tempfile.TemporaryDirectory() as folder:
                _, _, lookup, extract_info, length_of_file = self._init(folder, info, file_length=238.5)
                extract_info.assert_called_once()
                length_of_file.assert_called_once_with(os.path.join(folder, "_intercepted_abc.mp3"))
                lookup.assert_called_once_with("Title - Chapter 2", "Artist", duration=238.5)


class TestCoverOfLiveVideoUrl(unittest.TestCase):
    """--video_url of a live recording: as for its download, the cover is the video's thumbnail,
    and the cover from the song database only when there is no thumbnail."""

    _URL = "https://example.invalid/watch?v=abc"
    _THUMBNAIL = "https://thumb.example/t.jpg"

    def _init(self, info, thumbnail=_THUMBNAIL):
        """InitProcessData for a local file and the --video_url video of ``info``, whose song has a
        cover in the song database; ``thumbnail``: what the thumbnail download gives. Returns the
        process data, the covers saved from the song database and the thumbnail download."""
        from modules.musicbrainz_client import SongInfo

        saved = []
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "_intercepted_abc.mp3")
            with open(source, "wb") as file:
                file.write(b"audio")
            with patch.object(us.settings, "input_file_path", source), \
                    patch.object(us.settings, "output_folder_path", folder), \
                    patch.object(us.settings, "ignore_audio", False), \
                    patch.object(us.settings, "input_file_is_ultrastar_txt", False), \
                    patch.object(us.settings, "video_url", self._URL), \
                    patch("modules.Audio.video_platform.get_video_info", return_value=info), \
                    patch.object(us, "search_musicbrainz", return_value=SongInfo(
                        title="Title", artist="Artist", cover_url="https://cover.example/c.jpg",
                        cover_image_data=b"cover")), \
                    patch.object(us, "save_image", side_effect=lambda data, name, folder: saved.append(data)), \
                    patch("modules.Audio.video_platform.download_thumbnail", return_value=thumbnail) as download, \
                    patch.object(us, "is_video_file", return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                process_data = us.InitProcessData()
                song_folder = us.settings.output_folder_path
        return process_data, saved, download, song_folder

    def test_live_by_the_description(self):
        process_data, saved, download, song_folder = self._init(
            {"title": "Artist - Title", "description": "Recorded live at Some Hall.", "duration": 200,
             "thumbnail": "https://thumb.example/info.jpg"})
        media = process_data.media_info
        self.assertEqual(media.title, "Title (Live)")
        self.assertEqual((media.cover_url, media.cover_from_video), (self._THUMBNAIL, True))
        self.assertEqual(saved, [])
        self.assertEqual(download.call_args.args[:3], (self._URL, "Artist - Title (Live)", song_folder))
        # the thumbnail of the video's info: the video's info is not extracted again
        self.assertEqual(download.call_args.kwargs["thumbnail_url"], "https://thumb.example/info.jpg")
        self.assertEqual((media.video_thumbnail_url, media.video_thumbnail_tried),
                         ("https://thumb.example/info.jpg", True))

    def test_the_cover_from_the_song_database_without_a_thumbnail(self):
        process_data, saved, download, _ = self._init({"title": "Artist - Title (Live)", "duration": 200},
                                                      thumbnail="")
        media = process_data.media_info
        self.assertEqual((media.cover_url, media.cover_from_video), ("https://cover.example/c.jpg", False))
        self.assertEqual(saved, [b"cover"])
        download.assert_called_once()

    def test_studio_recording(self):
        process_data, saved, download, _ = self._init(
            {"title": "Artist - Title", "description": "See me live on tour!", "duration": 200})
        media = process_data.media_info
        self.assertEqual(media.title, "Title")
        self.assertEqual((media.cover_url, media.cover_from_video), ("https://cover.example/c.jpg", False))
        self.assertEqual(saved, [b"cover"])
        download.assert_not_called()
        self.assertFalse(process_data.media_info.video_thumbnail_tried)


class TestCoverOfSpecialVersion(unittest.TestCase):
    """Synced lyrics whose length does not fit the audio: the audio is a special version of the
    song, whose cover is the video's thumbnail instead of the cover from the song database."""

    _URL = "https://example.invalid/watch?v=abc"
    _THUMBNAIL = "https://thumb.example/t.jpg"

    def _swap(self, lyrics_length=236.0, audio_s=391.0, video_url=_URL, cover_from_video=False,
              thumbnail=_THUMBNAIL, synced="[00:01.00] hello world", **data):
        """_cover_of_special_version for synced lyrics of ``lyrics_length`` s and audio of
        ``audio_s`` s. Returns the media info, the thumbnail download and the console."""
        process_data = _process_data()
        process_data.media_info.cover_url = "https://cover.example/c.jpg"
        process_data.media_info.video_url = video_url
        process_data.media_info.cover_from_video = cover_from_video
        process_data.synced_lyrics = synced
        process_data.synced_lyrics_length = lyrics_length
        for name, value in data.items():
            setattr(process_data, name, value)
        out = io.StringIO()
        with patch.object(us.settings, "output_folder_path", "song folder"), \
                patch.object(us.settings, "input_file_is_ultrastar_txt", False), \
                patch.object(us, "audio_length", return_value=audio_s), \
                patch("modules.Audio.video_platform.download_thumbnail", return_value=thumbnail) as download, \
                contextlib.redirect_stdout(out):
            us._cover_of_special_version(process_data)
        return process_data.media_info, download, out.getvalue()

    def test_the_thumbnail_replaces_the_cover(self):
        media, download, out = self._swap()
        self.assertEqual((media.cover_url, media.cover_from_video), (self._THUMBNAIL, True))
        self.assertEqual(download.call_args.args[:3], (self._URL, "Artist - Song Title", "song folder"))
        self.assertIn("the synced lyrics are 236 s long, the audio 391 s: a special version, "
                      "the video's thumbnail is its cover", out)

    def test_also_when_their_text_was_used_without_their_times(self):
        media, download, _ = self._swap(synced=None, synced_lyrics_other_recording="only 7 of 62 lines",
                                        plain_lyrics="hello world")
        self.assertEqual((media.cover_url, media.cover_from_video), (self._THUMBNAIL, True))

    def test_lengths_that_fit(self):
        for lyrics_length, audio_s in ((380.0, 391.0), (None, 391.0), (236.0, None)):
            with self.subTest(lyrics_length=lyrics_length, audio_s=audio_s):
                media, download, out = self._swap(lyrics_length, audio_s)
                self.assertEqual((media.cover_url, media.cover_from_video), ("https://cover.example/c.jpg", False))
                download.assert_not_called()
                self.assertEqual(out, "")

    def test_no_synced_lyrics_used(self):
        media, download, _ = self._swap(synced=None)
        self.assertFalse(media.cover_from_video)
        download.assert_not_called()

    def test_without_a_video(self):
        media, download, _ = self._swap(video_url=None)
        self.assertEqual(media.cover_url, "https://cover.example/c.jpg")
        download.assert_not_called()

    def test_the_cover_is_the_thumbnail_already(self):
        _, download, _ = self._swap(cover_from_video=True)
        download.assert_not_called()

    def test_the_thumbnail_is_tried_once(self):
        """A song without any cover: the download tried the thumbnail already, it is not asked for again."""
        process_data = _process_data()
        process_data.media_info.video_url = self._URL
        process_data.media_info.video_thumbnail_tried = True
        process_data.synced_lyrics, process_data.synced_lyrics_length = "[00:01.00] hello world", 236.0
        with patch.object(us.settings, "input_file_is_ultrastar_txt", False), \
                patch.object(us, "audio_length", return_value=391.0), \
                patch("modules.Audio.video_platform.download_thumbnail") as download:
            us._cover_of_special_version(process_data)
        download.assert_not_called()
        # and the swap itself tries once
        media, download, _ = self._swap(thumbnail="")
        self.assertTrue(media.video_thumbnail_tried)

    def test_only_the_image_of_a_known_thumbnail_url(self):
        """The thumbnail's URL from the video's info: the image is requested, the video's info is
        not extracted again."""
        from modules.Audio import video_platform

        process_data = _process_data()
        process_data.media_info.video_url = self._URL
        process_data.media_info.video_thumbnail_url = "https://thumb.example/info.jpg"
        process_data.synced_lyrics, process_data.synced_lyrics_length = "[00:01.00] hello world", 236.0
        with patch.object(us.settings, "output_folder_path", "song folder"), \
                patch.object(us.settings, "input_file_is_ultrastar_txt", False), \
                patch.object(us, "audio_length", return_value=391.0), \
                patch.object(video_platform.yt_dlp, "YoutubeDL") as downloader_cls, \
                patch.object(video_platform, "save_image") as save, \
                contextlib.redirect_stdout(io.StringIO()):
            ydl = downloader_cls.return_value.__enter__.return_value
            ydl.urlopen.return_value.read.return_value = b"thumbnail"
            us._cover_of_special_version(process_data)
        ydl.extract_info.assert_not_called()
        ydl.urlopen.assert_called_once_with("https://thumb.example/info.jpg")
        save.assert_called_once_with(b"thumbnail", "Artist - Song Title", "song folder")
        self.assertEqual((process_data.media_info.cover_url, process_data.media_info.cover_from_video),
                         ("https://thumb.example/info.jpg", True))

    def test_the_cover_from_the_song_database_stays_without_a_thumbnail(self):
        media, download, out = self._swap(thumbnail="")
        self.assertEqual((media.cover_url, media.cover_from_video), ("https://cover.example/c.jpg", False))
        download.assert_called_once()
        self.assertEqual(download.call_args.kwargs["without"], "the cover from the song database stays")
        self.assertNotIn("a special version", out)

    def test_run_swaps_the_cover_before_the_chart_is_written(self):
        """In run(): once the lyrics are settled, before the cover goes into the tags and the chart."""
        process_data = _process_data()
        process_data.media_info.video_url = self._URL
        lyrics = SimpleNamespace(duration=236.0, synced_lyrics="[00:01.00] hello world", plain_lyrics="hello world")
        mocks, chart_written = {}, []

        def thumbnail(*args, **kwargs):
            chart_written.append(mocks["CreateUltraStarTxt"].called)
            return self._THUMBNAIL

        with patch.object(us, "audio_length", return_value=391.0), \
                patch("modules.Audio.video_platform.download_thumbnail", side_effect=thumbnail):
            _, out, _ = _run(process_data, _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5),
                             TestSyncedLyricsOfAnotherRecording()._transcribe, early_lyrics=lyrics,
                             align=TestSyncedLyricsOfAnotherRecording._align([]), plain_align=_aligned, mocks=mocks)
        self.assertEqual(chart_written, [False])
        self.assertEqual((process_data.media_info.cover_url, process_data.media_info.cover_from_video),
                         (self._THUMBNAIL, True))
        self.assertIn("a special version, the video's thumbnail is its cover", out)


class TestLocalFileLookup(unittest.TestCase):
    """A local file is looked up by its name and with its length."""

    def test_the_length_of_the_file(self):
        from modules.musicbrainz_client import SongInfo

        for file_length in (238.5, None):
            with self.subTest(file_length=file_length), tempfile.TemporaryDirectory() as folder:
                source = os.path.join(folder, "Artist - Title - Chapter 2.mp3")
                with open(source, "wb") as file:
                    file.write(b"audio")
                with patch.object(us.settings, "input_file_path", source), \
                        patch.object(us.settings, "output_folder_path", folder), \
                        patch.object(us.settings, "ignore_audio", False), \
                        patch.object(us.settings, "input_file_is_ultrastar_txt", False), \
                        patch.object(us.settings, "video_url", None), \
                        patch.object(us, "get_duration", return_value=file_length) as length_of_file, \
                        patch.object(us, "search_musicbrainz",
                                     return_value=SongInfo(title="Title", artist="Artist")) as lookup, \
                        patch.object(us, "is_video_file", return_value=False), \
                        contextlib.redirect_stdout(io.StringIO()):
                    process_data = us.InitProcessData()
                length_of_file.assert_called_once_with(source)
                lookup.assert_called_once_with("Title - Chapter 2", "Artist", duration=file_length)
                self.assertEqual(process_data.basename, "Artist - Title")


class TestRepitchRun(unittest.TestCase):
    """A re-pitch of a real UltraStar txt input: no transcription, and nothing in the info file about one."""

    def test_info_file(self):
        with tempfile.TemporaryDirectory() as folder:
            chart = os.path.join(folder, "chart.txt")
            with open(chart, "w", encoding="utf-8") as file:
                file.writelines(f"{line}\n" for line in (
                    "#TITLE:Song Title", "#ARTIST:Artist", "#LANGUAGE:German", "#MP3:song.mp3",
                    "#BPM:300", "#GAP:1000", ": 0 4 60 eins ", "E"))
            open(os.path.join(folder, "song.mp3"), "wb").close()
            # what InitProcessData makes of the txt (the chart's language tag is not read);
            # no online cover lookup for the chart without a cover
            with patch.object(us.settings, "input_file_path", chart), \
                    patch.object(us.settings, "output_folder_path", os.path.join(folder, "out")), \
                    patch.object(us.settings, "ignore_audio", False), \
                    patch.object(us.settings, "input_file_is_ultrastar_txt", False), \
                    patch("modules.Ultrastar.ultrastar_parser._fetch_musicbrainz_cover") as cover_lookup, \
                    contextlib.redirect_stdout(io.StringIO()):
                process_data = us.InitProcessData()
            cover_lookup.assert_called_once()
            mocks = {}
            notes, _, files = _run(process_data, _pitch_track([(1.0, 1.4, "C4")], length=2.5),
                                   _transcription([], "en"), mocks=mocks, ignore_audio=True,
                                   input_file_is_ultrastar_txt=True, bpm_override=None,
                                   write_settings_info=True)
        mocks["TranscribeAudio"].assert_not_called()
        self.assertEqual([n[0].strip() for n in notes], ["eins"])
        info = files["ultrasinger_parameter.info"]
        self.assertIn("Language detection:        unknown", info)
        self.assertIn("Language used:             (none)", info)
        self.assertNotIn("Transcriber language hint", info)


class TestPlotWithoutSeparation(unittest.TestCase):
    """--plot with --disable_separation and --disable_karaoke: no vocals stem to plot, the run goes on."""

    def test_plots_what_exists(self):
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as audio_folder:
            processing_audio = os.path.join(audio_folder, "song_mute.wav")
            sf.write(processing_audio, 0.1 * np.sin(np.linspace(0, 2000, 16000)), 16000)
            process_data = _process_data()
            # what the separation step names, although it made no stems
            process_data.process_data_paths.vocals_audio_file_path = os.path.join(audio_folder, "vocals.wav")
            notes, out, _ = _run(
                process_data, _pitch_track([(1.0, 1.24, "C4")], length=2.5),
                _transcription([("one ", 1.0, 1.24)], "en"), bpm_override=60.0,
                audio_paths=("song_mono.wav", processing_audio), create_plot=True,
                use_separated_vocal=False, create_karaoke=False,
            )
        self.assertEqual([n[0] for n in notes], ["one "])
        self.assertIn("Creating plot: processing audio", out)
        self.assertNotIn("Creating plot: vocals.wav", out)


class TestOutputWithoutSeparation(unittest.TestCase):
    """--disable_separation with --disable_karaoke: no stems, and the TXT names none."""

    def _write(self, separated):
        converted = []

        def convert(source, target):
            converted.append(target)
            open(target, "wb").close()

        with tempfile.TemporaryDirectory() as folder:
            process_data = _process_data(language="en")
            process_data.media_info.bpm = 120.0
            process_data.midi_segments = [us.MidiSegment("C4", 1.0, 1.5, "hel"), us.MidiSegment("D4", 1.5, 2.0, "lo ")]
            process_data.process_data_paths.vocals_audio_file_path = os.path.join(folder, "missing", "vocals.wav")
            process_data.process_data_paths.instrumental_audio_file_path = os.path.join(folder, "missing", "no_vocals.wav")
            with patch.object(us.settings, "output_folder_path", folder), \
                    patch.object(us.settings, "use_separated_vocal", separated), \
                    patch.object(us.settings, "create_karaoke", separated), \
                    patch.object(us.settings, "format_version", FormatVersion.V1_2_0), \
                    patch.object(us.settings, "ignore_audio", False), \
                    patch.object(us.settings, "write_metadata_tags", False), \
                    patch.object(us.settings, "calculate_score", False), \
                    patch.object(us, "convert_audio_format", side_effect=convert), \
                    contextlib.redirect_stdout(io.StringIO()):
                _, _, txt_path = us.CreateUltraStarTxt(process_data)
            with open(txt_path, encoding="utf-8") as file:
                return converted, file.read()

    def test_no_stems_without_separation(self):
        converted, txt = self._write(separated=False)
        self.assertEqual(converted, [])
        self.assertNotIn("#VOCALS:", txt)
        self.assertNotIn("#INSTRUMENTAL:", txt)
        self.assertIn("#MP3:Artist - Song Title.mp3", txt)

    def test_stems_with_separation(self):
        converted, txt = self._write(separated=True)
        self.assertEqual(len(converted), 2)
        self.assertIn("#VOCALS:Artist - Song Title [Vocals].mp3", txt)
        self.assertIn("#INSTRUMENTAL:Artist - Song Title [Instrumental].mp3", txt)


if __name__ == "__main__":
    unittest.main()


def _numbered(n, first=0):
    """Words "word<first>" ... of n words, ten to a line."""
    words = [f"word{k}" for k in range(first, first + n)]
    return "\n".join(" ".join(words[k:k + 10]) for k in range(0, n, 10))


def _lrc(n, first=0):
    return "\n".join(f"[00:{10 + k:02d}.00] {line}" for k, line in enumerate(_numbered(n, first).split("\n")))


class TestLyricsTheSingingHas(unittest.TestCase):
    """Found lyrics of LRCLIB that have few of the transcribed words are replaced by the lyrics of the
    second lyrics database when these have clearly more of them."""

    # 40 words heard; the lyrics of LRCLIB have none of them (another song)
    _HEARD = [(f"word{k}", 10.0 + k * 0.5, 10.3 + k * 0.5) for k in range(40)]
    _LRCLIB = SimpleNamespace(duration=300.0, synced_lyrics=_lrc(40, first=100), plain_lyrics=_numbered(40, first=100))

    def _run_with(self, other, early=None, transcribe=None, heard=None, **kwargs):
        heard = self._HEARD if heard is None else heard
        mocks = {}
        result = TranscriptionResult([us.TranscribedData(word=w, start=a, end=b) for w, a, b in heard], "en")
        check = SyncedLyricsCheck(lines=4, checked=True, confirmed=4)
        aligned = lambda **kw: ([us.MidiSegment(n.note, n.start, n.end, n.word) for n in _ALIGNED], check)  # noqa: E731
        with patch.object(us, "transcribe_audio", return_value=result), \
                patch.object(us, "audio_length", return_value=300.0), \
                patch("modules.lrclib_client.find_on_second_db",
                      **({"side_effect": other} if isinstance(other, Exception) else {"return_value": other})
                      ) as second_db, \
                patch("modules.Speech_Recognition.reference_lyrics_aligner."
                      "create_midi_segments_from_checked_synced_lyrics", side_effect=aligned) as checked:
            notes, out, files = _run(
                _process_data(), _pitch_track([(1.0, 1.4, "C4"), (1.5, 2.0, "D4")], length=2.5),
                transcribe or _transcription([("one ", 1.0, 1.24)], "en"),
                early_lyrics=self._LRCLIB if early is None else early, plain_align=_aligned, mocks=mocks,
                write_settings_info=True, check_synced_lyrics=True, **kwargs)
        return notes, out, files.get("ultrasinger_parameter.info", ""), mocks, second_db, checked

    def test_synced_lyrics_that_fit_replace_them(self):
        other = SimpleNamespace(source="music.163.com", duration=298.0, synced_lyrics=_lrc(40),
                                plain_lyrics=_numbered(40), artist_name="Artist", track_name="Song Title",
                                instrumental=False)
        notes, out, info, mocks, second_db, checked = self._run_with(other)
        second_db.assert_called_once_with("Artist", "Song Title", 300.0)
        self.assertEqual(checked.call_args.kwargs["synced_lyrics"], other.synced_lyrics)
        mocks["TranscribeAudio"].assert_not_called()
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("the lyrics of LRCLIB have only 0% of the 40 transcribed words: "
                      "searching music.163.com for lyrics", out)
        self.assertIn("The lyrics on music.163.com have 100% of the transcribed words: taking them", out)
        # the line the GUI queue reads
        self.assertIn("Found lyrics on music.163.com: ", out)
        self.assertIn("  music.163.com synced lyrics: found\n", info)
        self.assertIn("  Lyrics heard:             LRCLIB 0% of the 40 transcribed words; "
                      "music.163.com 100%: taken\n", info)

    def test_plain_lyrics_that_fit_are_placed_by_the_words(self):
        other = SimpleNamespace(source="music.163.com", duration=None, synced_lyrics=None,
                                plain_lyrics=_numbered(40), artist_name="Artist", track_name="Song Title",
                                instrumental=False)
        notes, _, info, mocks, _, checked = self._run_with(other)
        checked.assert_not_called()
        call = mocks["plain_align"].call_args.kwargs
        self.assertEqual((call["plain_lyrics"], call["transcription_words"]), (other.plain_lyrics, self._HEARD))
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("Pipeline:                 Reference-Lyrics-First (plain)", info)
        self.assertIn("  music.163.com plain lyrics: found\n", info)

    def test_lyrics_not_clearly_better_are_not_taken(self):
        # 8 of the 40 words: 20 %, not 25 % more than the 0 % of LRCLIB
        other = SimpleNamespace(source="music.163.com", duration=300.0, synced_lyrics=_lrc(8), plain_lyrics=None,
                                artist_name="Artist", track_name="Song Title", instrumental=False)
        _, out, info, _, second_db, checked = self._run_with(other)
        second_db.assert_called_once()
        self.assertEqual(checked.call_args.kwargs["synced_lyrics"], self._LRCLIB.synced_lyrics)
        self.assertIn("The lyrics on music.163.com have 20% of the transcribed words: keeping the ones of LRCLIB",
                      out)
        self.assertNotIn("Found lyrics on music.163.com", out)
        self.assertIn("  LRCLIB synced lyrics:     found\n", info)
        self.assertIn("Lyrics heard:             LRCLIB 0% of the 40 transcribed words; music.163.com 20%: "
                      "kept the ones of LRCLIB", info)

    def test_none_on_the_second_database(self):
        _, out, info, _, second_db, checked = self._run_with(None)
        second_db.assert_called_once()
        self.assertEqual(checked.call_args.kwargs["synced_lyrics"], self._LRCLIB.synced_lyrics)
        self.assertIn("music.163.com has no lyrics of the song: keeping the ones of LRCLIB", out)
        self.assertIn("Lyrics heard:             LRCLIB 0% of the 40 transcribed words; "
                      "none found on music.163.com", info)

    def test_lyrics_that_fit_are_kept_without_asking(self):
        fitting = SimpleNamespace(duration=300.0, synced_lyrics=_lrc(40), plain_lyrics=_numbered(40))
        _, out, info, _, second_db, checked = self._run_with(None, early=fitting)
        second_db.assert_not_called()
        self.assertEqual(checked.call_args.kwargs["synced_lyrics"], fitting.synced_lyrics)
        self.assertNotIn("searching music.163.com", out)
        self.assertIn("Lyrics heard:             LRCLIB 100% of the 40 transcribed words\n", info)

    def test_not_compared_with_few_transcribed_words(self):
        _, _, info, _, second_db, _ = self._run_with(None, heard=self._HEARD[:29])
        second_db.assert_not_called()
        self.assertNotIn("Lyrics heard", info)

    def test_not_for_lyrics_of_the_second_database(self):
        found = SimpleNamespace(source="music.163.com", duration=300.0, synced_lyrics=_lrc(40, first=100),
                                plain_lyrics=None)
        _, _, info, _, second_db, _ = self._run_with(None, early=found)
        second_db.assert_not_called()
        self.assertNotIn("Lyrics heard", info)

    def test_plain_lyrics_found_by_the_transcription(self):
        heard = self._HEARD

        def transcribe(process_data):
            # LRCLIB has plain lyrics only, of another song
            process_data.transcription_words = list(heard)
            process_data.transcribed_data = [_syllable(*word) for word in heard]
            process_data.media_info.language = "en"
            process_data.plain_lyrics = _numbered(40, first=100)
            process_data.lyrics_source = "LRCLIB"
            return None, None

        other = SimpleNamespace(source="music.163.com", duration=300.0, synced_lyrics=_lrc(40), plain_lyrics=None,
                                artist_name="Artist", track_name="Song Title", instrumental=False)
        notes, _, info, mocks, second_db, _ = self._run_with(
            other, early=SimpleNamespace(duration=None, synced_lyrics=None, plain_lyrics=None), transcribe=transcribe)
        second_db.assert_called_once()
        call = mocks["plain_align"].call_args.kwargs
        # the text of its synced lyrics, placed by the transcribed words
        self.assertEqual((call["plain_lyrics"], call["transcription_words"]), (_numbered(40), heard))
        self.assertEqual(notes, _ALIGNED_NOTES)
        self.assertIn("  music.163.com plain lyrics: the text of the synced lyrics\n", info)
        self.assertIn("music.163.com 100%: taken", info)

    def test_failed_search_keeps_the_lyrics(self):
        _, out, _, _, _, checked = self._run_with(OSError("offline"))
        self.assertEqual(checked.call_args.kwargs["synced_lyrics"], self._LRCLIB.synced_lyrics)
        self.assertIn("The search on music.163.com failed (offline): keeping the lyrics of LRCLIB", out)

    def test_not_in_languages_without_spaces(self):
        # their lines are no words: none would pair with the transcribed ones
        _, _, info, _, second_db, _ = self._run_with(None, language="ja")
        second_db.assert_not_called()
        self.assertNotIn("Lyrics heard", info)

    def test_replaced_before_they_correct_the_transcription(self):
        heard = [_syllable(w, a, b) for w, a, b in self._HEARD]
        process_data = SimpleNamespace(
            process_data_paths=SimpleNamespace(cache_folder_path="cache", whisper_audio_path="audio.wav"),
            media_info=SimpleNamespace(artist="Artist", title="Song Title", language="en"),
            transcribed_data=[], synced_lyrics=None, plain_lyrics=None, check_transcription=None,
            lyrics_heard=None, lyrics_instead=None, lyrics_source=None, plain_lyrics_from_synced=False,
            synced_lyrics_other_recording=None, synced_lyrics_length=None,
        )
        transcription = SimpleNamespace(detected_language="en", transcribed_data=heard)
        found = SimpleNamespace(duration=None, synced_lyrics=None, plain_lyrics=_numbered(40, first=100),
                                source="LRCLIB")
        other = SimpleNamespace(source="music.163.com", duration=300.0, synced_lyrics=None,
                                plain_lyrics=_numbered(40), artist_name="Artist", track_name="Song Title",
                                instrumental=False)
        with patch.object(us, "transcribe_audio", return_value=transcription), \
                patch.object(us, "audio_length", return_value=300.0), \
                patch.object(us, "check_lyrics_language", return_value=(True, None)), \
                patch("modules.Speech_Recognition.lyrics_corrector.correct_transcription_from_lyrics",
                      side_effect=lambda data, lyrics: (data, None)) as correct, \
                patch.object(us, "remove_silence_from_transcription_data", side_effect=lambda path, data: data), \
                patch("modules.lrclib_client.search_lyrics", return_value=found), \
                patch("modules.lrclib_client.find_on_second_db", return_value=other), \
                patch.object(us.settings, "lyrics_lookup", True), \
                patch.object(us.settings, "llm_correct_lyrics", False), \
                patch.object(us.settings, "hyphenation", False), \
                patch.object(us.settings, "language", None), \
                contextlib.redirect_stdout(io.StringIO()):
            us.TranscribeAudio(process_data)
        self.assertEqual(correct.call_args.args[1], other.plain_lyrics)
        self.assertEqual(process_data.plain_lyrics, other.plain_lyrics)
        self.assertEqual(process_data.lyrics_source, "music.163.com")
        self.assertIs(process_data.lyrics_instead, other)
