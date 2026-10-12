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
         align=None, **settings_values):
    """Run run() with fakes. Returns the notes it wrote, its output and the output folder's files."""
    written = {}

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
                ("CreateProcessAudio", lambda data: ("vocals.wav", "vocals_mute.wav")),
                ("detect_key_from_audio", lambda path: ("C", "major")),
                ("pitch_audio", lambda paths: pitched_data),
                ("TranscribeAudio", transcribe),
                ("check_lyrics_language", lambda lyrics, language, confident: (True, None)),
                ("CreateUltraStarTxt", create_txt),
                ("create_sheet", lambda *args: None),
                ("finish_cache_folder", lambda path: None),
                ("print_support", lambda: None),
        ):
            stack.enter_context(patch.object(us, name, side_effect=fake))
        stack.enter_context(patch("modules.lrclib_client.search_lyrics", return_value=early_lyrics))
        stack.enter_context(patch("modules.Speech_Recognition.Whisper.detect_language_with_confidence",
                                  return_value=fast_language))
        stack.enter_context(patch(
            "modules.Speech_Recognition.reference_lyrics_aligner.create_midi_segments_from_reference_lyrics",
            side_effect=align))
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
            early_lyrics=SimpleNamespace(synced_lyrics="[00:01.00] hello world", plain_lyrics=None),
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

    def _info(self, ignore_audio=False, **kwargs):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(us.settings, "ignore_audio", ignore_audio), \
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

    def test_remote_stt_after_the_transcription(self):
        info = self._info(reference_first_used=True, remote_stt_used=True)
        self.assertIn("Transcription:            full (its text replaced by the remote STT text)", info)

    def test_plain_lyrics_after_failed_synced_alignment(self):
        info = self._info(reference_first_used=True, has_synced_lyrics=True, plain_lyrics_used=True)
        self.assertIn("Pipeline:                 Reference-Lyrics-First (plain)", info)
        self.assertIn("LRCLIB synced lyrics:     found (but alignment failed)", info)
        self.assertIn("Transcription:            full (its text replaced by the plain lyrics)", info)

    def test_synced(self):
        info = self._info(reference_first_used=True, has_synced_lyrics=True, whisper_skipped=True)
        self.assertIn("Pipeline:                 Reference-Lyrics-First (synced)", info)

    def test_repitch(self):
        info = self._info(ignore_audio=True)
        self.assertIn("Pipeline:                 Re-pitch (notes and lyrics of the UltraStar txt)", info)
        self.assertNotIn("Standard transcription", info)

    def test_language_hint_only_with_language_option(self):
        info = self._info(lang_source="video_platform", detected_language="en")
        self.assertIn("Transcriber language hint: no (auto-detect)", info)


class TestRejectedSyncedLyricsAreDropped(unittest.TestCase):
    """Synced lyrics the full transcription's language rejects are not kept from the early lookup."""

    ENGLISH = ("[00:01.00] we are here today and we sing, because it is a day on which the sun is "
               "shining and we are not alone, and it is good that you are here, and we stay until the "
               "night comes and it is dark, and then we go home to the place where we are from")

    def test_early_synced_lyrics_are_cleared(self):
        process_data = SimpleNamespace(
            process_data_paths=SimpleNamespace(cache_folder_path="cache", whisper_audio_path="audio.wav"),
            media_info=SimpleNamespace(artist="Artist", title="Song Title", language="en"),
            transcribed_data=[], synced_lyrics=self.ENGLISH, plain_lyrics=None,
        )
        transcription = SimpleNamespace(detected_language="de", transcribed_data=[])
        lyrics = SimpleNamespace(synced_lyrics=self.ENGLISH, plain_lyrics=None)
        with patch.object(us, "transcribe_audio", return_value=transcription), \
                patch.object(us, "remove_silence_from_transcription_data", side_effect=lambda path, data: data), \
                patch("modules.lrclib_client.search_lyrics", return_value=lyrics), \
                patch.object(us.settings, "lyrics_lookup", True), \
                patch.object(us.settings, "llm_correct_lyrics", False), \
                patch.object(us.settings, "hyphenation", False), \
                patch.object(us.settings, "language", None), \
                contextlib.redirect_stdout(io.StringIO()):
            us.TranscribeAudio(process_data)
        self.assertIsNone(process_data.synced_lyrics)


class TestInputKinds(unittest.TestCase):
    """InitProcessData: what the input (-i) is and what --ignore_audio needs."""

    def _init(self, input_path, ignore_audio=False):
        def must_not_run(*args, **kwargs):
            raise AssertionError("this input must not be handled as audio or video")

        parsed = ("Artist - Song Title", "out", "out/Artist - Song Title.mp3", object(), "mp3")
        downloaded = ("Artist - Song Title", "out", "out/Artist - Song Title.mp3",
                      us.MediaInfo(title="Song Title", artist="Artist"))
        out = io.StringIO()
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
            process_data = us.InitProcessData()
            return process_data, us.settings.ignore_audio, download, out.getvalue()

    def test_txt_in_upper_case_is_an_ultrastar_txt(self):
        process_data, ignore_audio, _, _ = self._init("songs/SONG.TXT")
        self.assertTrue(ignore_audio)
        self.assertEqual(process_data.basename, "Artist - Song Title")

    def test_http_link_is_downloaded(self):
        _, _, download, _ = self._init("http://example.invalid/watch?v=1")
        download.assert_called_once()

    def test_ignore_audio_needs_a_txt(self):
        for path in ("song.mp3", "https://example.invalid/watch?v=1"):
            with self.subTest(path=path), self.assertRaises(SystemExit) as stop:
                self._init(path, ignore_audio=True)
            self.assertEqual(stop.exception.code, 1)


class TestVideoUrlNaming(unittest.TestCase):
    """--video_url: the song is looked up and named by the video, not by the temporary file name."""

    def test_folder_files_and_lookup_follow_the_video(self):
        from modules.musicbrainz_client import SongInfo

        url = "https://example.invalid/watch?v=abc"
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "_intercepted_abc.mp3")
            with open(source, "wb") as file:
                file.write(b"audio")
            out_root = os.path.join(folder, "out")
            os.makedirs(out_root)
            video = ("Artist", "Song: Title?", "Artist - Song: Title? (Official Video)", "en")
            with patch.object(us.settings, "input_file_path", source), \
                    patch.object(us.settings, "output_folder_path", out_root), \
                    patch.object(us.settings, "ignore_audio", False), \
                    patch.object(us.settings, "video_url", url), \
                    patch("modules.Audio.video_platform.get_video_title", return_value=video), \
                    patch.object(us, "search_musicbrainz",
                                 side_effect=lambda title, artist: SongInfo(title=title, artist=artist)) as lookup, \
                    patch.object(us, "is_video_file", return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                process_data = us.InitProcessData()
                song_folder = us.settings.output_folder_path

            lookup.assert_called_once_with("Song: Title?", "Artist")
            basename = us.os_helper.sanitize_filename("Artist - Song: Title?")
            self.assertEqual(process_data.basename, basename)
            self.assertEqual(song_folder, os.path.join(out_root, basename))
            audio = os.path.join(song_folder, f"{basename}.mp3")
            self.assertEqual(process_data.process_data_paths.audio_output_file_path, audio)
            self.assertTrue(os.path.isfile(audio))
            self.assertEqual((process_data.media_info.artist, process_data.media_info.title),
                             ("Artist", "Song: Title?"))
            self.assertEqual((process_data.media_info.language, process_data.media_info.video_url), ("en", url))


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
