"""Tests for the Qwen3-ASR transcriber (qwen3_asr.py); the model itself is never loaded."""

import sys
import types
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from src.modules.Speech_Recognition import qwen3_asr
from src.modules.Speech_Recognition.qwen3_asr import (
    QwenFailed,
    QwenUnavailable,
    TranscriberChoice,
    choose_transcriber,
    majority_language,
    transcribed_language,
    song_language,
    transcribe_with_qwen3,
)

GIB = 2 ** 30


class TestChooseTranscriber(unittest.TestCase):
    def _choose(self, free_gib, installed=True, requested="auto", device="cuda", language=None):
        with patch.object(qwen3_asr, "_qwen_asr_installed", return_value=installed), \
                patch.object(qwen3_asr, "free_gpu_memory", return_value=None if free_gib is None else free_gib * GIB):
            return choose_transcriber(requested, device, language)

    def test_whisper_when_asked_for(self):
        choice = self._choose(8, requested="whisper")
        self.assertEqual((choice.name, choice.reason), ("whisper", "selected"))

    def test_whisper_without_a_gpu(self):
        for requested in ("auto", "qwen3"):
            with self.subTest(requested=requested):
                self.assertEqual(self._choose(8, requested=requested, device="cpu").name, "whisper")
        self.assertEqual(self._choose(None).name, "whisper")

    def test_whisper_when_not_installed(self):
        choice = self._choose(8, installed=False)
        self.assertEqual(choice.name, "whisper")
        self.assertIn("not installed", choice.reason)

    def test_installed_is_only_looked_up(self):
        # importing qwen_asr takes seconds
        for spec, installed in ((None, False), (object(), True)):
            with self.subTest(installed=installed), \
                    patch.object(qwen3_asr.importlib.util, "find_spec", return_value=spec) as find_spec:
                self.assertIs(qwen3_asr._qwen_asr_installed(), installed)
            find_spec.assert_called_once_with("qwen_asr")

    def test_qwen3_with_enough_memory(self):
        for requested in ("auto", "qwen3"):
            with self.subTest(requested=requested):
                choice = self._choose(8, requested=requested)
                self.assertEqual((choice.name, choice.batch_size), ("qwen3", 8))
        choice = self._choose(5.5)
        self.assertEqual((choice.name, choice.batch_size), ("qwen3", 2))
        self.assertIn("5.5 GB", choice.reason)

    def test_whisper_with_too_little_memory(self):
        # smaller variants that would fit were less accurate than Whisper, so Whisper takes over
        for free_gib in (4.9, 3.5, 2.0):
            with self.subTest(free_gib=free_gib):
                choice = self._choose(free_gib)
                self.assertEqual(choice.name, "whisper")
                self.assertIn(f"only {free_gib:.1f} GB", choice.reason)
        # just below the limit, the free memory does not read as enough
        self.assertIn("only 4.9 GB", self._choose(4.96).reason)

    def test_whisper_when_the_memory_cannot_be_read(self):
        with patch.object(qwen3_asr, "_qwen_asr_installed", return_value=True), \
                patch.object(qwen3_asr, "free_gpu_memory", side_effect=RuntimeError("CUDA error")):
            choice = choose_transcriber("auto", "cuda")
        self.assertEqual(choice.name, "whisper")
        self.assertIn("could not be read", choice.reason)

    def test_language_given(self):
        self.assertEqual(self._choose(8, language="de").name, "qwen3")
        # Filipino under Whisper's code, as the rest of the pipeline uses it
        self.assertEqual(self._choose(8, language="tl").name, "qwen3")
        for requested in ("auto", "qwen3"):
            with self.subTest(requested=requested):
                choice = self._choose(8, requested=requested, language="uk")
                self.assertEqual(choice.name, "whisper")
                self.assertIn("does not transcribe the language uk", choice.reason)


class TestLanguage(unittest.TestCase):
    def test_majority_language(self):
        self.assertEqual(majority_language(["English", "German", "English", None, ""]), "en")
        self.assertEqual(majority_language(["german", "Klingon"]), "de")
        self.assertEqual(majority_language(["Filipino"]), "tl")
        self.assertEqual(majority_language([]), "en")
        self.assertIsNone(majority_language(["Klingon"], default=None))

    def test_majority_language_counts_each_language_of_a_piece(self):
        # a piece decoded in several chunks has the chunks' languages joined by commas
        self.assertEqual(majority_language(["Spanish,English", "Spanish", "English"]), "es")
        self.assertEqual(majority_language(["German, English", "German"]), "de")
        self.assertEqual(majority_language(["Klingon,Filipino"]), "tl")

    def test_the_words_decide_over_the_tags(self):
        piece = types.SimpleNamespace
        spanish = ("no sé por qué te quiero si tú no me quieres a mí y yo te espero cada noche en la "
                   "puerta de mi casa con el corazón en la mano porque eres todo lo que tengo y ya no "
                   "puedo más")
        words = spanish.split()
        results = [piece(text=" ".join(words[:12]), language="Spanish"),
                   piece(text=" ".join(words[12:24]), language="English"),
                   piece(text=" ".join(words[24:]), language="English")]
        self.assertEqual(transcribed_language(results), "es")

    def test_the_tags_name_a_language_the_words_cannot_tell_apart(self):
        piece = types.SimpleNamespace
        # Cyrillic words are read as "ru"; the tags tell Macedonian from it
        macedonian = " ".join(["Јас те сакам многу и секогаш ќе бидам со тебе"] * 5)
        self.assertEqual(transcribed_language([piece(text=macedonian, language="Macedonian")] * 3), "mk")
        # Han script is read as "zh"; the tags tell Cantonese from it
        cantonese = "我哋今晚一齊去睇戲好唔好" * 5
        self.assertEqual(transcribed_language([piece(text=cantonese, language="Cantonese")] * 3), "yue")

    def test_kanji_alone_do_not_override_a_japanese_tag(self):
        piece = types.SimpleNamespace
        kanji = "東京夜空星降雪桜咲春風" * 5
        self.assertEqual(transcribed_language([piece(text=kanji, language="Japanese")] * 3), "ja")

    def test_the_words_win_over_tags_of_another_script(self):
        piece = types.SimpleNamespace
        japanese = "わたしはあなたをずっとまっていますこころのなかで" * 5
        results = [piece(text=japanese, language="English")] * 3
        with patch("builtins.print") as printed:
            self.assertEqual(transcribed_language(results), "ja")
        # the log tells that the words overrode the tags
        self.assertIn("its words say", printed.call_args.args[0])
        self.assertIn("— taking ja", printed.call_args.args[0])

    def test_function_words_do_not_override_a_language_without_a_profile(self):
        piece = types.SimpleNamespace
        # Vietnamese "em" is a Portuguese function word, Czech shares many with Polish
        vietnamese = " ".join(["em yêu anh em nhớ anh em đợi anh em thương anh mãi mãi"] * 5)
        self.assertEqual(transcribed_language([piece(text=vietnamese, language="Vietnamese")] * 3), "vi")
        czech = " ".join(["to je na co se tak za po by mi ty jsi moje láska"] * 5)
        self.assertEqual(transcribed_language([piece(text=czech, language="Czech")] * 3), "cs")

    def test_function_words_do_not_override_a_language_known_by_its_script(self):
        piece = types.SimpleNamespace
        # Japanese in Latin letters: "no", "to", "a", "i" are function words elsewhere
        romaji = " ".join(["kimi no koe to boku no yume a i no uta to kaze no oto"] * 5)
        self.assertEqual(transcribed_language([piece(text=romaji, language="Japanese")] * 3), "ja")

    def test_the_words_decide_without_known_tags(self):
        piece = types.SimpleNamespace
        german = " ".join(["ich bin nicht mehr da und du bist nicht hier aber ich denke an dich"] * 3)
        with patch("builtins.print") as printed:
            self.assertEqual(transcribed_language([piece(text=german, language="")] * 2), "de")
        self.assertIn("name no known language", printed.call_args.args[0])
        # nothing known at all: English
        self.assertEqual(transcribed_language([piece(text="oh", language=None)]), "en")

    def test_the_tags_decide_when_the_words_leave_it_open(self):
        piece = types.SimpleNamespace
        # too few words to tell the language
        results = [piece(text="la la la", language="German"), piece(text="oh", language="German"),
                   piece(text="", language="English")]
        self.assertEqual(transcribed_language(results), "de")

    def test_given_language_wins(self):
        with patch("modules.Speech_Recognition.Whisper.detect_language_with_confidence") as detect:
            self.assertEqual(song_language("a.wav", "de"), "de")
        detect.assert_not_called()

    def test_quick_detection_only_when_confident(self):
        with patch("modules.Speech_Recognition.Whisper.detect_language_with_confidence",
                   return_value=("de", 0.9)) as detect:
            self.assertEqual(song_language("a.wav", None), "de")
        # on the CPU, so that the GPU memory found free for the model stays free
        self.assertEqual(detect.call_args.kwargs["device"], "cpu")
        with patch("modules.Speech_Recognition.Whisper.detect_language_with_confidence", return_value=("de", 0.5)):
            self.assertIsNone(song_language("a.wav", None))


class TestTranscribe(unittest.TestCase):
    """Pieces in, aligned segments out; the model is dropped afterwards."""

    def _run(self, texts, language=None, error_message=None):
        models, seen = [], {}

        class Result:
            def __init__(self, text, lang):
                self.text, self.language = text, lang

        class Model:
            def transcribe(self, audio, language=None):
                seen["pieces"], seen["language"] = len(audio), language
                if error_message is not None:
                    raise RuntimeError(error_message)
                return [Result(text, "English") for text in texts]

        def load_model(choice):
            model = Model()
            models.append(weakref.ref(model))
            return model

        def align_segments(segments, audio, lang, device, alignment_model=None, keep_numbers=False):
            seen["segments"], seen["align_language"] = segments, lang
            return []

        pieces = [(np.zeros(16000 * 30, dtype=np.float32), 0.0), (np.zeros(16000 * 25, dtype=np.float32), 30.0),
                  (np.zeros(16000 * 10, dtype=np.float32), 55.0)]
        with patch.object(qwen3_asr, "_load_model", side_effect=load_model), \
                patch.object(qwen3_asr, "audio_pieces", return_value=pieces), \
                patch.object(qwen3_asr, "song_language", return_value=language), \
                patch.object(qwen3_asr, "release_gpu_memory") as release, \
                patch("whisperx.load_audio", return_value=np.zeros(16000 * 65, dtype=np.float32)), \
                patch("modules.Speech_Recognition.Whisper.align_segments", side_effect=align_segments):
            try:
                result = transcribe_with_qwen3("a.wav", TranscriberChoice("qwen3", 2, "8.0 GB free"))
            except QwenFailed as e:
                result, seen["error"] = None, str(e)
            seen["released"] = release.called
        return result, seen, models

    def test_pieces_become_segments(self):
        result, seen, _ = self._run(["first words", "  ", "last words"])
        self.assertEqual(seen["pieces"], 3)
        # the silent piece gives no segment; the others keep their place in the song
        self.assertEqual(seen["segments"], [{"text": "first words", "start": 0.0, "end": 30.0},
                                            {"text": "last words", "start": 55.0, "end": 65.0}])
        self.assertEqual(result.detected_language, "en")

    def test_the_song_language_is_given_to_the_model(self):
        result, seen, _ = self._run(["eins", "zwei", "drei"], language="de")
        self.assertEqual((seen["language"], seen["align_language"], result.detected_language),
                         ("German", "de", "de"))

    def test_the_language_comes_from_the_words(self):
        # the model tags every piece "English" here; the words are Spanish
        words = ("no sé por qué te quiero si tú no me quieres a mí y yo te espero cada noche en la "
                 "puerta de mi casa con el corazón en la mano porque eres todo lo que tengo").split()
        result, seen, _ = self._run([" ".join(words[:10]), " ".join(words[10:20]), " ".join(words[20:])])
        self.assertEqual((seen["language"], seen["align_language"], result.detected_language), (None, "es", "es"))

    def test_the_model_is_dropped(self):
        _, seen, models = self._run(["a", "b", "c"])
        self.assertIsNone(models[0]())
        self.assertTrue(seen["released"])

    def test_the_model_is_dropped_when_the_transcription_fails(self):
        result, seen, models = self._run([], error_message="CUDA out of memory\nTried to allocate ...")
        self.assertIsNone(result)
        self.assertEqual(seen["error"], "Qwen3-ASR failed (RuntimeError: CUDA out of memory)")
        self.assertTrue(seen["released"])
        self.assertNotIn("segments", seen)
        # once the error is handled, nothing holds the model any more
        self.assertIsNone(models[0]())

    def test_alignment_errors_are_passed_on(self):
        # Whisper would run the same alignment: no reason to fall back to it
        def align_fails(*args, **kwargs):
            raise RuntimeError("alignment failed")

        pieces = [(np.zeros(16000 * 30, dtype=np.float32), 0.0)]

        class Model:
            def transcribe(self, audio, language=None):
                return [types.SimpleNamespace(text="words", language="English")]

        with patch.object(qwen3_asr, "_load_model", return_value=Model()), \
                patch.object(qwen3_asr, "audio_pieces", return_value=pieces), \
                patch.object(qwen3_asr, "song_language", return_value="en"), \
                patch("whisperx.load_audio", return_value=np.zeros(16000 * 30, dtype=np.float32)), \
                patch("modules.Speech_Recognition.Whisper.align_segments", side_effect=align_fails):
            with self.assertRaises(RuntimeError) as raised:
                transcribe_with_qwen3("a.wav", TranscriberChoice("qwen3", 2, ""))
        self.assertNotIsInstance(raised.exception, QwenFailed)

    def test_a_broken_install_fails_over(self):
        with patch.object(qwen3_asr, "audio_pieces", side_effect=ImportError("DLL load failed")), \
                patch.object(qwen3_asr, "song_language", return_value="en"), \
                patch("whisperx.load_audio", return_value=np.zeros(100, dtype=np.float32)):
            with self.assertRaises(QwenFailed) as raised:
                transcribe_with_qwen3("a.wav", TranscriberChoice("qwen3", 2, ""))
        self.assertEqual(str(raised.exception), "Qwen3-ASR failed (ImportError: DLL load failed)")

    def test_no_words_no_alignment(self):
        result, seen, _ = self._run(["", " ", ""])
        self.assertNotIn("segments", seen)
        self.assertEqual(result.transcribed_data, [])

    def test_no_pieces_no_model(self):
        with patch.object(qwen3_asr, "_load_model") as load, \
                patch.object(qwen3_asr, "audio_pieces", return_value=[]), \
                patch.object(qwen3_asr, "song_language", return_value="de"), \
                patch("whisperx.load_audio", return_value=np.zeros(100, dtype=np.float32)):
            result = transcribe_with_qwen3("a.wav", TranscriberChoice("qwen3", 2, ""))
        load.assert_not_called()
        self.assertEqual((result.transcribed_data, result.detected_language), ([], "de"))

    def test_unknown_detected_language_is_unavailable(self):
        with patch.object(qwen3_asr, "_load_model") as load, \
                patch.object(qwen3_asr, "song_language", return_value="uk"), \
                patch("whisperx.load_audio") as load_audio:
            with self.assertRaises(QwenUnavailable) as raised:
                transcribe_with_qwen3("a.wav", TranscriberChoice("qwen3", 2, ""))
        self.assertIn("does not transcribe the language uk", str(raised.exception))
        load.assert_not_called()
        load_audio.assert_not_called()


class TestLoadModel(unittest.TestCase):
    def test_float16_where_bfloat16_is_only_emulated(self):
        import torch

        loaded = {}

        class Qwen3ASRModel:
            @staticmethod
            def from_pretrained(name, **kwargs):
                loaded.update(kwargs, name=name)

        fake = types.ModuleType("qwen_asr")
        fake.Qwen3ASRModel = Qwen3ASRModel
        for native, dtype in ((True, torch.bfloat16), (False, torch.float16)):
            with self.subTest(native=native), patch.dict(sys.modules, {"qwen_asr": fake}), \
                    patch("torch.cuda.is_bf16_supported", return_value=native) as supported:
                qwen3_asr._load_model(TranscriberChoice("qwen3", 2))
                supported.assert_called_once_with(including_emulation=False)
                self.assertEqual((loaded["name"], loaded["dtype"], loaded["max_inference_batch_size"]),
                                 ("Qwen/Qwen3-ASR-1.7B", dtype, 2))

    def test_no_notice_about_the_temperature_while_loading(self):
        import logging
        logger = logging.getLogger("transformers.generation.configuration_utils")

        class Qwen3ASRModel:
            @staticmethod
            def from_pretrained(name, **kwargs):
                # what transformers reports about the generation config of the model
                logger.warning("The following generation flags are not valid and may be ignored: ['temperature'].")
                logger.warning("The following generation flags are not valid and may be ignored: ['top_p'].")
                return "model"

        fake = types.ModuleType("qwen_asr")
        fake.Qwen3ASRModel = Qwen3ASRModel
        with patch.dict(sys.modules, {"qwen_asr": fake}), patch("torch.cuda.is_bf16_supported", return_value=True), \
                self.assertLogs(logger, "WARNING") as logs:
            self.assertEqual(qwen3_asr._load_model(TranscriberChoice("qwen3", 2)), "model")
        self.assertEqual([r.getMessage() for r in logs.records],
                         ["The following generation flags are not valid and may be ignored: ['top_p']."])
        # only while loading
        self.assertEqual(logger.filters, [])


def _word(text, start, end):
    return types.SimpleNamespace(word=text + " ", start=start, end=end)


def _voice(seconds, quiet=()):
    """A tone of ``seconds``, silent within the ``quiet`` spans (start s, end s)."""
    t = np.arange(int(seconds * 16000)) / 16000
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for start, end in quiet:
        audio[int(start * 16000):int(end * 16000)] = 0
    return audio


class TestUnheardSpans(unittest.TestCase):
    def test_sung_span_without_words(self):
        words = [_word("a", 1.0, 2.0), _word("b", 8.0, 13.0), _word("c", 38.0, 39.0)]
        # sung throughout: from b to c there are 25 s without words; from a to b only 6 s
        self.assertEqual(qwen3_asr.unheard_spans(words, _voice(40.0)), [(13.0, 38.0)])

    def test_quiet_span_is_no_retry(self):
        words = [_word("a", 1.0, 2.0), _word("b", 8.0, 13.0), _word("c", 38.0, 39.0)]
        self.assertEqual(qwen3_asr.unheard_spans(words, _voice(40.0, quiet=[(13.0, 38.0)])), [])

    def test_start_and_end_of_the_song(self):
        words = [_word("a", 10.0, 11.0), _word("b", 17.0, 18.0)]
        self.assertEqual(qwen3_asr.unheard_spans(words, _voice(30.0)), [(0.0, 10.0), (18.0, 30.0)])

    def test_silence(self):
        self.assertEqual(qwen3_asr.unheard_spans([], np.zeros(16000 * 30, dtype=np.float32)), [])


class TestSpansHeardAgain(unittest.TestCase):
    WORDS = [_word("a", 1.0, 2.0), _word("d", 38.0, 39.0)]

    def _run(self, texts=None, error=None, align_error=None, words=None):
        seen = {}

        class Model:
            def transcribe(self, audio, language=None):
                seen["pieces"], seen["language"] = [len(wav) / 16000 for wav, _ in audio], language
                if error:
                    raise RuntimeError(error)
                return [types.SimpleNamespace(text=text, language="English") for text in texts]

        def align_segments(segments, audio, lang, device, alignment_model=None, keep_numbers=False):
            seen["segments"] = segments
            if align_error:
                raise ValueError(align_error)
            # one word placed beyond the span
            return [_word("b", 10.0, 11.0), _word("c", 20.0, 21.0), _word("x", 39.5, 40.0)]

        def pieces(audio):
            return [(audio, 0.0)]

        with patch.object(qwen3_asr, "_load_model", return_value=Model()), \
                patch.object(qwen3_asr, "audio_pieces", side_effect=pieces), \
                patch.object(qwen3_asr, "release_gpu_memory") as release, \
                patch("modules.Speech_Recognition.Whisper.align_segments", side_effect=align_segments):
            words = qwen3_asr._with_spans_heard_again(list(words or self.WORDS), [(2.0, 38.0)], _voice(40.0),
                                                      TranscriberChoice("qwen3", 2), "en", "cuda", None, False)
            seen["released"] = release.called
        return words, seen

    def test_words_of_the_span_join_in_order(self):
        words, seen = self._run(["b c"])
        self.assertEqual([w.word.strip() for w in words], ["a", "b", "c", "d"])
        self.assertEqual((seen["pieces"], seen["language"]), ([36.0], "English"))
        self.assertEqual(seen["segments"], [{"text": "b c", "start": 2.0, "end": 38.0}])
        self.assertTrue(seen["released"])

    def test_nothing_heard_again(self):
        words, seen = self._run(["  "])
        self.assertEqual([w.word.strip() for w in words], ["a", "d"])
        self.assertNotIn("segments", seen)

    def test_transcription_hears_its_unheard_spans_again(self):
        class Model:
            def transcribe(self, audio, language=None):
                return [types.SimpleNamespace(text="a d", language="English")]

        heard = [_word("a", 1.0, 2.0), _word("d", 38.0, 39.0)]
        with patch.object(qwen3_asr, "_load_model", return_value=Model()), \
                patch.object(qwen3_asr, "audio_pieces", return_value=[(np.zeros(16000 * 40, np.float32), 0.0)]), \
                patch.object(qwen3_asr, "song_language", return_value="en"), \
                patch.object(qwen3_asr, "release_gpu_memory"), \
                patch("whisperx.load_audio", return_value=_voice(40.0)), \
                patch("modules.Speech_Recognition.Whisper.align_segments", return_value=list(heard)), \
                patch.object(qwen3_asr, "_with_spans_heard_again", return_value=["all"]) as again:
            result = transcribe_with_qwen3("a.wav", TranscriberChoice("qwen3", 2, "8.0 GB free"), "cuda",
                                           alignment_model="m", keep_numbers=True)
        self.assertEqual(result.transcribed_data, ["all"])
        args = again.call_args[0]
        self.assertEqual((args[0], args[1], args[4:]), (heard, [(2.0, 38.0)], ("en", "cuda", "m", True)))

    def test_the_other_words_keep_their_order(self):
        # a word without a timestamp is placed after the one before it, which may be after the next one
        words = [_word("a", 1.0, 2.0), _word("2", 2.1, 2.2), _word("a2", 2.05, 2.5), _word("d", 38.0, 39.0)]
        found, _ = self._run(["b c"], words=words)
        self.assertEqual([w.word.strip() for w in found], ["a", "2", "a2", "b", "c", "d"])

    def test_alignment_failure_keeps_the_words(self):
        words, _ = self._run(["b c"], align_error="unknown language")
        self.assertEqual([w.word.strip() for w in words], ["a", "d"])

    def test_failure_keeps_the_words(self):
        words, seen = self._run(error="CUDA out of memory")
        self.assertEqual([w.word.strip() for w in words], ["a", "d"])
        self.assertTrue(seen["released"])


if __name__ == "__main__":
    unittest.main()
