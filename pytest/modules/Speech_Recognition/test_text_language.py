"""Tests for lyrics language identification (modules.Speech_Recognition.text_language).

All texts are invented for the test; no real song lyrics are used.
"""

import pytest

from modules.Speech_Recognition.text_language import (
    check_lyrics_language,
    detect_text_language,
    lyrics_language_mismatch,
)

EN = ("[00:01.00] I walk along the river and I think of you, the night is cold but you are "
      "still here with me, and when the morning comes we will be free, oh I know it's true, "
      "we never stop, it's all for you and me, so take my hand and let us go now ") * 2
DE = ("[00:01.00] Ich gehe durch die Stadt und denke nur an dich, die Nacht ist kalt doch du bist "
      "immer noch bei mir, und wenn der Morgen kommt dann sind wir frei, ich weiß es ja, "
      "wir hören nicht auf, das ist alles nur für dich und mich, komm mit mir ") * 2
IT = ("Cammino lungo il fiume e penso a te, la notte è fredda ma tu sei ancora qui con me, "
      "e quando arriva il giorno non ho più paura, io lo so che sei la mia vita, "
      "non dire niente ma resta con me per sempre ") * 2
ES = ("Camino por el río y pienso en ti, la noche es fría pero tú estás aquí conmigo, "
      "y cuando llega el día no tengo miedo, yo sé que eres mi vida, "
      "no digas nada pero quédate conmigo para siempre ") * 2


class TestDetect:
    @pytest.mark.parametrize("text, lang", [(EN, "en"), (DE, "de"), (IT, "it"), (ES, "es")])
    def test_languages(self, text, lang):
        assert detect_text_language(text) == lang

    def test_too_short_is_unknown(self):
        assert detect_text_language("I love you") is None

    def test_scripts(self):
        assert detect_text_language("こんにちは、世界。今日はいい天気ですね。") == "ja"
        assert detect_text_language("你好世界，今天天气很好，我们一起去公园吧。") == "zh"
        assert detect_text_language("Привет мир, сегодня хорошая погода, пойдём гулять.") == "ru"

    def test_lrc_timestamps_ignored(self):
        assert detect_text_language("[00:12.34]" + EN) == "en"

    def test_ambiguous_text_is_unknown(self):
        mixed = EN[: len(EN) // 2] + DE[: len(DE) // 2]
        assert detect_text_language(mixed) is None


class TestMismatch:
    def test_same_language(self):
        assert lyrics_language_mismatch(DE, "de") is None

    def test_other_language(self):
        assert lyrics_language_mismatch(EN, "de") == "en"

    def test_unknown_sung_language_not_judged(self):
        assert lyrics_language_mismatch(EN, "fi") is None
        assert lyrics_language_mismatch(EN, None) is None

    def test_sibling_languages_not_reported(self):
        assert lyrics_language_mismatch(ES, "pt") is None

    def test_region_codes(self):
        assert lyrics_language_mismatch(DE, "de-AT") is None


class TestCheckLyricsLanguage:
    def test_matching_lyrics_used(self):
        assert check_lyrics_language(DE, "de", language_is_confident=True) == (True, None)

    def test_confident_language_rejects_other_version(self):
        # sung in German with a confident detection -> English lyrics are another version
        assert check_lyrics_language(EN, "de", language_is_confident=True) == (False, "en")

    def test_weak_guess_keeps_lyrics_and_reports(self):
        # uncertain detection said Spanish, the lyrics are clearly Italian:
        # either side may be wrong, so the lyrics are kept and the other language reported
        assert check_lyrics_language(IT, "es", language_is_confident=False) == (True, "it")

    def test_unclear_lyrics_change_nothing(self):
        assert check_lyrics_language("la la la", "en", language_is_confident=False) == (True, None)
