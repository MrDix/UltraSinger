"""Tests for hyphenation.py"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import requests

from src.modules.Speech_Recognition.hyphenation import hyphenation, language_check
from hyphen import Hyphenator, dictools


def _installed_dictionary(language):
    """``(content, url)`` of the installed dictionary for ``language``, or None.

    The installed dictionaries (the user's, shared by all programs that use
    them) are only read here, never changed. A dictionary installed for
    ``language`` is listed under that name or, when the language has regions
    (``"en"``), stored as ``hyph_<language>.dic``.
    """
    store = dictools.DEFAULT_DICT_PATH
    try:
        with open(os.path.join(store, "dictionaries.json"), "rb") as file:
            listed = json.load(file)
        entry = listed.get(language) or {"file": f"hyph_{language}.dic", "url": ""}
        with open(os.path.join(store, entry["file"]), "rb") as file:
            return file.read(), entry["url"]
    except (OSError, ValueError, KeyError, AttributeError):
        return None


class TestHypenation(unittest.TestCase):

    def setUp(self):
        # A dictionary folder of the test's own: the installed dictionaries are
        # only copied from, never removed or changed
        self.dictionaries = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dictionaries, ignore_errors=True)

    def hyphenator(self, language):
        """A Hyphenator with a copy of the installed dictionary, else with one
        downloaded into the test's folder; skips the test when the dictionary
        is not installed and cannot be downloaded (offline)."""
        if not dictools.is_installed(language, self.dictionaries):
            installed = _installed_dictionary(language)
            if installed:
                content, url = installed
                dictools.Dictionaries(self.dictionaries).add(language, content, [language], url)
        try:
            return Hyphenator(language, directory=self.dictionaries)
        except requests.RequestException as error:
            self.skipTest(f"no {language} dictionary is installed and it cannot be downloaded ({error})")

    def test_hypenation(self):
        """Test case for hyphenation function."""
        assert hyphenation("darkness", self.hyphenator("en")) == ["dark", "ness"]
        assert hyphenation("Hombre", self.hyphenator("es")) == ["Hom", "bre"]
        assert hyphenation("begegnen", self.hyphenator("de_DE")) == ["be", "geg", "nen"]
        assert hyphenation(".b,e~g'eg*nen, ", self.hyphenator("de_DE")) == [".b,e", "~g'eg", "*nen, "]
        assert hyphenation("Abend, ", self.hyphenator("de_AT")) == None
        assert hyphenation("Abend.", self.hyphenator("de_AT")) == None

    @patch('hyphen.dictools.list_installed')
    def test_language_check_has_installed_language(self, mock_list_installed):
        mock_list_installed.return_value = ['de', 'de_DE', 'en', 'en_US', 'en_GB']
        assert language_check("en") == "en_US"
        assert language_check("de") == "de_DE"

    @patch('hyphen.dictools.list_installed')
    def test_language_check_not_installed_language(self, mock_list_installed):
        mock_list_installed.return_value = []
        assert language_check("fr") == "fr_FR"
        assert language_check("de") == "de"
        assert language_check("none") == None


if __name__ == "__main__":
    unittest.main()
