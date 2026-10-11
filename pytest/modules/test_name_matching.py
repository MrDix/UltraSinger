"""Tests for comparing song titles and artist names from different sources."""

import unittest

from src.modules.name_matching import strip_title_additions


class TestStripTitleAdditions(unittest.TestCase):
    def test_additions_of_video_titles(self):
        cases = {
            "Song Title - US Version (Official Video)": "Song Title",
            "Song Title (US Version) (Official Music Video)": "Song Title",
            "Song Title (US)": "Song Title",
            "Song Title [HD]": "Song Title",
            "Song Title HD": "Song Title",
            "Song Title 4K 1080p": "Song Title",
            "Song Title (feat. Guest Name)": "Song Title",
            "Song Title feat. Guest Name": "Song Title",
            "Song Title ft. Guest Name": "Song Title",
            "Song Title | Label Name": "Song Title",
            "Song Title // Live": "Song Title",
            "Song Title – Remastered 2011": "Song Title",
            "Song Title Official Music Video": "Song Title",
            "Song Title official lyric video": "Song Title",
            "(Some Words) Song Title": "Song Title",
        }
        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(strip_title_additions(title), expected)

    def test_titles_without_additions_stay(self):
        for title in ("Song Title", "Some Title Video", "The Official Word", "Hyphen-Word Title"):
            with self.subTest(title=title):
                self.assertEqual(strip_title_additions(title), title)

    def test_nothing_left(self):
        self.assertEqual(strip_title_additions("(Untitled)"), "(Untitled)")
        self.assertEqual(strip_title_additions(""), "")


if __name__ == "__main__":
    unittest.main()
