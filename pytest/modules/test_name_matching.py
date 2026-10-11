"""Tests for comparing song titles and artist names from different sources."""

import unittest

from src.modules.name_matching import strip_title_additions, titles_match


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
            "Song Title - Live at Some Place": "Song Title",
            "Song Title Official Music Video": "Song Title",
            "Song Title official lyric video": "Song Title",
            "Song Title (Live at Some Place)": "Song Title",
            "Song Title (2011 Remaster)": "Song Title",
            "Song Title (Some Name Remix)": "Song Title",
            "Song Title [Radio Edit]": "Song Title",
            "Song Title (With Lyrics)": "Song Title",
            "Song Title - Part Two | Label Name": "Song Title - Part Two",
        }
        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(strip_title_additions(title), expected)

    def test_titles_without_additions_stay(self):
        for title in ("Song Title", "Some Title Video", "The Official Word", "Hyphen-Word Title",
                      "(Some Words) Song Title", "Song Title (Part 2)", "Song Title - Part 2",
                      "Song Title (Between Us)"):
            with self.subTest(title=title):
                self.assertEqual(strip_title_additions(title), title)

    def test_nothing_left(self):
        self.assertEqual(strip_title_additions("(Official Video)"), "(Official Video)")
        self.assertEqual(strip_title_additions(""), "")


class TestTitlesMatch(unittest.TestCase):
    def test_same_song(self):
        for a, b in (("Song Title (US)", "Song Title [Edit]"),
                     ("Song Title - US Version (Official Video)", "Song Title"),
                     ("(Some Words) Song Title", "Song Title"),
                     ("Song Title (Remastered 2011)", "Song Title (2009 Remaster)")):
            with self.subTest(a=a, b=b):
                self.assertTrue(titles_match(a, b))

    def test_other_part_is_another_song(self):
        for a, b in (("Intro (Part 1)", "Intro (Part 2)"), ("Song Title - Part 1", "Song Title - Part 2"),
                     ("Song Title", "Song Title Goes On"), ("Intro", "Intro (Part 2)"),
                     ("Song Title", "Song Title (Reprise)"), ("Song Title [Some Words]", "Song Title")):
            with self.subTest(a=a, b=b):
                self.assertFalse(titles_match(a, b))

    def test_title_of_brackets_only(self):
        self.assertTrue(titles_match("(Untitled)", "(Untitled)"))


if __name__ == "__main__":
    unittest.main()
