"""Tests for comparing song titles and artist names from different sources."""

import unittest

from src.modules.name_matching import similarity, strip_title_additions, title_before_dash, titles_match


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
            "Song Title (Live)": "Song Title",
            "Song Title (Live 1985)": "Song Title",
            "Song Title (Recorded Live)": "Song Title",
            "Song Title (20th Anniversary Edition)": "Song Title",
            "Song Title (Deluxe Edition)": "Song Title",
            "Song Title (Slowed + Reverb)": "Song Title",
            "Song Title (with Guest Name)": "Song Title",
            "Song Title (Vidéo officielle)": "Song Title",
            "Song Title (Video Edit)": "Song Title",
            "Song Title - From the Album Some Album": "Song Title",
            "Song Title (From \"Some Film\")": "Song Title",
            "Song Title ()": "Song Title",
            "Song Title (Live from Some Place)": "Song Title",
            "Song Title - Live from Some Place": "Song Title",
            "Song Title (Music from the Motion Picture)": "Song Title",
            "Song Title (From the Original Motion Picture Soundtrack)": "Song Title",
            "Song Title (Live & Acoustic)": "Song Title",
            "Song Title (Audio Only)": "Song Title",
            "Song Title (Clip Vidéo)": "Song Title",
            "Song Title (Video Official)": "Song Title",
            "Song Title (Letra Oficial)": "Song Title",
            "Song Title - Singer Name live one take performance": "Song Title",
            "Song Title (Acoustic Performance)": "Song Title",
            "Song Title - Drum Playthrough": "Song Title",
            "Song Title - Some Name - Remix": "Song Title",
            "Song Title - 4K": "Song Title",
            "Song Title - Remastered 09": "Song Title",
            # a part of the work stays, also before an addition
            "Song Title - Part 2 - Live Performance": "Song Title - Part 2",
            "Song Title - Vol. 3 - Official Video": "Song Title - Vol. 3",
        }
        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(strip_title_additions(title), expected)

    def test_titles_without_additions_stay(self):
        for title in ("Song Title", "Some Title Video", "The Official Word", "Hyphen-Word Title",
                      "(Some Words) Song Title", "Song Title (Part 2)", "Song Title - Part 2",
                      "Song Title (Between Us)",
                      # words that start additions, but here start other words of a title
                      "Song Title - Us Placeholder", "Song Title - From Placeholder Words",
                      "Song Title (Video Placeholder)", "Song Title (Live Placeholder)",
                      "Song Title (Original Placeholder)", "Song Title - Radio Placeholder",
                      "Song Title - With Placeholder", "Song Title - Us & Placeholder",
                      "Song Title (Live for Placeholder)"):
            with self.subTest(title=title):
                self.assertEqual(strip_title_additions(title), title)

    def test_nothing_left(self):
        self.assertEqual(strip_title_additions("(Official Video)"), "(Official Video)")
        self.assertEqual(strip_title_additions(""), "")


class TestTitleBeforeDash(unittest.TestCase):
    def test_description_after_the_dash(self):
        for title in ("Song Title - Singer Name in the studio", "Song Title – some words about the band",
                      "Song Title — some words about the band",
                      "Song Title - Singer Name in the studio (Official Video)",
                      "Song Title - Singer Name live, one take", "Song Title - at home with the band 2020",
                      "Song Title HD - Singer Name in the studio"):
            with self.subTest(title=title):
                self.assertEqual(title_before_dash(title), "Song Title")
        # as the song database search passes it, and with a part in brackets
        self.assertEqual(title_before_dash("song title - singer name in the studio"), "song title")
        self.assertEqual(title_before_dash("Song Title (Part 2) - Singer Name in the studio"), "Song Title (Part 2)")

    def test_no_description(self):
        for title in ("Song Title", "Song Title - Reprise", "Song Title - Two Words",
                      "Song Title - Part 2 of the Story", "Song Title - Part Two of It",
                      "Song Title - Vol. 3 of the Series", "Song Title - Chapter II of It",
                      "Song Title - the 1 and only", "Song Title - Live at Some Place",
                      "Hyphen-Word Title with more words", " - Some Words Without Title",
                      # a medley, a separator that is no dash, a work in movements
                      "Song A - Song B - Song C", "Song Title | Singer Name in the studio",
                      "(Official Video) | Label Name Words", "Song Title - The Second Movement"):
            with self.subTest(title=title):
                self.assertEqual(title_before_dash(title), "")


class TestTitlesMatch(unittest.TestCase):
    def test_same_song(self):
        for a, b in (("Song Title (US)", "Song Title [Edit]"),
                     ("Song Title - US Version (Official Video)", "Song Title"),
                     ("(Some Words) Song Title", "Song Title"),
                     ("Song Title (Remastered 2011)", "Song Title (2009 Remaster)"),
                     ("Song Title (feat. Guest)", "Song Title (with Guest)")):
            with self.subTest(a=a, b=b):
                self.assertTrue(titles_match(a, b))

    def test_other_part_is_another_song(self):
        for a, b in (("Intro (Part 1)", "Intro (Part 2)"), ("Song Title - Part 1", "Song Title - Part 2"),
                     ("Song Title", "Song Title Goes On"), ("Intro", "Intro (Part 2)"),
                     ("Song Title", "Song Title (Reprise)"), ("Song Title [Some Words]", "Song Title"),
                     # long titles too, where the shared part makes the characters very similar
                     ("Some Much Longer Song Title", "Some Much Longer Song Title (Part 2)"),
                     ("Some Much Longer Song Title", "Some Much Longer Song Title (Reprise)")):
            with self.subTest(a=a, b=b):
                self.assertFalse(titles_match(a, b))

    def test_longer_artist_name_is_another_artist(self):
        self.assertLess(similarity("Some Much Longer Artist Name Tribute", "Some Much Longer Artist Name",
                                   artist=True), 0.8)

    def test_leading_article_left_out(self):
        self.assertEqual(similarity("The Some Band", "Some Band", artist=True), 1.0)
        self.assertEqual(similarity("Die Some Band", "Some Band", artist=True), 1.0)
        self.assertTrue(titles_match("The Some Song Title", "Some Song Title"))
        self.assertEqual(similarity("The", "The", artist=True), 1.0)
        # in titles, "Die" or "Las" can be a word of the title
        self.assertFalse(titles_match("Die Some Words", "Some Words"))

    def test_title_of_brackets_only(self):
        self.assertTrue(titles_match("(Untitled)", "(Untitled)"))


if __name__ == "__main__":
    unittest.main()
