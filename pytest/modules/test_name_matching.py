"""Tests for comparing song titles and artist names from different sources."""

import unittest

from src.modules.name_matching import (
    is_live_title,
    similarity,
    strip_title_additions,
    title_before_dash,
    title_key,
    title_without_part,
    titles_match,
    with_live_mark,
)


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
            "Song Title - No. 5 - Official Audio": "Song Title - No. 5",
            "Song Title - II - Lyric Video": "Song Title - II",
            "Song Title - I - Live Performance": "Song Title - I",
            "Song Title - V - Official Video": "Song Title - V",
            "Song Title - X - Lyric Video": "Song Title - X",
            "Song Title - No I Won't - Official Video": "Song Title",
            # a numbered description does not
            "Song Title - Take 2 - Official Video": "Song Title",
            "Song Title - Week 3 - Official Video": "Song Title",
            "Song Title - No Edits - Official Video": "Song Title",
            "Song Title - Part 2 - Take 3 - Official Video": "Song Title - Part 2",
            # an addition before a description
            "Song Title - Live - Some Festival 2020": "Song Title",
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
                      "Song Title HD - Singer Name in the studio",
                      # "x" and "I" inside the description are words, not part numbers
                      "Song Title - Singer x Rapper live at home", "Song Title - live as I sing it"):
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
                      "(Official Video) | Label Name Words", "Song Title - The Second Movement",
                      "Song Title - Part2 of the Story", "Song Title - Vol3 of the Series",
                      # a Roman part number at the start of the rest, or after "No."
                      "Song Title - V live in the studio", "Song Title - I and the band live",
                      "Song Title - No. V of the Cycle"):
            with self.subTest(title=title):
                self.assertEqual(title_before_dash(title), "")

    def test_any_rest(self):
        """For a search whose finds must show by their length that they are the song."""
        for title, before in (("Song Title [OFFICIAL VIDEO] - Chapter 2", "Song Title"),
                              ("Song Title - Part 2", "Song Title"), ("Song Title - Reprise", "Song Title"),
                              ("Song A - Song B - Song C", "Song A"),
                              ("Song Title", ""), ("Song Title | Label Name", ""), (" - Two Words", "")):
            with self.subTest(title=title):
                self.assertEqual(title_before_dash(title, any_rest=True), before)


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

    def test_part_numbers_written_differently(self):
        for a, b in (("Song Title - I", "Song Title, Part I"), ("Song Title - Part I", "Song Title (Part 1)"),
                     ("Song Title, Pt. 1", "Song Title Part One"), ("Song Title Pt. II", "Song Title, Part 2"),
                     ("Song Title (Part 1)", "Song Title, Part I"), ("Song Title - 2", "Song Title Part II"),
                     ("Song Title, Part 10", "Song Title - X")):
            with self.subTest(a=a, b=b):
                self.assertTrue(titles_match(a, b))
                self.assertEqual(title_key(a), title_key(b))

    def test_part_numbers_do_not_make_titles_alike(self):
        # the titles are compared without their part numbers: short ones stay apart, and a number
        # after a word like "No." or "Act" is no part number
        for a, b in (("Go 2", "So 2"), ("Me 2", "We 2"), ("Up 2", "Us 2"), ("Go (2)", "So (2)"),
                     ("Song Title No. 2", "Song Title 2"), ("Song Title (Act 2)", "Song Title 2"),
                     ("Song Title (Chapter 2)", "Song Title 2"), ("Song Title Vol. 2", "Song Title 2"),
                     ("Song Title Op. 2", "Song Title 2"), ("Track 7", "Track 8"), ("Intro", "Intro (Part 2)"),
                     # another number on one side only
                     ("Song Title 2", "Song Title 1 & 2"), ("Song Title 3", "Song Title 2 & 3"),
                     ("Song Title 2", "Song Title 3 2"), ("Song Title 2", "Song Title 1999 2")):
            with self.subTest(a=a, b=b):
                self.assertFalse(titles_match(a, b))
        for a, b in (("Song Title", "Song Title 2"), ("Blink 182", "Blink 182"), ("Catch 22", "Catch 22"),
                     ("Summer of 69", "Summer of 69"), ("Area 51", "Area 51"), ("Room 101", "Room 101")):
            with self.subTest(a=a, b=b):
                self.assertTrue(titles_match(a, b))

    def test_part_number_before_punctuation(self):
        for title in ("Song Title 2!", "Song Title 2?", "Song Title #2", "Song Title (2)"):
            with self.subTest(title=title):
                self.assertEqual(title_key(title), title_key("Song Title 2"))
                self.assertTrue(titles_match(title, "Song Title 2"))

    def test_bare_numbers_at_the_end(self):
        # a sequel number without a part word is the part number after a dash
        for a, b in (("Song Title - II", "Song Title II"), ("Song Title - 2", "Song Title 2"),
                     ("Song Title - 2", "Song Title (2)"), ("Song Title - III", "Song Title III"),
                     ("Song Title II", "Song Title 2"), ("Song Title - I", "Song Title I"),
                     ("Song Title 2", "Song Title Part 2")):
            with self.subTest(a=a, b=b):
                self.assertTrue(titles_match(a, b))
        for a, b in (("Song Title - II", "Song Title 3"), ("Song Title II", "Song Title III"),
                     ("Song Title 2", "Song Title, Part I"), ("Song Title - I", "Song Title II")):
            with self.subTest(a=a, b=b):
                self.assertFalse(titles_match(a, b))
        # numbers that belong to a word, years and words stay as they are
        for title in ("Part of Me", "Part Time Lover", "One", "Number 1", "Vol. 2", "Song Title 1999",
                      "The Part I Played", "Song Title Part 11", "Generation X", "Me & I", "2"):
            with self.subTest(title=title):
                self.assertNotIn("part 1", title_key(title).replace("part 11", ""))
                self.assertNotIn("part 2", title_key(title))

    def test_other_part_numbers_written_differently(self):
        for a, b in (("Song Title - II", "Song Title, Part I"), ("Song Title Part 1", "Song Title Part II"),
                     ("Song Title", "Song Title, Part I"), ("Song Title - I", "Song Title")):
            with self.subTest(a=a, b=b):
                self.assertFalse(titles_match(a, b))

    def test_i_as_a_word_is_no_part_number(self):
        for a, b in (("I", "I"), ("I Will", "I Will"), ("Song Title - I Will", "Song Title - I Will"),
                     ("The Part I Played", "The Part I Played")):
            with self.subTest(a=a, b=b):
                self.assertTrue(titles_match(a, b))
        self.assertNotIn("part 1", title_key("The Part I Played"))
        self.assertNotIn("part 1", title_key("Song Title - Part One of It"))

    def test_title_without_part(self):
        cases = {"Song Title - I": "Song Title", "Song Title, Part I": "Song Title", "Song Title (Pt. 2)": "Song Title",
                 "Song Title Part One": "Song Title", "Song Title - I (Official Video)": "Song Title",
                 "Song Title": "", "I": "", "Me & I": "", "Part 1": "", "Song Title - Part Two of It": "",
                 "The Part I Played": ""}
        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(title_without_part(title), expected)

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


class TestLiveTitle(unittest.TestCase):
    """A video title marks a live recording in an addition to the song title."""

    def test_live_additions(self):
        for title in ("Artist - Song (Live)", "Artist - Song [Live at Some Arena 2019]", "Artist - Song (LIVE)",
                      "Artist - Song (Official Live Video)", "Artist - Song - Live from the Studio",
                      "Artist - Song (Live Session)", "Artist - Song - Live"):
            with self.subTest(title=title):
                self.assertTrue(is_live_title(title))

    def test_no_live_mark(self):
        for title in ("Artist - Live Forever", "Artist - Song (Live Forever)", "Live Band - Song (Lyrics)",
                      "Artist - Song (Official Video)", "Artist - Song | Live Nation", '"Song" - Artist', ""):
            with self.subTest(title=title):
                self.assertFalse(is_live_title(title))

    def test_mark_added_once(self):
        self.assertEqual(with_live_mark("Song", "Artist - Song (Live)"), "Song (Live)")
        self.assertEqual(with_live_mark("Song (live)", "Artist - Song (Live)"), "Song (live)")
        self.assertEqual(with_live_mark("Song", "Artist - Song (Official Video)"), "Song")
        self.assertEqual(with_live_mark("", "Artist - Song (Live)"), "")
