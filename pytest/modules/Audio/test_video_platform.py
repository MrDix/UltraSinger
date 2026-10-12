"""Tests for video_platform.py"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import yt_dlp

from src.modules.Audio import video_platform
from src.modules.Audio.video_platform import get_video_title, strip_unmatched_suffixes
from src.modules.Audio.video_platform import download_and_convert_thumbnail
from src.modules.musicbrainz_client import SongInfo


class TestStripUnmatchedSuffixes(unittest.TestCase):
    """Tests for strip_unmatched_suffixes()."""

    def test_strips_live_when_not_in_video_title(self):
        """The music service says '(live)' but the video is an official studio MV."""
        track = "Pump It (live)"
        video_title = "Electric Callboy - PUMP IT (OFFICIAL VIDEO)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Pump It")

    def test_strips_live_at_venue_when_not_in_video_title(self):
        """Extended live suffix like '(Live at Wembley)' should also be stripped."""
        track = "Song Title (Live at Wembley)"
        video_title = "Artist - Song Title (Official Video)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song Title")

    def test_keeps_live_when_present_in_video_title(self):
        """If the video title also says 'live', keep the suffix."""
        track = "Song Title (Live)"
        video_title = "Artist - Song Title (Live at Wembley)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song Title (Live)")

    def test_strips_acoustic_when_not_in_video_title(self):
        track = "My Song (Acoustic)"
        video_title = "Band - My Song (Official Video)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "My Song")

    def test_strips_remastered_when_not_in_video_title(self):
        track = "Song (Remastered 2024)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_strips_remaster_when_not_in_video_title(self):
        track = "Song (Remaster)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_strips_radio_edit_when_not_in_video_title(self):
        track = "Song (Radio Edit)"
        video_title = "Artist - Song (Official Video)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_strips_explicit_when_not_in_video_title(self):
        track = "Song (Explicit)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_keeps_feat_even_when_not_in_video_title(self):
        """Feature credits are not on the whitelist and should never be stripped."""
        track = "Song (feat. Other Artist)"
        video_title = "Main Artist - Song (Official Video)"
        self.assertEqual(
            strip_unmatched_suffixes(track, video_title),
            "Song (feat. Other Artist)",
        )

    def test_keeps_feat_when_in_video_title(self):
        """Feature credits should be kept when the video title mentions them."""
        track = "Song (feat. Other Artist)"
        video_title = "Main Artist - Song (feat. Other Artist) [Official]"
        self.assertEqual(
            strip_unmatched_suffixes(track, video_title),
            "Song (feat. Other Artist)",
        )

    def test_keeps_part_number(self):
        """'(Part 2)' is not on the whitelist and should never be stripped."""
        track = "Song (Part 2)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song (Part 2)")

    def test_strips_multiple_trailing_whitelisted_suffixes(self):
        """Multiple trailing whitelist suffixes that are all absent should be stripped."""
        track = "Track Name (Remastered) (live)"
        video_title = "Artist - Track Name (Official Video)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Track Name")

    def test_no_parenthetical_returns_unchanged(self):
        track = "Simple Track Name"
        video_title = "Artist - Simple Track Name (Official Video)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Simple Track Name")

    def test_empty_track_returns_empty(self):
        self.assertEqual(strip_unmatched_suffixes("", "Some Video"), "")

    def test_case_insensitive_matching(self):
        """Suffix matching against video title should be case-insensitive."""
        track = "Song (LIVE)"
        video_title = "Artist - Song (live session)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song (LIVE)")

    def test_strips_only_whitelisted_keeps_non_whitelisted(self):
        """When a non-whitelisted suffix sits behind a whitelisted one,
        only the whitelisted outer one is stripped."""
        track = "Song (Remix) (live)"
        video_title = "Artist - Song (Remix)"
        # "(live)" is whitelisted and not in title -> strip
        # "(Remix)" is NOT whitelisted -> keep regardless
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song (Remix)")

    def test_strips_lyrics_when_not_in_video_title(self):
        track = "Song (Lyrics)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_strips_lyric_video_when_not_in_video_title(self):
        track = "Song (Lyric Video)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_strips_official_video_suffix(self):
        track = "Song (Official Video)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_no_false_positive_substring_match(self):
        """'live' inside 'delivered' should NOT count as a match."""
        track = "Song (Live)"
        video_title = "Artist - Song delivered to you"
        # "live" appears as substring of "delivered" but NOT as whole word
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_word_boundary_keeps_real_match(self):
        """'live' as a standalone word in the title should keep the suffix."""
        track = "Song (Live)"
        video_title = "Artist - Song live version"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song (Live)")


class TestGetVideoTitle(unittest.TestCase):
    @patch("yt_dlp.YoutubeDL")
    def test_get_video_title(self, mock_yt_dl):
        # Arrange
        # Also test leading and trailing whitespaces
        mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "   Test Artist   ",
            "track": "   Test Track   ",
            "title": "   Test Artist - Test Track   ",
            "channel": "   Test Channel   "
        }
        url = "   https://fakeUrl   "

        # Act
        artist, title, video_title, _language = get_video_title(url)

        # Assert
        self.assertEqual(artist, "Test Artist")
        self.assertEqual(title, "Test Track")
        self.assertEqual(video_title, "Test Artist - Test Track")
        mock_yt_dl.assert_called_once()

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_title_returns_language(self, mock_yt_dl):
        """Language metadata from yt-dlp should be returned as 4th element."""
        mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "Artist",
            "track": "Track",
            "title": "Artist - Track",
            "channel": "Channel",
            "language": "de",
        }

        artist, title, video_title, language = get_video_title("https://fakeUrl")

        self.assertEqual(language, "de")
        self.assertEqual(artist, "Artist")

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_title_empty_language_when_missing(self, mock_yt_dl):
        """When yt-dlp has no language field, return empty string."""
        mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "Artist",
            "track": "Track",
            "title": "Artist - Track",
            "channel": "Channel",
        }

        _, _, _, language = get_video_title("https://fakeUrl")

        self.assertEqual(language, "")

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_title_strips_music_service_suffix(self, mock_yt_dl):
        """Track metadata says '(live)' but video title does not."""
        mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "Electric Callboy",
            "track": "Pump It (live)",
            "title": "Electric Callboy - PUMP IT (OFFICIAL VIDEO)",
            "channel": "Electric Callboy",
        }

        artist, title, video_title, _language = get_video_title("https://fakeUrl")

        self.assertEqual(artist, "Electric Callboy")
        self.assertEqual(title, "Pump It")
        self.assertEqual(video_title, "Electric Callboy - PUMP IT (OFFICIAL VIDEO)")

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_title_keeps_matching_suffix(self, mock_yt_dl):
        """If track and video title agree on the suffix, keep it."""
        mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "Artist",
            "track": "Song (Live)",
            "title": "Artist - Song (Live at Festival)",
            "channel": "ArtistChannel",
        }

        artist, title, _video_title, _language = get_video_title("https://fakeUrl")

        self.assertEqual(artist, "Artist")
        self.assertEqual(title, "Song (Live)")

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_title_no_artist_with_dash(self, mock_yt_dl):
        """Fallback: no artist field, title contains dash."""
        mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "title": "  Some Artist - Some Song  ",
            "channel": "SomeChannel",
        }

        artist, title, video_title, _language = get_video_title("https://fakeUrl")

        self.assertEqual(artist, "Some Artist")
        self.assertEqual(title, "Some Song")
        self.assertEqual(video_title, "Some Artist - Some Song")

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_title_multi_dash_preserves_title(self, mock_yt_dl):
        """Multiple dashes in title: only split on the first one."""
        mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "title": "Artist Name - Song - Part 2",
            "channel": "ArtistChannel",
        }

        artist, title, _video_title, _language = get_video_title("https://fakeUrl")

        self.assertEqual(artist, "Artist Name")
        self.assertEqual(title, "Song - Part 2")

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_title_no_artist_no_dash(self, mock_yt_dl):
        """Fallback: no artist field, no dash in title -> use channel."""
        mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "title": "Just A Title",
            "channel": "MyChannel",
        }

        artist, title, video_title, _language = get_video_title("https://fakeUrl")

        self.assertEqual(artist, "MyChannel")
        self.assertEqual(title, "Just A Title")
        self.assertEqual(video_title, "Just A Title")

    @patch("yt_dlp.YoutubeDL")
    def test_channel_additions_are_not_part_of_the_artist(self, mock_yt_dl):
        for channel, expected in (("SomeArtistVEVO", "SomeArtist"), ("Some Artist - Topic", "Some Artist"),
                                  ("Some Artist Official", "Some Artist"), ("VEVO", "VEVO")):
            with self.subTest(channel=channel):
                mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
                    "title": "Just A Title",
                    "channel": channel,
                }
                artist, _title, _video_title, _language = get_video_title("https://fakeUrl")
                self.assertEqual(artist, expected)

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_title_artist_without_track_falls_through(self, mock_yt_dl):
        """If yt-dlp provides artist but no track, fall through to title parsing."""
        mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "Some Artist",
            "title": "Some Artist - Some Song",
            "channel": "SomeChannel",
        }

        artist, title, _video_title, _language = get_video_title("https://fakeUrl")

        # Should fall through to title-split logic
        self.assertEqual(artist, "Some Artist")
        self.assertEqual(title, "Some Song")

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_title_track_none_falls_through(self, mock_yt_dl):
        """If yt-dlp provides artist but track is None, fall through."""
        mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "Some Artist",
            "track": None,
            "title": "Some Artist - Some Song",
            "channel": "SomeChannel",
        }

        artist, title, _video_title, _language = get_video_title("https://fakeUrl")

        # Should fall through to title-split logic
        self.assertEqual(artist, "Some Artist")
        self.assertEqual(title, "Some Song")

    @patch("src.modules.Audio.video_platform.yt_dlp.YoutubeDL")
    @patch("src.modules.Audio.video_platform.save_image")
    def test_download_and_convert_thumbnail(self, mock_save_image, mock_yt_dl):
        # Arrange
        mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {"thumbnail": "test_thumbnail_url"}
        mock_yt_dl.return_value.__enter__.return_value.urlopen.return_value.read.return_value = b"test_image_data"

        mock_save_image.return_value = None

        ydl_opts = {}
        url = "https://fakeUrl"
        clear_filename = "test"
        output_path = "/path/to/output"

        # Act
        download_and_convert_thumbnail(ydl_opts, url, clear_filename, output_path)

        # Assert
        mock_yt_dl.assert_called_once_with(ydl_opts)
        mock_yt_dl.return_value.__enter__.return_value.extract_info.assert_called_once_with(url, download=False)
        mock_yt_dl.return_value.__enter__.return_value.urlopen.assert_called_once_with("test_thumbnail_url")


class TestDownloadRetries(unittest.TestCase):
    """A stalled connection is retried instead of ending the download."""

    @patch("src.modules.Audio.video_platform.yt_dlp.YoutubeDL")
    def test_download_retries_like_the_command_line(self, mock_yt_dl):
        download = getattr(video_platform, "__download_video_with_audio")
        with tempfile.TemporaryDirectory() as folder:
            video = os.path.join(folder, "test.mp4")
            open(video, "wb").close()
            mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
                "requested_downloads": [{"filepath": video}]}

            download("https://fakeUrl", "test", folder)

        ydl_opts = mock_yt_dl.call_args.args[0]
        self.assertEqual(ydl_opts["retries"], 10)
        self.assertEqual(ydl_opts["fragment_retries"], 10)

    def test_retries_are_shown_and_other_screen_messages_are_not(self):
        logger = video_platform._FilteredLogger()
        retry = "[download] Got error: Read timed out. Retrying (1/10)..."
        with patch("builtins.print") as mock_print:
            logger.debug(retry)
            logger.debug("[download]  42.0% of 10.00MiB")
        mock_print.assert_called_once_with(f"WARNING: {retry}")


_AUDIO = {"format_id": "251", "ext": "webm", "vcodec": "none", "acodec": "opus", "url": "https://a"}
_VIDEO = {"format_id": "614", "ext": "mp4", "vcodec": "vp09.00.40.08", "acodec": "none",
          "height": 1080, "url": "https://v"}
_PREMIUM = {**_VIDEO, "format_id": "616", "format_note": "Premium"}


class TestVideoFormat(unittest.TestCase):
    """The video is taken from a premium format only when no other mp4 video is offered."""

    @staticmethod
    def _pick(formats):
        """Format ids that the download's format string picks (formats sorted worst to best)."""
        selector = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}).build_format_selector(
            video_platform._VIDEO_FORMAT)
        picked = next(iter(selector({"formats": formats, "has_merged_format": False,
                                     "incomplete_formats": False})))
        return [f["format_id"] for f in picked.get("requested_formats") or [picked]]

    def test_premium_video_is_skipped(self):
        self.assertEqual(self._pick([_AUDIO, _VIDEO, _PREMIUM]), ["614", "251"])

    def test_premium_video_is_taken_when_it_is_the_only_one(self):
        self.assertEqual(self._pick([_AUDIO, _PREMIUM]), ["616", "251"])


class TestPlaylistUrl(unittest.TestCase):
    """A watch URL that also names a playlist stands for its video: title, download and
    thumbnail are those of the video, the playlist is not read."""

    @patch("src.modules.Audio.video_platform.yt_dlp.YoutubeDL")
    def test_only_the_video(self, mock_yt_dl):
        url = "https://fakeUrl/watch?v=x&list=y"
        with tempfile.TemporaryDirectory() as folder:
            video = os.path.join(folder, "Artist - Title.mp4")
            open(video, "wb").close()
            mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
                "title": "Artist - Title", "thumbnail": None, "requested_downloads": [{"filepath": video}]}
            get_video_title(url)
            getattr(video_platform, "__download_video_with_audio")(url, "Artist - Title", folder)
            getattr(video_platform, "__download_thumbnail")(url, "Artist - Title", folder)
        options = [call.args[0] for call in mock_yt_dl.call_args_list]
        self.assertEqual(len(options), 3)
        self.assertEqual([o.get("noplaylist") for o in options], [True, True, True])


class TestOutputTemplate(unittest.TestCase):
    """A "%" in the song name or the output folder is part of the file name, no field of the video's data."""

    def test_percent_in_the_names(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, "50% Songs")
            name = "Artist - 100%(Live) %s %%"
            template = video_platform._output_template(folder, name)
            path = yt_dlp.YoutubeDL({"outtmpl": template, "quiet": True}).prepare_filename(
                {"id": "x", "title": "Video", "ext": "webm"})
        self.assertEqual(os.path.normpath(path), os.path.normpath(os.path.join(folder, name + ".webm")))


class TestDownloadedFile(unittest.TestCase):
    """The song goes on with the file that was downloaded, whatever its container."""

    def _download(self, folder, filepath):
        download = getattr(video_platform, "__download_video_with_audio")
        with patch.object(video_platform.yt_dlp, "YoutubeDL") as ydl:
            ydl.return_value.__enter__.return_value.extract_info.return_value = {
                "requested_downloads": [{"filepath": filepath}]}
            return download("https://fakeUrl", "Artist - Title", folder)

    def test_single_file_keeps_its_container(self):
        with tempfile.TemporaryDirectory() as folder:
            video = os.path.join(folder, "Artist - Title.webm")
            open(video, "wb").close()
            self.assertEqual(self._download(folder, video), video)

    def test_missing_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(Exception, "no downloaded file"):
                self._download(folder, os.path.join(folder, "Artist - Title.mp4"))

    def test_webm_and_a_failed_thumbnail(self):
        with tempfile.TemporaryDirectory() as out:
            song_folder = os.path.join(out, "Artist - Title")
            video = os.path.join(song_folder, "Artist - Title.webm")

            def extract_info(url, download):
                open(video, "wb").close()
                return {"requested_downloads": [{"filepath": video}]}

            with patch.object(video_platform, "get_video_title",
                              return_value=("Artist", "Title", "Artist - Title", "en")), \
                    patch.object(video_platform, "search_musicbrainz",
                                 return_value=SongInfo(title="Title", artist="Artist")), \
                    patch.object(video_platform.yt_dlp, "YoutubeDL") as ydl, \
                    patch.object(video_platform, "download_and_convert_thumbnail",
                                 side_effect=OSError("HTTP Error 404: Not Found")), \
                    patch.object(video_platform, "separate_audio_video",
                                 return_value=("audio.ogg", "video.webm", "ogg", "webm")) as split, \
                    redirect_stdout(io.StringIO()) as log:
                ydl.return_value.__enter__.return_value.extract_info.side_effect = extract_info
                _, folder, _, media_info = video_platform.download_from_video_platform("https://fakeUrl", out)
        self.assertEqual(split.call_args.args[0], video)
        self.assertEqual(folder, song_folder)
        # The cover is optional: the song goes on without it
        self.assertEqual((media_info.video_extension, media_info.cover_url), ("webm", ""))
        self.assertIn("the thumbnail could not be downloaded", log.getvalue())


if __name__ == "__main__":
    unittest.main()
