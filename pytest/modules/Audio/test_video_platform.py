"""Tests for video_platform.py"""

import contextlib
import copy
import http.server
import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stdout
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

    @staticmethod
    def _download_options():
        """The yt-dlp options of the download of a video with its audio."""
        download = getattr(video_platform, "__download_video_with_audio")
        with patch("src.modules.Audio.video_platform.yt_dlp.YoutubeDL") as mock_yt_dl, \
                tempfile.TemporaryDirectory() as folder:
            video = os.path.join(folder, "test.mp4")
            open(video, "wb").close()
            mock_yt_dl.return_value.__enter__.return_value.extract_info.return_value = {
                "requested_downloads": [{"filepath": video}]}
            download("https://fakeUrl", "test", folder)
        return mock_yt_dl.call_args.args[0]

    def test_download_retries_like_the_command_line(self):
        ydl_opts = self._download_options()
        self.assertEqual(ydl_opts["retries"], 10)
        self.assertEqual(ydl_opts["fragment_retries"], 10)
        self.assertEqual(ydl_opts["retry_sleep_functions"],
                         {"http": video_platform._retry_pause, "fragment": video_platform._fragment_retry_pause})

    def test_the_pause_before_a_retry_doubles_up_to_half_a_minute(self):
        pauses = [video_platform._retry_pause(n) for n in range(10)]
        self.assertEqual(pauses, [1, 2, 4, 8, 16, 30, 30, 30, 30, 30])
        self.assertEqual(sum(pauses), 181)  # about 3 minutes over the 10 retries

    def test_the_pause_before_a_retry_of_a_fragment_stays_short(self):
        """A stream has many fragments; one that is gone for good costs all of its retries."""
        pauses = [video_platform._fragment_retry_pause(n) for n in range(10)]
        self.assertEqual(pauses, [1, 2, 4, 5, 5, 5, 5, 5, 5, 5])
        self.assertEqual(sum(pauses), 42)

    def test_the_downloader_waits_the_pauses_and_shows_them(self):
        """With yt-dlp's own HTTP downloader: the download and the fragment pause, both shown."""
        ydl = yt_dlp.YoutubeDL(self._download_options())
        downloader = yt_dlp.downloader.http.HttpFD(ydl, ydl.params)
        error = Exception("HTTP Error 503: Service Unavailable")
        with patch("yt_dlp.utils._utils.time.sleep") as sleep, patch("builtins.print") as mock_print:
            downloader.report_retry(error, 4, 10)
            downloader.report_retry(error, 4, 10, frag_index=7)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [8.0, 5.0])
        retry = "WARNING: [download] Got error: HTTP Error 503: Service Unavailable. Retrying"
        self.assertEqual([c.args[0] for c in mock_print.call_args_list], [
            f"{retry} (4/10)...", "[download] Sleeping 8.00 seconds ...",
            f"{retry} fragment 7 (4/10)...", "[download] Sleeping 5.00 seconds ..."])

    def test_yt_dlp_waits_the_pause_before_each_retry(self):
        """yt-dlp passes the number of the retry, counted from 0, as n."""
        shown = []
        with patch("yt_dlp.utils._utils.time.sleep") as sleep:
            for count in (1, 2, 6):
                yt_dlp.utils.RetryManager.report_retry(
                    Exception("HTTP Error 503: Service Unavailable"), count, 10,
                    sleep_func=video_platform._retry_pause, info=shown.append, warn=lambda msg: None)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.0, 2.0, 30.0])
        self.assertEqual(shown[0], "Sleeping 1.00 seconds ...")

    def test_retries_and_waits_are_shown_and_other_screen_messages_are_not(self):
        logger = video_platform._FilteredLogger()
        retry = "[download] Got error: Read timed out. Retrying (1/10)..."
        site_wait = "[download] Sleeping 12.00 seconds as required by the site..."
        with patch("builtins.print") as mock_print:
            logger.debug(retry)
            logger.debug("Sleeping 1.00 seconds ...")
            logger.debug(site_wait)
            logger.debug("[download]  42.0% of 10.00MiB")
        self.assertEqual([c.args[0] for c in mock_print.call_args_list],
                         [f"WARNING: {retry}", "[download] Sleeping 1.00 seconds ...", site_wait])


_AUDIO = {"format_id": "251", "ext": "webm", "vcodec": "none", "acodec": "opus", "url": "https://a"}
_AUDIO_M4A = {"format_id": "140", "ext": "m4a", "vcodec": "none", "acodec": "mp4a.40.2", "url": "https://m"}
_VIDEO = {"format_id": "614", "ext": "mp4", "vcodec": "vp09.00.40.08", "acodec": "none",
          "height": 1080, "url": "https://v"}
_VIDEO_720 = {**_VIDEO, "format_id": "136", "vcodec": "avc1.4d401f", "height": 720}
_PREMIUM = {**_VIDEO, "format_id": "616", "format_note": "Premium"}
_VIDEO_WITH_AUDIO = {"format_id": "18", "ext": "mp4", "vcodec": "avc1.42001E", "acodec": "mp4a.40.2",
                     "height": 360, "url": "https://c"}


class TestVideoFormat(unittest.TestCase):
    """The video is taken from a premium format only when no other mp4 video is offered."""

    @staticmethod
    def _pick(formats, left_out=()):
        """Format ids that the download's format string picks (formats sorted worst to best),
        None for none."""
        selector = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}).build_format_selector(
            video_platform._video_format(left_out))
        picked = next(iter(selector({"formats": formats, "has_merged_format": False,
                                     "incomplete_formats": False})), None)
        return picked and [f["format_id"] for f in picked.get("requested_formats") or [picked]]

    def test_premium_video_is_skipped(self):
        self.assertEqual(self._pick([_AUDIO, _VIDEO, _PREMIUM]), ["614", "251"])

    def test_premium_video_is_taken_when_it_is_the_only_one(self):
        self.assertEqual(self._pick([_AUDIO, _PREMIUM]), ["616", "251"])

    def test_without_formats_left_out_the_format_string_is_the_same(self):
        self.assertEqual(video_platform._video_format(), video_platform._VIDEO_FORMAT)

    def test_a_video_format_left_out_gives_the_next_best_video(self):
        self.assertEqual(self._pick([_AUDIO, _VIDEO_720, _VIDEO], ["614"]), ["136", "251"])

    def test_an_audio_format_left_out_gives_another_audio_format(self):
        self.assertEqual(self._pick([_AUDIO_M4A, _AUDIO, _VIDEO], ["251"]), ["614", "140"])

    def test_formats_left_out_never_give_the_audio_alone(self):
        self.assertIsNone(self._pick([_AUDIO, _VIDEO], ["614"]))
        self.assertEqual(self._pick([_AUDIO, _VIDEO_WITH_AUDIO, _VIDEO], ["614"]), ["18"])
        self.assertIsNone(self._pick([_AUDIO, _VIDEO_WITH_AUDIO, _VIDEO], ["614", "18"]))


# The format sorting of the video platform's extractor, and the quality it gives
# the formats of a resolution (from its label: "hd1080" ...)
_PLATFORM_SORT = ("quality", "res", "fps", "hdr:12", "source", "vcodec", "channels", "acodec", "lang", "proto")
_PLATFORM_QUALITY = {360: 6, 480: 7, 720: 8, 1080: 9, 1440: 10, 2160: 11}
_PLATFORM_AUDIO = [
    {"format_id": "251", "ext": "webm", "vcodec": "none", "acodec": "opus", "quality": 3,
     "source_preference": -1, "protocol": "https", "url": "https://audio.invalid/251"},
    {"format_id": "140", "ext": "m4a", "vcodec": "none", "acodec": "mp4a.40.2", "quality": 3,
     "source_preference": -1, "protocol": "https", "url": "https://audio.invalid/140"},
]
_GENTLE = video_platform.DownloadLimits(2000, 1080, 60)  # the defaults of the command line


def _platform_video(format_id, height, fps=30, vcodec="av01.0.08M.08", width=None, label=None, **fields):
    """A video-only mp4 format as the video platform's extractor gives it: 16:9 unless
    ``width``, with the quality of its ``label`` (1080 for "1080p"; by default its shorter side)."""
    width = width or height * 16 // 9
    return {"format_id": format_id, "ext": "mp4", "vcodec": vcodec, "acodec": "none", "height": height,
            "width": width, "fps": fps, "quality": _PLATFORM_QUALITY[label or min(width, height)],
            "source_preference": -1, "protocol": "https", "url": f"https://video.invalid/{format_id}", **fields}


def _sized(ladder):
    """Formats named by their size, of (width, height, label) as the video platform names them."""
    return [_platform_video(f"{width}x{height}", height, width=width, label=label)
            for width, height, label in ladder]


def _picked(options, formats):
    """The format ids that yt-dlp picks with ``options`` (it sorts the formats itself)."""
    info = {"id": "vid", "title": "Video", "extractor": "test", "extractor_key": "Test",
            "webpage_url": "https://video.invalid/watch", "formats": copy.deepcopy(formats),
            "_format_sort_fields": _PLATFORM_SORT}
    with redirect_stdout(io.StringIO()):
        picked = yt_dlp.YoutubeDL(options).process_ie_result(info, download=False)
    return [f["format_id"] for f in picked.get("requested_formats") or [picked]]


class TestDownloadLimits(unittest.TestCase):
    """A video comes gently: at up to the rate limit, in the best format at or below
    the resolution and frame rate, else in the smallest one above (yt-dlp's own
    sorting and selector, with the format sorting of the video platform's extractor)."""

    _UP_TO_2160P = [_platform_video("398", 720), _platform_video("399", 1080), _platform_video("400", 1440),
                    _platform_video("401", 2160)]
    # Videos of other shapes, as the video platform offers them: (width, height, label)
    _SHAPES = {
        "16:9": [(1280, 720, 720), (1920, 1080, 1080), (2560, 1440, 1440), (3840, 2160, 2160)],
        "wide (2.39:1)": [(1280, 534, 720), (1920, 800, 1080), (2560, 1066, 1440), (3840, 1600, 2160)],
        "4:3": [(960, 720, 720), (1440, 1080, 1080), (1920, 1440, 1440), (2880, 2160, 2160)],
        "portrait": [(720, 1280, 720), (1080, 1920, 1080), (1440, 2560, 1440), (2160, 3840, 2160)],
        "square": [(720, 720, 720), (1080, 1080, 1080), (1440, 1440, 1440)],
    }

    @staticmethod
    def _options(limits):
        """The yt-dlp options of the download within ``limits``, and the console."""
        download = getattr(video_platform, "__download_video_with_audio")
        with patch.object(video_platform.yt_dlp, "YoutubeDL") as ydl, tempfile.TemporaryDirectory() as folder, \
                redirect_stdout(io.StringIO()) as log:
            video = os.path.join(folder, "Artist - Title.mp4")
            open(video, "wb").close()
            ydl.return_value.__enter__.return_value.extract_info.return_value = {
                "requested_downloads": [{"filepath": video}]}
            download("https://fakeUrl", "Artist - Title", folder, None, None, limits)
        return ydl.call_args.args[0], log.getvalue()

    def _pick(self, formats, limits=_GENTLE, left_out=()):
        """The format ids that the download within ``limits`` picks."""
        options = self._options(limits)[0]
        return _picked({**options, "format": video_platform._video_format(left_out, limits.long_side())}, formats)

    def _pick_size(self, ladder, limits=_GENTLE):
        """The size of the video that the download within ``limits`` picks of ``ladder``."""
        return self._pick(_sized(ladder) + _PLATFORM_AUDIO, limits)[0]

    def test_the_options_of_the_limits(self):
        options, log = self._options(_GENTLE)
        self.assertEqual(options["ratelimit"], 2000 * 1024)  # bytes per second
        self.assertEqual(options["format_sort"], ["res:1080", "fps:60"])
        # First the formats that fit the 16:9 frame of 1080p on their long side
        self.assertEqual(options["format"], video_platform._video_format((), 1928))
        self.assertEqual(options["format"], "bestvideo[ext=mp4][format_note!*=?Premium][width<=?1928][height<=?1928]"
                                            "+bestaudio/" + video_platform._VIDEO_FORMAT)
        self.assertIn("Downloading Video with Audio (up to 1080p, 60 fps, 2000 KiB/s)\n", log)

    def test_without_limits_the_options_leave_yt_dlp_as_it_is(self):
        """No rate limit at all (yt-dlp divides by it, 0 would fail) and its own order."""
        options, log = self._options(video_platform.DownloadLimits())
        self.assertIsNone(options["ratelimit"])
        self.assertEqual(options["format_sort"], [])
        self.assertEqual(options["format"], video_platform._VIDEO_FORMAT)
        self.assertIn("Downloading Video with Audio\n", log)

    def test_the_long_side_of_the_frame(self):
        limits = video_platform.DownloadLimits
        self.assertEqual(limits(max_resolution=1080).long_side(), 1928)  # 1920 and the margin
        self.assertEqual(limits(max_resolution=480).long_side(), 862)  # 853.3 up to 854: 856 fits too
        self.assertEqual(limits(rate_limit=500, max_fps=30).long_side(), 0)

    def test_one_limit_alone(self):
        limits = video_platform.DownloadLimits
        self.assertEqual(limits(max_resolution=720).format_sort(), ["res:720"])
        # The resolution comes first also without its limit: the frame rate picks among its formats
        self.assertEqual(limits(max_fps=30).format_sort(), ["res", "fps:30"])
        self.assertEqual((limits(rate_limit=500).format_sort(), limits(rate_limit=500).ratelimit()), ([], 512000))
        self.assertEqual(limits(max_fps=30).describe(), "up to 30 fps")

    def test_1080p_of_a_video_up_to_2160p(self):
        self.assertEqual(self._pick(self._UP_TO_2160P + _PLATFORM_AUDIO), ["399", "251"])

    def test_without_limits_the_best_as_before(self):
        formats = self._UP_TO_2160P + _PLATFORM_AUDIO
        self.assertEqual(self._pick(formats, video_platform.DownloadLimits()), ["401", "251"])
        self.assertEqual(_picked({"format": video_platform._VIDEO_FORMAT}, formats), ["401", "251"])

    def test_the_smallest_resolution_above_when_none_is_at_or_below(self):
        """A soft limit: no video fails for it."""
        self.assertEqual(self._pick([_platform_video("401", 2160)] + _PLATFORM_AUDIO), ["401", "251"])
        self.assertEqual(self._pick([_platform_video("401", 2160), _platform_video("400", 1440)] + _PLATFORM_AUDIO),
                         ["400", "251"])

    def test_the_highest_frame_rate_up_to_the_limit(self):
        formats = [_platform_video("137", 1080, 30, "avc1.640028"), _platform_video("299", 1080, 60, "avc1.64002a"),
                   *_PLATFORM_AUDIO]
        self.assertEqual(self._pick(formats), ["299", "251"])
        self.assertEqual(self._pick(formats, video_platform.DownloadLimits(2000, 1080, 30)), ["137", "251"])

    def test_the_resolution_comes_before_the_frame_rate(self):
        """A video of 60 fps comes at 60 fps from 720p on: its 1080p is taken with a limit of 30 fps too."""
        formats = [_platform_video("397", 480, 30), _platform_video("398", 720, 60),
                   _platform_video("399", 1080, 60), *_PLATFORM_AUDIO]
        self.assertEqual(self._pick(formats, video_platform.DownloadLimits(2000, 1080, 30)), ["399", "251"])

    def test_the_codec_order_of_a_resolution_stays(self):
        """AV1, then VP9, then H.264, with the limits as without them."""
        formats = [_platform_video("137", 1080, vcodec="avc1.640028"), _platform_video("399", 1080),
                   _platform_video("614", 1080, vcodec="vp09.00.40.08", protocol="m3u8_native"), *_PLATFORM_AUDIO]
        for limits in (_GENTLE, video_platform.DownloadLimits()):
            with self.subTest(limits=limits):
                self.assertEqual(self._pick(formats, limits), ["399", "251"])
                self.assertEqual(self._pick(formats, limits, ["399"]), ["614", "251"])
                self.assertEqual(self._pick(formats, limits, ["399", "614"]), ["137", "251"])

    def test_a_format_left_out_gives_the_next_best_within_the_limits(self):
        formats = self._UP_TO_2160P + [_platform_video("137", 1080, vcodec="avc1.640028"), *_PLATFORM_AUDIO]
        self.assertEqual(self._pick(formats, left_out=["399"]), ["137", "251"])  # not 1440p
        self.assertEqual(self._pick(formats, left_out=["399", "137"]), ["398", "251"])  # the best below

    def test_premium_formats_are_still_skipped(self):
        premium = _platform_video("616", 1080, vcodec="vp09.00.40.08", format_note="1080p Premium",
                                  source_preference=99, protocol="m3u8_native")
        self.assertEqual(self._pick([_platform_video("399", 1080), premium, *_PLATFORM_AUDIO]), ["399", "251"])
        self.assertEqual(self._pick([premium, *_PLATFORM_AUDIO]), ["616", "251"])

    def test_1080p_of_videos_of_each_shape(self):
        """As the video platform names it. By the shorter side alone, the 1440p of the wide
        video (2560x1066) passed for 1080p; by the width first, 4:3, portrait and square
        videos came in 1440p."""
        expected = {"16:9": "1920x1080", "wide (2.39:1)": "1920x800", "4:3": "1440x1080",
                    "portrait": "1080x1920", "square": "1080x1080"}
        for shape, ladder in self._SHAPES.items():
            with self.subTest(shape=shape):
                self.assertEqual(self._pick_size(ladder), expected[shape])

    def test_480p_of_a_rounded_width(self):
        """853.3 pixels are 854, or 856 in some videos: still 480p, also of a portrait video."""
        limits = video_platform.DownloadLimits(2000, 480, 60)
        self.assertEqual(self._pick_size([(640, 360, 360), (856, 480, 480), (1280, 720, 720)], limits), "856x480")
        self.assertEqual(self._pick_size([(360, 640, 360), (480, 854, 480), (720, 1280, 720)], limits), "480x854")

    def test_the_smallest_resolution_above_also_for_other_shapes(self):
        self.assertEqual(self._pick_size([(2560, 1066, 1440), (3840, 1600, 2160)]), "2560x1066")
        self.assertEqual(self._pick_size([(1920, 1440, 1440), (2880, 2160, 2160)]), "1920x1440")
        self.assertEqual(self._pick_size([(1440, 2560, 1440), (2160, 3840, 2160)]), "1440x2560")

    def test_the_format_of_the_download_without_a_request(self):
        """For the GUI's quality badge: of the same formats, the video the download takes."""
        for limits in (_GENTLE, video_platform.DownloadLimits(), video_platform.DownloadLimits(0, 720, 30)):
            for shape, ladder in self._SHAPES.items():
                formats = _sized(ladder) + _PLATFORM_AUDIO
                with self.subTest(limits=limits, shape=shape), redirect_stdout(io.StringIO()) as out, \
                        contextlib.redirect_stderr(io.StringIO()) as err:
                    video = video_platform.video_format_of_download(
                        {"formats": formats, "_format_sort_fields": _PLATFORM_SORT}, limits)
                    self.assertEqual(video["format_id"], self._pick(formats, limits)[0])
                    self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))
        self.assertIsNone(video_platform.video_format_of_download({"formats": []}, _GENTLE))
        self.assertIsNone(video_platform.video_format_of_download({"formats": _PLATFORM_AUDIO}, _GENTLE))

    def test_the_download_keeps_to_the_rate_limit(self):
        """With yt-dlp and a local server: 48 KiB at 32 KiB/s take 1.5 s."""
        files = {"/v": b"v" * 48 * 1024}
        with _media_server(files) as (_, base), tempfile.TemporaryDirectory() as folder:
            start = time.monotonic()
            path, _ = TestFormatFallback._download(base, [TestFormatFallback._format(base, "v", 360)], folder,
                                                   limits=video_platform.DownloadLimits(rate_limit=32))
            elapsed = time.monotonic() - start
            with open(path, "rb") as file:
                data = file.read()
        self.assertEqual(data, files["/v"])
        self.assertGreaterEqual(elapsed, 1.4)


class _MediaHandler(http.server.BaseHTTPRequestHandler):
    """Serves the files of its server, also from a byte on. A refused file breaks
    off once and then gets "503 Service Unavailable", as one format of a video did
    on the video platform's server (with ``at_once``, it gets 503 from the start)."""

    def do_GET(self):
        server = self.server
        path = self.path.split("?")[0]
        server.hits.append(path)
        server.ranges.append(self.headers.get("Range"))
        count = server.hits.count(path)
        body = server.files[path]
        # ``refused_until``: the request after which a refused file comes after all
        if path in server.refused and count <= server.refused_until:
            if count > 1 or server.at_once:
                self.send_error(503, "Service Unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:len(body) // 3])
            return
        start = re.fullmatch(r"bytes=(\d+)-\d*", self.headers.get("Range") or "")
        start = int(start.group(1)) if start else 0
        if start:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])

    def log_message(self, *args):
        pass


@contextmanager
def _media_server(files, refused=(), at_once=False, refused_until=float("inf")):
    """A local server of ``files`` (path -> bytes) and its address; ``hits``
    lists the paths it was asked for, ``ranges`` the Range header of each."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MediaHandler)
    server.files, server.refused, server.at_once, server.hits = files, set(refused), at_once, []
    server.refused_until, server.ranges = refused_until, []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class TestFormatFallback(unittest.TestCase):
    """A format that does not come, also after all retries, is left out: the
    download starts again with the next best one (yt-dlp and a local server)."""

    @staticmethod
    def _download(base, formats, folder, name="Artist - Title", limits=video_platform.DownloadLimits()):
        """The downloaded file, or the error of the download, and the console."""
        info = {"id": "vid", "title": "Video", "extractor": "test", "extractor_key": "Test",
                "webpage_url": f"{base}/watch", "formats": formats}

        def extract_info(ydl, url, download=True, **kwargs):
            return ydl.process_ie_result(copy.deepcopy(info), download=download)

        with patch.object(yt_dlp.YoutubeDL, "extract_info", extract_info), \
                patch.object(video_platform, "_retry_pause", return_value=0), \
                patch.object(video_platform, "_fragment_retry_pause", return_value=0), \
                redirect_stdout(io.StringIO()) as log:
            try:
                result = getattr(video_platform, "__download_video_with_audio")(
                    f"{base}/watch", name, folder, None, None, limits)
            except yt_dlp.utils.DownloadError as error:
                result = error
        return result, log.getvalue()

    @staticmethod
    def _format(base, format_id, height, **codecs):
        return {"format_id": format_id, "url": f"{base}/{format_id}", "ext": "mp4", "height": height,
                **(codecs or {"vcodec": "avc1.42001E", "acodec": "mp4a.40.2"})}

    def test_the_next_best_format_after_one_that_does_not_come(self):
        files = {"/hd": b"h" * 30000, "/sd": b"s" * 20000}
        with _media_server(files, refused={"/hd"}) as (server, base), tempfile.TemporaryDirectory() as folder:
            path, log = self._download(
                base, [self._format(base, "sd", 360), self._format(base, "hd", 720)], folder)
            with open(path, "rb") as file:
                data = file.read()
            left = os.listdir(folder)
        # Both formats have the same file name: the second does not go on from what came of the first
        self.assertEqual(data, files["/sd"])
        self.assertEqual(left, ["Artist - Title.mp4"])
        self.assertEqual(server.hits.count("/hd"), 11)  # the download and its 10 retries
        self.assertEqual(server.hits[-1], "/sd")
        self.assertIn("format hd (720p) could not be downloaded - downloading the next best format instead", log)
        self.assertIn("Downloading format sd (360p)", log)

    def test_the_partial_files_of_the_name_that_yt_dlp_writes(self):
        """yt-dlp expands an environment variable in the name it is given."""
        files = {"/hd": b"h" * 30000, "/sd": b"s" * 20000}
        with _media_server(files, refused={"/hd"}) as (_, base), tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"ULTRASINGER_TEST_NAME": "Title"}):
            path, _ = self._download(base, [self._format(base, "sd", 360), self._format(base, "hd", 720)],
                                     folder, name="Artist - $ULTRASINGER_TEST_NAME")
            with open(path, "rb") as file:
                data = file.read()
            left = os.listdir(folder)
        self.assertEqual(data, files["/sd"])
        self.assertEqual(left, ["Artist - Title.mp4"])

    def test_a_fragment_that_does_not_come_ends_its_format(self):
        """Instead of a gap in the video: the next best format comes in full, and nothing
        of the fragmented one is left (yt-dlp leaves its file open)."""
        files = {"/f1": b"1" * 30000, "/f2": b"2" * 30000, "/sd": b"s" * 20000}
        with _media_server(files, refused={"/f2"}) as (server, base), tempfile.TemporaryDirectory() as folder:
            fragmented = {**self._format(base, "hd", 720), "protocol": "http_dash_segments",
                          "fragments": [{"url": f"{base}/f1"}, {"url": f"{base}/f2"}]}
            path, log = self._download(base, [self._format(base, "sd", 360), fragmented], folder)
            with open(path, "rb") as file:
                data = file.read()
            left = os.listdir(folder)
        self.assertEqual(data, files["/sd"])
        self.assertEqual(left, ["Artist - Title.mp4"])
        self.assertEqual(server.hits.count("/f2"), 11)  # the fragment and its 10 retries
        self.assertIn("format hd (720p) could not be downloaded", log)

    def test_a_fragmented_format_refused_from_the_start_leaves_nothing(self):
        """No data came, so no progress was reported: its empty partial file and its
        state file go all the same."""
        files = {"/f1": b"1" * 30000, "/sd": b"s" * 20000}
        with _media_server(files, refused={"/f1"}, at_once=True) as (server, base), \
                tempfile.TemporaryDirectory() as folder:
            fragmented = {**self._format(base, "hd", 720), "protocol": "http_dash_segments",
                          "fragments": [{"url": f"{base}/f1"}]}
            path, _ = self._download(base, [self._format(base, "sd", 360), fragmented], folder)
            with open(path, "rb") as file:
                data = file.read()
            left = os.listdir(folder)
        self.assertEqual(data, files["/sd"])
        self.assertEqual(left, ["Artist - Title.mp4"])

    @staticmethod
    def _media():
        """A second of video without audio (mp4) and of audio (m4a), made by ffmpeg."""
        with tempfile.TemporaryDirectory() as media:
            video, audio = os.path.join(media, "v.mp4"), os.path.join(media, "a.m4a")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1",
                            "-c:v", "mpeg4", video], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac", audio],
                           check=True)
            with open(video, "rb") as v, open(audio, "rb") as a:
                return v.read(), a.read()

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_the_same_video_with_the_next_best_audio(self):
        """The video came before the audio failed: it is not downloaded again."""
        video_data, audio_data = self._media()
        files = {"/v": video_data, "/a160": audio_data, "/a64": audio_data}
        with _media_server(files, refused={"/a160"}) as (server, base), tempfile.TemporaryDirectory() as folder:
            path, log = self._download(base, [
                {**self._format(base, "a64", None, vcodec="none", acodec="mp4a.40.2"), "ext": "m4a", "abr": 64},
                {**self._format(base, "a160", None, vcodec="none", acodec="mp4a.40.2"), "ext": "m4a", "abr": 160},
                self._format(base, "v", 720, vcodec="mp4v.20.3", acodec="none"),
            ], folder)
            self.assertEqual(path, os.path.join(folder, "Artist - Title.mp4"))
            left = os.listdir(folder)
        self.assertEqual(left, ["Artist - Title.mp4"])
        self.assertEqual([hit for hit in server.hits if hit != "/a160"], ["/v", "/a64"])
        self.assertIn("Downloading format v (720p) + a64 (audio only)", log)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_the_next_best_video_with_the_same_audio(self):
        video_data, audio_data = self._media()
        files = {"/1080": video_data, "/720": video_data, "/a": audio_data}
        with _media_server(files, refused={"/1080"}) as (server, base), tempfile.TemporaryDirectory() as folder:
            path, log = self._download(base, [
                {**self._format(base, "a", None, vcodec="none", acodec="mp4a.40.2"), "ext": "m4a"},
                self._format(base, "720", 720, vcodec="mp4v.20.3", acodec="none"),
                self._format(base, "1080", 1080, vcodec="mp4v.20.3", acodec="none"),
            ], folder)
            self.assertEqual(path, os.path.join(folder, "Artist - Title.mp4"))
            left = os.listdir(folder)
        # Merged, and nothing of the video that did not come is left
        self.assertEqual(left, ["Artist - Title.mp4"])
        self.assertEqual([hit for hit in server.hits if hit != "/1080"], ["/720", "/a"])
        self.assertIn("format 1080 (1080p) could not be downloaded", log)
        self.assertIn("Downloading format 720 (720p) + a (audio only)", log)

    def test_the_error_stands_after_the_last_format_left_out(self):
        files = {f"/{n}": bytes([n]) * 3000 for n in range(1, 6)}
        with _media_server(files, refused=set(files)) as (server, base), tempfile.TemporaryDirectory() as folder:
            error, log = self._download(
                base, [self._format(base, str(n), 100 * n) for n in range(1, 6)], folder)
        self.assertIsInstance(error, yt_dlp.utils.DownloadError)
        self.assertIn("HTTP Error 503", str(error))
        # The best format and the next best 3
        self.assertEqual(sorted(set(server.hits)), ["/2", "/3", "/4", "/5"])
        self.assertEqual(log.count("could not be downloaded"), video_platform._MAX_FORMAT_FALLBACKS)


class TestFormatFallbackOnlyForAFormat(unittest.TestCase):
    """An error that is no format's ends the download at once."""

    _NOT_DELIVERED = "ERROR: \r[download] Got error: HTTP Error 503: Service Unavailable. Giving up after 10 retries"

    def _attempts(self, extract_info):
        """How often the download starts when ``extract_info`` (given the download's
        formats and the number of the try) fails, and the error it ends with."""
        with patch.object(video_platform.yt_dlp, "YoutubeDL") as ydl, \
                tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            instance = ydl.return_value.__enter__.return_value
            instance.extract_info.side_effect = lambda url, download: extract_info(
                instance.add_post_processor.call_args.args[0], instance.extract_info.call_count)
            with self.assertRaises(yt_dlp.utils.DownloadError) as raised:
                getattr(video_platform, "__download_video_with_audio")("https://fakeUrl", "Artist - Title", folder)
        self.assertEqual(instance.add_post_processor.call_args.kwargs, {"when": "before_dl"})
        return instance.extract_info.call_count, str(raised.exception)

    @staticmethod
    def _picked(formats):
        formats.run({"requested_formats": [{"format_id": "614"}, {"format_id": "251"}]})

    def test_an_error_before_a_format_was_picked(self):
        def extract_info(formats, attempt):
            raise yt_dlp.utils.DownloadError("ERROR: [test] vid: Video unavailable")
        self.assertEqual(self._attempts(extract_info)[0], 1)

    def test_an_error_after_all_formats_came(self):
        def extract_info(formats, attempt):
            self._picked(formats)
            for format_id in ("614", "251"):
                formats.progress({"status": "finished", "info_dict": {"format_id": format_id}})
            raise yt_dlp.utils.DownloadError("ERROR: Postprocessing: Conversion failed!")
        self.assertEqual(self._attempts(extract_info)[0], 1)

    def test_a_file_that_cannot_be_written(self):
        def extract_info(formats, attempt):
            self._picked(formats)
            error = OSError(28, "No space left on device")
            raise yt_dlp.utils.DownloadError(f"ERROR: unable to write data: {error}", (OSError, error, None))
        self.assertEqual(self._attempts(extract_info)[0], 1)

    def test_a_file_that_cannot_be_renamed(self):
        """yt-dlp reports it without its cause, after its own retries."""
        def extract_info(formats, attempt):
            self._picked(formats)
            raise yt_dlp.utils.DownloadError(
                "ERROR: Unable to rename file: [WinError 32] The process cannot access the file")
        self.assertEqual(self._attempts(extract_info)[0], 1)

    def test_no_ffmpeg_to_merge(self):
        def extract_info(formats, attempt):
            self._picked(formats)
            raise yt_dlp.utils.DownloadError("ERROR: You have requested merging of multiple formats but "
                                             "ffmpeg is not installed. Aborting due to --abort-on-error")
        self.assertEqual(self._attempts(extract_info)[0], 1)

    def test_an_error_of_a_format_starts_the_download_again(self):
        def extract_info(formats, attempt):
            self._picked(formats)
            raise yt_dlp.utils.DownloadError(self._NOT_DELIVERED)
        self.assertEqual(self._attempts(extract_info),
                         (1 + video_platform._MAX_FORMAT_FALLBACKS, self._NOT_DELIVERED))

    def test_the_error_of_the_format_stands_when_no_other_format_is_left(self):
        def extract_info(formats, attempt):
            if attempt == 1:
                self._picked(formats)
                raise yt_dlp.utils.DownloadError(self._NOT_DELIVERED)
            raise yt_dlp.utils.DownloadError("ERROR: [test] vid: Requested format is not available")
        self.assertEqual(self._attempts(extract_info), (2, self._NOT_DELIVERED))

    def test_a_partial_file_that_cannot_be_deleted(self):
        """The next format could go on from it: no other format is tried."""
        def extract_info(formats, attempt):
            self._picked(formats)
            raise yt_dlp.utils.DownloadError(self._NOT_DELIVERED)
        stuck = [("Artist - Title.mp4.part", PermissionError(13, "locked"), True)]
        with patch.object(video_platform, "_remove_partial_files", return_value=stuck):
            self.assertEqual(self._attempts(extract_info), (1, self._NOT_DELIVERED))

    def test_a_partial_file_of_the_format_left_out_that_cannot_be_deleted(self):
        """No later try takes it up: the next format is tried."""
        def extract_info(formats, attempt):
            self._picked(formats)
            raise yt_dlp.utils.DownloadError(self._NOT_DELIVERED)
        stuck = [("Artist - Title.f614.mp4.part", PermissionError(13, "locked"), False)]
        with patch.object(video_platform, "_remove_partial_files", return_value=stuck):
            self.assertEqual(self._attempts(extract_info)[0], 1 + video_platform._MAX_FORMAT_FALLBACKS)


class TestClientIp(unittest.TestCase):
    """The address that a stream URL is made out to tells IPv4 from IPv6."""

    def test_in_the_query(self):
        url = "https://stream.example/videoplayback?expire=1&ip=2001%3Adb8%3A%3A1&id=o-A"
        self.assertEqual(video_platform._client_ip(url), "2001:db8::1")

    def test_in_the_path_of_a_playlist(self):
        url = "https://playlist.example/api/manifest/hls_playlist/expire/1/ip/2001:db8::1/id/x/index.m3u8"
        self.assertEqual(video_platform._client_ip(url), "2001:db8::1")

    def test_none(self):
        self.assertEqual(video_platform._client_ip("https://video.example/v.mp4"), "")
        self.assertEqual(video_platform._client_ip("https://video.example/ip"), "")
        self.assertEqual(video_platform._client_ip(None), "")


class TestIpv4Retry(unittest.TestCase):
    """A format that came over IPv6 is tried once more over IPv4 before the next format."""

    _NOT_DELIVERED = "ERROR: \r[download] Got error: HTTP Error 503: Service Unavailable. Giving up after 10 retries"

    def _tries(self, ip, comes_on=None, unreachable_on=(), ended=None):
        """The format string and source address of each try when the try ``comes_on``
        (counted from 1) comes, or none; the tries ``unreachable_on`` fail before they pick
        their formats. ``ip`` is the address in the stream URLs, or a list of them, one
        for each try. And the console. ``ended`` gets the error the download ended with
        and the formats each try was to go on from."""
        with patch.object(video_platform.yt_dlp, "YoutubeDL") as ydl, \
                tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as log:
            instance = ydl.return_value.__enter__.return_value
            video = os.path.join(folder, "Artist - Title.mp4")

            def extract_info(url, download):
                if instance.extract_info.call_count in unreachable_on:
                    raise yt_dlp.utils.DownloadError("ERROR: [test] vid: Unable to download API page")
                address = ip[instance.extract_info.call_count - 1] if isinstance(ip, list) else ip
                instance.add_post_processor.call_args.args[0].run({"requested_formats": [
                    {"format_id": "399", "url": f"https://v.example/videoplayback?ip={address}"},
                    {"format_id": "251", "url": f"https://a.example/videoplayback?ip={address}"}]})
                if instance.extract_info.call_count == comes_on:
                    open(video, "wb").close()
                    return {"requested_downloads": [{"filepath": video}]}
                raise yt_dlp.utils.DownloadError(self._NOT_DELIVERED)

            instance.extract_info.side_effect = extract_info
            try:
                getattr(video_platform, "__download_video_with_audio")("https://fakeUrl", "Artist - Title", folder)
            except yt_dlp.utils.DownloadError as error:
                if ended is not None:
                    ended["error"] = str(error)
            if ended is not None:
                ended["goes_on_from"] = [c.args[0]._goes_on_from for c in instance.add_post_processor.call_args_list]
        return [(c.args[0]["format"], c.args[0].get("source_address")) for c in ydl.call_args_list], log.getvalue()

    def test_a_format_over_ipv6_is_tried_again_over_ipv4(self):
        tries, log = self._tries("2001:db8::1", comes_on=2)
        self.assertEqual(tries, [(video_platform._VIDEO_FORMAT, None), (video_platform._VIDEO_FORMAT, "0.0.0.0")])
        self.assertIn("format 399 could not be downloaded over IPv6 - trying it again over IPv4", log)
        self.assertIn("Downloading format 399 + 251", log)

    def test_a_format_over_ipv4_is_left_out_at_once(self):
        tries, _ = self._tries("192.0.2.1", comes_on=2)
        self.assertEqual(tries, [(video_platform._VIDEO_FORMAT, None), (video_platform._video_format(["399"]), None)])

    def test_the_next_formats_come_over_ipv4_as_well(self):
        tries, _ = self._tries("2001:db8::1")
        self.assertEqual([address for _, address in tries],
                         [None] + ["0.0.0.0"] * (1 + video_platform._MAX_FORMAT_FALLBACKS))

    def test_without_ipv4_the_next_format_comes_as_before(self):
        """The try over IPv4 does not get as far as its formats: no IPv4 here."""
        tries, log = self._tries("2001:db8::1", comes_on=3, unreachable_on=(2,))
        self.assertEqual(tries, [(video_platform._VIDEO_FORMAT, None), (video_platform._VIDEO_FORMAT, "0.0.0.0"),
                                 (video_platform._video_format(["399"]), None)])
        self.assertIn("the video could not be reached over IPv4 - going on as before", log)

    def test_the_last_format_left_to_try_is_tried_over_ipv4_too(self):
        """IPv6 can come and go on a network: three formats came over IPv4 and were left
        out, the fourth came over IPv6. The limit is on the formats left out."""
        fallbacks = video_platform._MAX_FORMAT_FALLBACKS
        ips = ["192.0.2.1"] * fallbacks + ["2001:db8::1"] * 2
        tries, log = self._tries(ips, comes_on=fallbacks + 2)
        self.assertEqual([address for _, address in tries], [None] * (fallbacks + 1) + ["0.0.0.0"])
        self.assertEqual(tries[-1][0], tries[-2][0])  # the same formats again
        self.assertIn("could not be downloaded over IPv6 - trying it again over IPv4", log)

    def test_at_the_limit_an_ipv4_try_without_formats_ends_the_download(self):
        fallbacks = video_platform._MAX_FORMAT_FALLBACKS
        ended = {}
        tries, log = self._tries(["192.0.2.1"] * fallbacks + ["2001:db8::1"], unreachable_on=(fallbacks + 2,),
                                 ended=ended)
        self.assertEqual(len(tries), fallbacks + 2)
        self.assertEqual(ended["error"], self._NOT_DELIVERED)
        self.assertNotIn("going on as before", log)

    def test_only_the_try_over_ipv4_goes_on_from_the_try_before(self):
        ended = {}
        self._tries("2001:db8::1", ended=ended)
        self.assertEqual(ended["goes_on_from"], [None, ["399", "251"]] + [None] * video_platform._MAX_FORMAT_FALLBACKS)

    def test_a_later_try_that_picks_no_format_ends_the_download(self):
        """After the try over IPv4 the next one fails before it picks its formats: the
        error of the format that did not come stands, and IPv4 is kept."""
        ended = {}
        tries, log = self._tries("2001:db8::1", unreachable_on=(3,), ended=ended)
        self.assertEqual([address for _, address in tries], [None, "0.0.0.0", "0.0.0.0"])
        self.assertEqual(ended["error"], self._NOT_DELIVERED)
        self.assertNotIn("could not be reached over IPv4", log)

    def test_over_ipv4_the_format_goes_on_from_where_it_stopped(self):
        """With yt-dlp and a local server that refuses the format the 11 times of its first try."""
        files = {"/hd": b"h" * 30000, "/sd": b"s" * 20000}
        with _media_server(files, refused={"/hd"}, refused_until=11) as (server, base), \
                tempfile.TemporaryDirectory() as folder:
            hd = {**TestFormatFallback._format(base, "hd", 720), "url": f"{base}/hd?ip=2001:db8::1"}
            path, log = TestFormatFallback._download(base, [TestFormatFallback._format(base, "sd", 360), hd], folder)
            with open(path, "rb") as file:
                data = file.read()
            left = os.listdir(folder)
        self.assertEqual(data, files["/hd"])
        self.assertEqual(left, ["Artist - Title.mp4"])
        self.assertNotIn("/sd", server.hits)
        self.assertIn("over IPv6 - trying it again over IPv4", log)
        # The 12th request went on from what came of the third that was sent
        self.assertEqual(server.hits.count("/hd"), 12)
        start = re.fullmatch(r"bytes=(\d+)-", server.ranges[-1] or "")
        self.assertTrue(start and 0 < int(start.group(1)) <= len(files["/hd"]) // 3, server.ranges[-1])

    def test_over_ipv4_another_format_does_not_go_on_from_the_first(self):
        """The extraction over IPv4 picks another single format, of the same file name."""
        files = {"/a": b"a" * 30000, "/b": b"b" * 20000}
        with _media_server(files, refused={"/a"}) as (_, base), tempfile.TemporaryDirectory() as folder:
            first = {**TestFormatFallback._format(base, "a", 720), "url": f"{base}/a?ip=2001:db8::1"}
            infos = iter([[first], [TestFormatFallback._format(base, "b", 360)]])

            def extract_info(ydl, url, download=True, **kwargs):
                info = {"id": "vid", "title": "Video", "extractor": "test", "extractor_key": "Test",
                        "webpage_url": f"{base}/watch", "formats": next(infos)}
                return ydl.process_ie_result(info, download=download)

            with patch.object(yt_dlp.YoutubeDL, "extract_info", extract_info), \
                    patch.object(video_platform, "_retry_pause", return_value=0), redirect_stdout(io.StringIO()):
                path = getattr(video_platform, "__download_video_with_audio")(f"{base}/watch", "Artist - Title", folder)
            with open(path, "rb") as file:
                data = file.read()
        self.assertEqual(data, files["/b"])


class TestRemovePartialFiles(unittest.TestCase):
    """After a format failed, the partial files of the song go; what came in full stays."""

    def _remove(self, names, song, failing=None, left_out=("399",)):
        """The files left of ``names``, and those reported as not deleted with whether
        the next try could go on from them (``failing``: the error of each delete)."""
        with tempfile.TemporaryDirectory() as folder:
            for name in names:
                open(os.path.join(folder, name), "wb").close()
            with patch.object(video_platform.os, "remove", side_effect=failing) if failing else nullcontext():
                stuck = video_platform._remove_partial_files(os.path.join(folder, song), left_out)
            return sorted(os.listdir(folder)), sorted((os.path.basename(path), taken_up)
                                                      for path, _, taken_up in stuck)

    def test_the_partial_files_of_the_song(self):
        song = "Artist [Live] - Title"
        partial = [f"{song}.mp4.part", f"{song}.f399.mp4.part", f"{song}.f399.mp4.ytdl",
                   f"{song}.f399.mp4.part-Frag3", f"{song}.f399.mp4.part-Frag3.part"]
        kept = [f"{song}.f251.webm", "Other - Song.mp4.part"]
        self.assertEqual(self._remove(partial + kept, song), (sorted(kept), []))

    def test_only_the_end_of_a_name_counts(self):
        song = "Artist.part-Frag - Title.part"
        self.assertEqual(self._remove([f"{song}.mp4"], song), ([f"{song}.mp4"], []))

    def test_a_file_that_cannot_be_deleted(self):
        """No later try takes up a file of a format left out, also of one left out before."""
        names = ["Artist - Title.mp4.part", "Artist - Title.f399.mp4.part", "Artist - Title.f398.mp4.part",
                 "Artist - Title.f3990.mp4.part"]
        left, stuck = self._remove(names, "Artist - Title", PermissionError(13, "locked"), left_out=["399", "398"])
        self.assertEqual(left, sorted(names))
        self.assertEqual(stuck, [("Artist - Title.f398.mp4.part", False), ("Artist - Title.f399.mp4.part", False),
                                 ("Artist - Title.f3990.mp4.part", True), ("Artist - Title.mp4.part", True)])


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

            with patch.object(video_platform, "get_video_info",
                              return_value={"title": "Artist - Title", "language": "en"}), \
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


class TestFailedDownloadFolder(unittest.TestCase):
    """A download that fails takes its new song folder along: the next try makes
    a folder of its own ("Artist - Title (1)", ...)."""

    _NOT_DELIVERED = "ERROR: \r[download] Got error: HTTP Error 503: Service Unavailable. Giving up after 10 retries"

    @contextmanager
    def _song(self, download=None, separate=None):
        """The video's song is "Artist - Title"; ``download`` and ``separate`` stand in
        for the download and for the separation of audio and video. Yields the console."""
        with ExitStack() as stack:
            stack.enter_context(patch.object(video_platform, "get_video_info",
                                             return_value={"title": "Artist - Title", "language": "en"}))
            stack.enter_context(patch.object(video_platform, "search_musicbrainz",
                                             return_value=SongInfo(title="Title", artist="Artist")))
            if download:
                stack.enter_context(patch.object(video_platform, "__download_video_with_audio", side_effect=download))
            if separate:
                stack.enter_context(patch.object(video_platform, "separate_audio_video", side_effect=separate))
            yield stack.enter_context(redirect_stdout(io.StringIO()))

    def test_the_folder_of_a_failed_download_goes(self):
        def download(url, name, folder, *args):
            open(os.path.join(folder, f"{name}.f399.mp4.part"), "wb").close()
            raise yt_dlp.utils.DownloadError(self._NOT_DELIVERED)

        with tempfile.TemporaryDirectory() as out:
            with self._song(download) as log, self.assertRaises(yt_dlp.utils.DownloadError) as raised:
                video_platform.download_from_video_platform("https://fakeUrl", out)
            self.assertEqual(os.listdir(out), [])
        self.assertEqual(str(raised.exception), self._NOT_DELIVERED)
        self.assertIn("Removed the folder of the failed download", log.getvalue())

    def test_a_folder_with_other_files_stays(self):
        def download(url, name, folder, *args):
            open(os.path.join(folder, "notes.txt"), "w").close()
            raise yt_dlp.utils.DownloadError(self._NOT_DELIVERED)

        with tempfile.TemporaryDirectory() as out:
            with self._song(download) as log, self.assertRaises(yt_dlp.utils.DownloadError):
                video_platform.download_from_video_platform("https://fakeUrl", out)
            self.assertEqual(os.listdir(os.path.join(out, "Artist - Title")), ["notes.txt"])
        self.assertIn("holds other files and stays", log.getvalue())

    @staticmethod
    def _downloaded(url, name, folder, *args):
        path = os.path.join(folder, f"{name}.mp4")
        open(path, "wb").close()
        return path

    def test_a_failed_separation_of_audio_and_video(self):
        def separate(path, name, folder, **kwargs):
            # Its files of the audio and of the video without it
            for file in (f"{name}.ogg", f"{name}_video.mp4"):
                open(os.path.join(folder, file), "wb").close()
            raise RuntimeError("ffmpeg failed")

        with tempfile.TemporaryDirectory() as out:
            with self._song(self._downloaded, separate), self.assertRaises(RuntimeError):
                video_platform.download_from_video_platform("https://fakeUrl", out)
            self.assertEqual(os.listdir(out), [])

    def test_a_folder_with_a_chart_stays(self):
        """A sync client can bring a song of the same name in from another computer."""
        for chart in ("Artist - Title.txt", "Artist - Title.TXT"):
            def download(url, name, folder, *args, chart=chart):
                open(os.path.join(folder, chart), "w").close()
                raise yt_dlp.utils.DownloadError(self._NOT_DELIVERED)

            with self.subTest(chart), tempfile.TemporaryDirectory() as out:
                with self._song(download), self.assertRaises(yt_dlp.utils.DownloadError):
                    video_platform.download_from_video_platform("https://fakeUrl", out)
                self.assertEqual(os.listdir(os.path.join(out, "Artist - Title")), [chart])

    def test_a_folder_that_cannot_be_removed(self):
        def download(url, name, folder, *args):
            raise yt_dlp.utils.DownloadError(self._NOT_DELIVERED)

        with tempfile.TemporaryDirectory() as out:
            with self._song(download) as log, \
                    patch.object(video_platform.shutil, "rmtree", side_effect=PermissionError(13, "locked")), \
                    self.assertRaises(yt_dlp.utils.DownloadError):
                video_platform.download_from_video_platform("https://fakeUrl", out)
        self.assertIn("the folder of the failed download could not be removed", log.getvalue())

    def test_a_cover_that_cannot_be_saved(self):
        """The download came: the song goes on with the video's thumbnail as its cover."""
        song = SongInfo(title="Title", artist="Artist", cover_url="https://cover.example/c.jpg",
                        cover_image_data=b"no image")
        with tempfile.TemporaryDirectory() as out:
            with self._song(self._downloaded, separate=lambda *args, **kwargs: ("a.ogg", "v.mp4", "ogg", "mp4")) \
                    as log, patch.object(video_platform, "search_musicbrainz", return_value=song), \
                    patch.object(video_platform, "__download_thumbnail", return_value="https://thumb.example/t.jpg"):
                *_, media_info = video_platform.download_from_video_platform("https://fakeUrl", out)
            self.assertEqual(os.listdir(out), ["Artist - Title"])
        self.assertEqual(media_info.cover_url, "https://thumb.example/t.jpg")
        self.assertIn("the cover from the song database could not be saved", log.getvalue())

    def test_a_cancelled_download(self):
        def download(url, name, folder, *args):
            open(os.path.join(folder, f"{name}.mp4.part"), "wb").close()
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as out:
            with self._song(download), self.assertRaises(KeyboardInterrupt):
                video_platform.download_from_video_platform("https://fakeUrl", out)
            self.assertEqual(os.listdir(out), [])

    def test_a_fragmented_format_that_ends_the_download(self):
        """yt-dlp leaves the file of the fragmented format open while its error is on the way
        (yt-dlp and a local server)."""
        files = {"/f1": b"1" * 30000, "/f2": b"2" * 30000}
        with _media_server(files, refused={"/f2"}) as (_, base), \
                tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as out:
            fragmented = {**TestFormatFallback._format(base, "hd", 720), "protocol": "http_dash_segments",
                          "fragments": [{"url": f"{base}/f1"}, {"url": f"{base}/f2"}]}
            info = {"id": "vid", "title": "Video", "extractor": "test", "extractor_key": "Test",
                    "webpage_url": f"{base}/watch", "formats": [fragmented]}

            def extract_info(ydl, url, download=True, **kwargs):
                return ydl.process_ie_result(copy.deepcopy(info), download=download)

            with self._song(), patch.object(yt_dlp.YoutubeDL, "extract_info", extract_info), \
                    patch.object(video_platform, "_retry_pause", return_value=0), \
                    patch.object(video_platform, "_fragment_retry_pause", return_value=0), \
                    patch.object(video_platform, "_not_delivered", return_value=False), \
                    self.assertRaises(yt_dlp.utils.DownloadError):
                # Taken for no format's error, it goes on as it came, with its frames
                video_platform.download_from_video_platform(f"{base}/watch", out)
            left = os.listdir(out)
        self.assertEqual(left, [])


class TestSongLookup(unittest.TestCase):
    """The song database is asked with the length of the video, which comes with its title."""

    def _lookup(self, info):
        """The song lookup and the info extraction of a download of the video of ``info``."""
        with tempfile.TemporaryDirectory() as out, \
                patch.object(video_platform.yt_dlp, "YoutubeDL") as ydl, \
                patch.object(video_platform, "search_musicbrainz",
                             return_value=SongInfo(title="Title", artist="Artist")) as lookup, \
                patch.object(video_platform, "__download_video_with_audio",
                             side_effect=TestFailedDownloadFolder._downloaded), \
                patch.object(video_platform, "separate_audio_video",
                             return_value=("a.ogg", "v.mp4", "ogg", "mp4")), \
                patch.object(video_platform, "__download_thumbnail", return_value=""), \
                redirect_stdout(io.StringIO()):
            ydl.return_value.__enter__.return_value.extract_info.return_value = info
            video_platform.download_from_video_platform("https://fakeUrl", out)
        return lookup, ydl.return_value.__enter__.return_value.extract_info

    def test_the_length_of_the_video(self):
        lookup, extract_info = self._lookup({"title": "Artist - Title - Chapter 2", "duration": 238})
        lookup.assert_called_once_with("Title - Chapter 2", "Artist", quick=False, duration=238)
        # The title and the length come from one extraction
        extract_info.assert_called_once_with("https://fakeUrl", download=False)

    def test_an_unknown_length(self):
        for info in ({"title": "Artist - Title"}, {"title": "Artist - Title", "duration": None},
                     {"title": "Artist - Title", "duration": 0}):
            with self.subTest(info=info):
                lookup, _ = self._lookup(info)
                self.assertIsNone(lookup.call_args.kwargs["duration"])


if __name__ == "__main__":
    unittest.main()
