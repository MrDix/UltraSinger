"""Video Platform Downloader"""

import gc
import glob
import logging
import math
import os
import re
import shutil
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass

import yt_dlp
from yt_dlp.networking.exceptions import network_exceptions
from yt_dlp.postprocessor.common import PostProcessor
from yt_dlp.utils import ContentTooShortError

from modules.os_helper import sanitize_filename, get_unused_song_output_dir
from modules import os_helper
from modules.ProcessData import MediaInfo
from modules.Audio.bpm import get_bpm_from_file
from modules.console_colors import ULTRASINGER_HEAD, gold_highlighted, red_highlighted
from modules.Image.image_helper import save_image
from modules.musicbrainz_client import SongInfo, search_musicbrainz
from modules.name_matching import artist_from_channel
from modules.ffmpeg_helper import separate_audio_video


# Suffixes that the music service commonly adds to track names which may not
# reflect the actual video content.  Only these are considered for removal;
# arbitrary parenthetical metadata like "(feat. …)" or "(Part 2)" is kept.
_STRIPPABLE_SUFFIXES = re.compile(
    r"^("
    r"live(?:\s.+)?|"           # live, live at …, live from …
    r"acoustic(?:\s.+)?|"       # acoustic, acoustic version
    r"remaster(?:ed)?(?:\s.+)?|"  # remastered, remastered 2024
    r"(?:official\s)?(?:video|audio|music\svideo)|"
    r"lyrics?(?:\svideo)?|"     # lyric, lyrics, lyric video
    r"explicit|clean|"
    r"radio\s?edit|"
    r"single\sversion"
    r")$",
    re.IGNORECASE,
)


def strip_unmatched_suffixes(track: str, video_title: str, verbose: bool = True) -> str:
    """Remove trailing parenthetical suffixes from the music service track name
    that do not appear in the actual video title.

    The music service metadata sometimes adds qualifiers like '(live)' or
    '(acoustic)' that do not match the uploaded video.  By cross-checking
    against the video title we can strip these misleading suffixes while
    keeping legitimate ones like '(feat. …)'.

    Only *trailing* parenthesised groups whose content matches a known
    set of media qualifiers are considered, so arbitrary metadata like
    '(feat. Guest)' or '(Part 2)' is never touched. ``verbose`` reports
    each stripped suffix.
    """
    # Iteratively strip trailing (...) groups that are absent from the title
    while True:
        m = re.search(r'\s*\(([^)]+)\)\s*$', track)
        if not m:
            break
        suffix_content = m.group(1).strip()
        # Only consider known media qualifiers for stripping
        if not _STRIPPABLE_SUFFIXES.match(suffix_content):
            break  # not a known qualifier - leave it alone
        if re.search(
            rf"(?<!\w){re.escape(suffix_content)}(?!\w)",
            video_title,
            re.IGNORECASE,
        ):
            break  # suffix is present in video title - keep it
        if verbose:
            print(
                f"{ULTRASINGER_HEAD} {red_highlighted(f'Stripped music service suffix \"({suffix_content})\" - not in video title')}"
            )
        track = track[: m.start()].strip()
    return track


def _apply_po_token(ydl_opts: dict, po_token: str | None) -> dict:
    """Pass a browser-captured GVS PO token to yt-dlp.

    The video platform limits clients without a Proof-of-Origin token to a
    reduced format set (e.g. only 360p progressive) and blocks downloads
    with HTTP 403.  The GUI's embedded browser captures the token its own
    player sends; passing it through restores the full format list.
    """
    if po_token:
        ydl_opts["extractor_args"] = {
            "youtube": {"po_token": [f"web.gvs+{po_token}"]}
        }
    return ydl_opts


# The yt-dlp library does not retry by default (its command line retries 10
# times), so one connection that stalled for the socket timeout ended the
# download, or left a fragment of a fragmented stream out. Each retry goes on
# where the download stopped.
_DOWNLOAD_RETRIES = 10

# Longest pause before a retry (s): of the download, and of a fragment of a
# fragmented stream, which has many (a fragment that is gone for good costs
# all of its retries before its format is given up)
_MAX_RETRY_PAUSE_S = 30.0
_MAX_FRAGMENT_RETRY_PAUSE_S = 5.0


def _retry_pause(n: int) -> float:
    """Seconds to wait before retry ``n + 1`` of a download: 1, 2, 4, ... up to
    ``_MAX_RETRY_PAUSE_S``, about 3 minutes over the 10 retries.

    A server that cuts its connections ever earlier, or answers "503 Service
    Unavailable", wants fewer requests: retries in quick succession used up
    all 10 within a minute, and the download failed.
    """
    return min(2.0 ** n, _MAX_RETRY_PAUSE_S)


def _fragment_retry_pause(n: int) -> float:
    """Seconds to wait before retry ``n + 1`` of a fragment: as ``_retry_pause``,
    up to ``_MAX_FRAGMENT_RETRY_PAUSE_S``, 42 s over the 10 retries."""
    return min(2.0 ** n, _MAX_FRAGMENT_RETRY_PAUSE_S)


# Premium formats (higher-bitrate copies for subscribers) can come very slowly,
# with connections that stall again and again, even when the format picked in
# their place comes at a normal speed. So they are only taken when no other
# mp4 video is offered.
_VIDEO_FORMAT = "bestvideo[ext=mp4][format_note!*=?Premium]+bestaudio/bestvideo[ext=mp4]+bestaudio/best"

# A server can refuse one format of a video for hours ("503 Service
# Unavailable", also after all retries) while the site's player plays the video
# all the same. The download then starts again without that format, up to this
# many times: with the next best video or audio format, never with the audio
# alone.
_MAX_FORMAT_FALLBACKS = 3

# The format ids that a format string can leave out (the video platform's are
# numbers, some with a suffix such as "251-drc")
_FORMAT_ID = re.compile(r"[\w.-]+")

# What yt-dlp says when a server does not deliver a format, also after the
# retries. An error of this computer (a file that cannot be written or renamed,
# no ffmpeg to merge) says otherwise, and another format would not help.
_NOT_DELIVERED = ("[download] Got error:", "Did not get any data blocks",
                  "The downloaded file is empty", "not found, unable to continue")


def _not_delivered(error: yt_dlp.utils.DownloadError) -> bool:
    """Whether a download failed because the server did not deliver a format."""
    cause = (error.exc_info or (None, None, None))[1]
    return (isinstance(cause, (*network_exceptions, ContentTooShortError))
            or any(marker in str(error) for marker in _NOT_DELIVERED))


# Within a limit of resolution, the video is first taken of the mp4 formats
# (no premium ones) whose long side fits the 16:9 frame of that resolution:
# by its shorter side alone, the "1440p" of a wide video (2560x1066) passed for
# 1080p, and sorting by the width first takes the 1440p of a 4:3 or a portrait
# video. The margin is for an odd rounding (856 for 854 pixels).
_VIDEO_FORMAT_IN_FRAME = "bestvideo[ext=mp4][format_note!*=?Premium][width<=?{side}][height<=?{side}]+bestaudio"
_FRAME_MARGIN = 8


def _video_format(left_out: Sequence[str] = (), long_side: int = 0) -> str:
    """``_VIDEO_FORMAT`` without the formats ``left_out``: each of its parts
    takes the best format that is none of them. With ``long_side``, the formats
    no longer than that on either side come first."""
    # Quoted: yt-dlp compares a number as a number, and no format id (a
    # string) equals one, so "[format_id!=399]" left no format out
    skip = "".join(f"[format_id!='{format_id}']" for format_id in left_out)
    format_spec = (_VIDEO_FORMAT_IN_FRAME.format(side=long_side) + "/" if long_side else "") + _VIDEO_FORMAT
    return "/".join("+".join(f"{part}{skip}" for part in alternative.split("+"))
                    for alternative in format_spec.split("/"))


@dataclass(frozen=True)
class DownloadLimits:
    """Limits that keep the download of a video gentle, 0 for no limit: the
    video platform can slow down or block a connection that downloads a lot
    (also for watching videos on the other devices of the network).

    ``rate_limit`` is the highest speed in KiB/s. ``max_resolution`` (as in
    "1080p": up to 1920x1080, or 1080x1920 for a portrait video) and
    ``max_fps`` are soft limits, so that no video fails for them: the best
    format at or below them is taken, else the smallest one above. The frame
    rate picks among the formats of the resolution.
    """

    rate_limit: int = 0
    max_resolution: int = 0
    max_fps: int = 0

    def long_side(self) -> int:
        """The longest side of a video within the resolution limit: that of its
        16:9 frame (1920 for 1080), with the margin; 0 without the limit."""
        return math.ceil(self.max_resolution * 16 / 9) + _FRAME_MARGIN if self.max_resolution else 0

    def format_sort(self) -> list[str]:
        """yt-dlp's format sorting for the limits of resolution and frame rate;
        [] without them, which keeps yt-dlp's own order."""
        sort = [f"res:{self.max_resolution}"] if self.max_resolution else []
        if self.max_fps:
            # The resolution comes first, also without a limit
            sort = (sort or ["res"]) + [f"fps:{self.max_fps}"]
        return sort

    def ratelimit(self) -> int | None:
        """The rate limit in bytes per second for yt-dlp; None without one
        (yt-dlp divides by it, so 0 would fail)."""
        return self.rate_limit * 1024 or None

    def describe(self) -> str:
        """The limits as the console names them, e.g. "up to 1080p, 60 fps,
        2000 KiB/s"; "" for none."""
        limits = [text for value, text in ((self.max_resolution, f"{self.max_resolution}p"),
                                           (self.max_fps, f"{self.max_fps} fps"),
                                           (self.rate_limit, f"{self.rate_limit} KiB/s")) if value]
        return f"up to {', '.join(limits)}" if limits else ""


class _SilentLogger:
    """A yt-dlp logger that shows nothing (a format choice without a download)."""

    def debug(self, msg):
        pass

    info = warning = error = debug


def video_format_of_download(info: dict, limits: DownloadLimits) -> dict | None:
    """The video format that a download within ``limits`` takes of the formats
    of ``info`` (the yt-dlp info of a video), None for none. It sends no request:
    for the GUI's quality badge."""
    # As yt-dlp sorts and picks the formats of a download
    formats = [dict(f) for f in info.get("formats") or [] if not f.get("has_drm") or f["has_drm"] == "maybe"]
    with yt_dlp.YoutubeDL({"format_sort": limits.format_sort(), "quiet": True, "no_warnings": True,
                           "logger": _SilentLogger()}) as ydl:
        ydl.sort_formats({"formats": formats, "_format_sort_fields": info.get("_format_sort_fields")})
        selector = ydl.build_format_selector(_video_format((), limits.long_side()))
        picked = next(iter(selector({
            "formats": formats,
            "has_merged_format": any("none" not in (f.get("acodec"), f.get("vcodec")) for f in formats),
            "incomplete_formats": (all(f.get("vcodec") == "none" for f in formats)
                                   or all(f.get("acodec") == "none" for f in formats)),
        })), None)
    if picked is None:
        return None
    return next((f for f in picked.get("requested_formats") or [picked] if f.get("vcodec") != "none"), None)


def _client_ip(url: str) -> str:
    """The address of this computer that a stream URL of the video platform is
    made out to ("ip=" in its query, or "/ip/<address>/" in the path of a
    playlist), "" for none."""
    parsed = urllib.parse.urlparse(url or "")
    in_query = urllib.parse.parse_qs(parsed.query).get("ip")
    if in_query:
        return in_query[0]
    parts = parsed.path.split("/")
    return urllib.parse.unquote(parts[parts.index("ip") + 1]) if "ip" in parts[:-1] else ""


def _format_name(fmt: dict) -> str:
    """A format as the console names it, e.g. "399 (1920x1080)" or "251 (audio only)"."""
    resolution = fmt.get("resolution")
    return f"{fmt.get('format_id')} ({resolution})" if resolution else str(fmt.get("format_id"))


class _DownloadFormatsPP(PostProcessor):
    """The formats of a download, and which of them came in full.

    yt-dlp runs it before the download (``when="before_dl"``) with the formats
    it picked; ``progress`` is a progress hook of the download.
    """

    def __init__(self, announce: bool = False, goes_on_from: list[str] | None = None):
        super().__init__()
        self._announce = announce
        # The format ids of the try whose partial files this one goes on from
        self._goes_on_from = goes_on_from
        self.formats: list[dict] = []
        self._finished: set[str] = set()
        # The file that yt-dlp writes, without its extension (it expands
        # environment variables such as "$NAME" in the name it was given)
        self.stem = ""

    def run(self, info):
        # A video and an audio format to merge, or a single format
        self.formats = list(info.get("requested_formats") or [info])
        self.stem = os.path.splitext(info.get("_filename") or "")[0]
        if self._goes_on_from is not None and [f.get("format_id") for f in self.formats] != self._goes_on_from:
            # Other formats than before: nothing of those may be taken up (a
            # single format has no id in its file name)
            for path, reason, _ in _remove_partial_files(self.stem, ()):
                raise yt_dlp.utils.DownloadError(
                    f"the partial file {os.path.basename(path)} of another format could not be deleted ({reason})")
        if self._announce:
            print(f"{ULTRASINGER_HEAD} Downloading format "
                  f"{' + '.join(_format_name(f) for f in self.formats)}")
        return [], info

    def progress(self, status: dict):
        if status.get("status") == "finished":
            self._finished.add((status.get("info_dict") or {}).get("format_id"))

    def failed(self) -> dict | None:
        """The format that did not come in full, if any: yt-dlp downloads the
        formats one after the other and stops at the first that fails (one
        downloader for all of them, as for a live stream, counts as the first)."""
        return next((f for f in self.formats if f.get("format_id") not in self._finished), None)


def _remove_partial_files(stem: str, left_out: Sequence[str]) -> list[tuple[str, OSError, bool]]:
    """Delete the partial files of a download to ``stem`` plus an extension: of
    a format, with the fragments and the state file of a fragmented one. The
    formats that came in full stay, for the next try.

    Returns the files that could not be deleted, with the reason and whether the
    next try could go on from one (not from one of the formats ``left_out``,
    which it never takes).
    """
    stuck = []
    for path in glob.glob(f"{glob.escape(stem)}.*") if stem else ():
        # What follows the name, e.g. ".f399.mp4.part" (the name can hold anything)
        suffix = os.path.basename(path)[len(os.path.basename(stem)):]
        if suffix.endswith((".part", ".ytdl")) or ".part-Frag" in suffix:
            try:
                os.remove(path)
            except OSError as error:
                stuck.append((path, error, not any(suffix.startswith(f".f{f}.") for f in left_out)))
    return stuck


def get_video_info(url: str, cookiefile: str | None = None, po_token: str | None = None) -> dict:
    """The yt-dlp info of the video (its title, its length, ...), without
    downloading it."""
    ydl_opts = _apply_po_token({
        "cookiefile": cookiefile,
        "logger": _FilteredLogger(),
        # A watch URL that also names a playlist stands for its video only
        "noplaylist": True,
    }, po_token)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(
            url, download=False  # We just want to extract the info
        )


def get_video_title(url: str, cookiefile: str | None = None,
                     po_token: str | None = None) -> tuple[str, str, str, str]:
    """Get the title and language of the video.

    Returns:
        A 4-tuple of (artist, track, video_title, language), see
        ``video_title_from_info``.
    """
    return video_title_from_info(get_video_info(url, cookiefile, po_token))


def video_title_from_info(info: dict, verbose: bool = True) -> tuple[str, str, str, str]:
    """The title and language of a video from its yt-dlp info (the GUI's
    lyrics badge has that info already).

    Returns:
        A 4-tuple of (artist, track, video_title, language).
        *video_title* is the raw title string from the video's page,
        useful for cross-checking metadata from other sources.
        *language* is the ISO 639-1 code from yt-dlp metadata (e.g. "en"),
        or empty string if unavailable.
    ``verbose`` reports a suffix stripped from the track.
    """
    video_title = (info.get("title") or "").strip()
    artist = (info.get("artist") or "").strip()
    track = (info.get("track") or "").strip()
    language = (info.get("language") or "").strip()
    if artist and track:
        return artist, strip_unmatched_suffixes(track, video_title, verbose), video_title, language
    channel = (info.get("channel") or "").strip()
    if "-" in video_title:
        parts = video_title.split("-", 1)
        return parts[0].strip(), parts[1].strip(), video_title, language
    # "ArtistVEVO" or "Artist - Topic" is no artist name the lookups would find
    return artist_from_channel(channel), video_title, video_title, language


def find_song(artist: str, title: str, video_title: str, quick: bool = False,
              duration: float | None = None) -> SongInfo:
    """The song of a video, from the ``artist`` and ``title`` of
    ``video_title_from_info`` as the song database names them (``quick`` and
    ``duration``, the length of the video in seconds: see
    ``search_musicbrainz``)."""
    song_info = search_musicbrainz(title, artist, quick=quick, duration=duration)

    # The song database may re-introduce suffixes that were already stripped
    # from the yt-dlp metadata (e.g. "(live)" when the video is not actually
    # live). Cross-check its title against the video title as well.
    song_info.title = strip_unmatched_suffixes(song_info.title, video_title, verbose=not quick)
    return song_info


def _output_template(output_path: str, clear_filename: str) -> str:
    """yt-dlp output template for the file ``clear_filename`` in ``output_path``.

    yt-dlp reads "%(...)" as a field of the video's data ("100%(Live)" raised a
    KeyError), so a "%" of the path is written as "%%".
    """
    return f"{output_path}/{clear_filename}".replace("%", "%%") + ".%(ext)s"


def __download_video_with_audio(url: str, clear_filename: str, output_path: str, cookiefile: str = None,
                                 po_token: str = None, limits: DownloadLimits = DownloadLimits()) -> str:
    """Download video with audio from the video platform and return the path of the file"""

    note = limits.describe()
    print(f"{ULTRASINGER_HEAD} Downloading Video with Audio" + (f" ({note})" if note else ""))
    left_out: list[str] = []
    last_error = None
    ipv4 = False
    # The try over IPv6 that is tried again over IPv4: its failed format, the
    # ids of its formats and its file without extension
    over_ipv6: tuple[dict, list[str], str] | None = None
    retrying = False  # this try is that one again, over IPv4
    while True:
        ydl_opts = _apply_po_token({
            "format": _video_format(left_out, limits.long_side()),
            # The formats within the limits of resolution and frame rate first
            "format_sort": limits.format_sort(),
            "ratelimit": limits.ratelimit(),
            "outtmpl": _output_template(output_path, clear_filename),
            # Only for formats that are merged: a single file keeps its own (e.g. webm)
            "merge_output_format": "mp4",
            "retries": _DOWNLOAD_RETRIES,
            "fragment_retries": _DOWNLOAD_RETRIES,
            "retry_sleep_functions": {"http": _retry_pause, "fragment": _fragment_retry_pause},
            # A fragment that does not come ends the download of its format,
            # which the next best format then replaces, instead of a gap in the video
            "skip_unavailable_fragments": False,
            "cookiefile": cookiefile,
            "logger": _FilteredLogger(),
            "noplaylist": True,  # as in get_video_info
        }, po_token)
        if ipv4:
            ydl_opts["source_address"] = "0.0.0.0"  # IPv4 only, as yt-dlp's --force-ipv4
        formats = _DownloadFormatsPP(announce=bool(left_out) or retrying,
                                     goes_on_from=over_ipv6[1] if retrying else None)
        try:
            return __start_download(ydl_opts, url, formats)
        except yt_dlp.utils.DownloadError as error:
            if not formats.formats and last_error is not None:
                if not retrying or len(left_out) >= _MAX_FORMAT_FALLBACKS:
                    # No format is left to take (or the new try failed before one
                    # was picked): the error of the format that did not come stands
                    raise last_error
                # Over IPv4 the try did not get as far as its formats: IPv4 does
                # not work here, and the format is left out as it would have been
                ipv4 = False
                failed = over_ipv6[0]
                print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} the video could not be reached "
                      f"over IPv4 - going on as before")
            else:
                # No format failed when the error came after all of them (merging)
                failed = formats.failed() or {}
                if not _FORMAT_ID.fullmatch(str(failed.get("format_id") or "")) or not _not_delivered(error):
                    raise
                # Only its message: the error's frames keep files of the format open
                last_error = yt_dlp.utils.DownloadError(error.msg)
        retrying = False
        # yt-dlp leaves the partial file of a fragmented format open until the
        # frames of the error are collected
        gc.collect()
        format_id = str(failed["format_id"])
        if over_ipv6 is None and ":" in _client_ip(failed.get("url")):
            # The format came over IPv6, which some networks carry worse than
            # IPv4, and yt-dlp keeps to the address that answered first: the
            # same format is tried again over IPv4, from where it stopped
            ipv4 = retrying = True
            over_ipv6 = (failed, [f.get("format_id") for f in formats.formats], formats.stem)
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} format {_format_name(failed)} "
                  f"could not be downloaded over IPv6 - trying it again over IPv4")
            continue
        # The limit is on the formats left out, not on the one try over IPv4
        if len(left_out) >= _MAX_FORMAT_FALLBACKS:
            raise last_error
        left_out.append(format_id)
        # Without a format picked over IPv4, the files are those of the try over IPv6
        stuck = _remove_partial_files(formats.stem or (over_ipv6[2] if over_ipv6 else ""), left_out)
        for path, reason, taken_up in stuck:
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} the partial file {os.path.basename(path)} "
                  f"could not be deleted ({reason})"
                  + (" - no other format is tried, the next one could go on from it" if taken_up else ""))
        if any(taken_up for _, _, taken_up in stuck):
            raise last_error
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} format {_format_name(failed)} "
              f"could not be downloaded - downloading the next best format instead")


def __download_thumbnail(url: str, clear_filename: str, output_path: str, cookiefile: str = None,
                          po_token: str = None) -> str:
    """Download thumbnail from the video platform; its URL, or "" without one"""

    print(f"{ULTRASINGER_HEAD} Downloading thumbnail")
    ydl_opts = _apply_po_token({
        "skip_download": True,
        "writethumbnail": True,
        "cookiefile": cookiefile,
        "noplaylist": True,  # as in get_video_info
    }, po_token)

    try:
        thumbnail_url = download_and_convert_thumbnail(ydl_opts, url, clear_filename, output_path)
    except Exception as e:  # noqa: BLE001 - the cover is optional, the downloaded song is kept
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} the thumbnail could not be downloaded "
              f"({e}) - the song gets no cover")
        return ""
    return thumbnail_url


def download_and_convert_thumbnail(ydl_opts, url: str, clear_filename: str, output_path: str) -> str:
    """Download and convert thumbnail from the video platform"""

    ydl_opts.setdefault("logger", _FilteredLogger())
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        thumbnail_url = info_dict.get("thumbnail")
        if thumbnail_url:
            response = ydl.urlopen(thumbnail_url)
            image_data = response.read()
            save_image(image_data, clear_filename, output_path)
            return thumbnail_url
        else:
            return ""




_logger = logging.getLogger(__name__)

# yt-dlp warnings that are harmless and should not be shown to users
_SUPPRESSED_WARNINGS = [
    "SABR-only streaming experiment",
    "YouTube is forcing SABR streaming for this client",
    "SABR-only or Server-Side Ad Placement experiment",
    "SABR streaming",  # catch-all for future yt-dlp wording changes
]


class _FilteredLogger:
    """Custom yt-dlp logger that suppresses known harmless warnings and shows retries."""

    def debug(self, msg):
        # yt-dlp passes its screen messages here. Retries, the pauses before
        # them and the wait the site asks for before a download are shown, so
        # that a download that waits does not look hung.
        if msg.startswith("[download] Got error"):
            print(f"WARNING: {msg}")
            return
        if msg.startswith("Sleeping "):
            print(f"[download] {msg}")
            return
        if msg.startswith("[download] Sleeping "):
            print(msg)
            return
        _logger.debug(msg)

    def info(self, msg):
        print(msg)

    def warning(self, msg):
        if any(phrase in msg for phrase in _SUPPRESSED_WARNINGS):
            _logger.debug("Suppressed yt-dlp warning: %s", msg)
            return
        print(f"WARNING: {msg}")

    def error(self, msg):
        print(f"ERROR: {msg}")


def __start_download(ydl_opts, url: str, formats: _DownloadFormatsPP | None = None) -> str:
    """Start the download the ydl_opts; the path of the downloaded file
    (``formats`` learns the formats of the download and how far they came)"""
    ydl_opts.setdefault("logger", _FilteredLogger())
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        if formats is not None:
            ydl.add_post_processor(formats, when="before_dl")
            ydl.add_progress_hook(formats.progress)
        info = ydl.extract_info(url, download=True)
    # The file as it is after merging, with the extension of its real container
    downloads = (info or {}).get("requested_downloads") or []
    path = downloads[-1].get("filepath") if downloads else None
    if not path or not os.path.isfile(path):
        raise Exception(f"Download failed: no downloaded file ({path})")
    return path


def _remove_failed_song_folder(folder: str, basename: str):
    """Delete the song folder that a download made, when the download failed: a
    new try makes a folder of its own ("Artist - Title (1)", ...). A folder that
    holds anything but the files of this download stays, as does one with a
    chart (.txt) in it, which a sync client can bring in from another computer."""
    # The download and its audio and video apart
    own = (f"{basename}.", f"{basename}_video.")
    try:
        names = os.listdir(folder)
        if any(not name.startswith(own) or name.lower().endswith(".txt")
               or not os.path.isfile(os.path.join(folder, name)) for name in names):
            print(f"{ULTRASINGER_HEAD} The folder of the failed download holds other files and stays: {folder}")
            return
        shutil.rmtree(folder)
        print(f"{ULTRASINGER_HEAD} Removed the folder of the failed download: {folder}")
    except OSError as error:
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} the folder of the failed download "
              f"could not be removed ({error}): {folder}")


def download_from_video_platform(input_url: str, output_folder_path: str, cookiefile: str = None, *,
                                  keep_audio_in_video: bool = False,
                                  po_token: str = None,
                                  limits: DownloadLimits = DownloadLimits()) -> tuple[str, str, str, MediaInfo]:
    """Download from the video platform"""
    info = get_video_info(input_url, cookiefile, po_token)
    (artist, title, video_title, yt_language) = video_title_from_info(info)

    # Get additional data for song, with the length of the video (see search_musicbrainz)
    song_info = find_song(artist, title, video_title, duration=info.get("duration") or None)

    basename_without_ext = sanitize_filename(f"{song_info.artist} - {song_info.title}")
    song_output = os.path.join(output_folder_path, basename_without_ext)
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)

    try:
        print(f"{ULTRASINGER_HEAD} Downloading from the video platform")
        video_with_audio_path = __download_video_with_audio(
            input_url, basename_without_ext, song_output, cookiefile, po_token, limits
        )

        # Separate audio and video
        audio_file_path, final_video_path, audio_ext, video_ext = separate_audio_video(
            video_with_audio_path, basename_without_ext, song_output,
            keep_audio_in_video=keep_audio_in_video,
        )
    except BaseException as error:
        if isinstance(error, yt_dlp.utils.YoutubeDLError):
            # Its frames in yt-dlp keep the partial file of a fragmented format
            # open, which then could not be deleted: it goes on without them
            error.__traceback__ = None
            gc.collect()
        _remove_failed_song_folder(song_output, basename_without_ext)
        raise error

    cover_url = None
    if song_info.cover_url is not None and song_info.cover_image_data is not None:
        try:
            save_image(song_info.cover_image_data, basename_without_ext, song_output)
            cover_url = song_info.cover_url
        except Exception as e:  # noqa: BLE001 - the cover is optional: the video's thumbnail stands in
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} the cover from the song database could "
                  f"not be saved ({e}) - the video's thumbnail is taken instead")
    if cover_url is None:
        cover_url = __download_thumbnail(
            input_url, basename_without_ext, song_output, cookiefile, po_token
        )

    return (
        basename_without_ext,
        song_output,
        audio_file_path,
        MediaInfo(
            artist=song_info.artist,
            title=song_info.title,
            year=song_info.year,
            genre=song_info.genres,
            language=yt_language or None,
            cover_url=cover_url,
            video_url=input_url,
            audio_extension=audio_ext,
            video_extension=video_ext
        ),
    )
