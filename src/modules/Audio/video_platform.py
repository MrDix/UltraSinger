"""Video Platform Downloader"""

import logging
import os
import re

import yt_dlp

from modules.os_helper import sanitize_filename, get_unused_song_output_dir
from modules import os_helper
from modules.ProcessData import MediaInfo
from modules.Audio.bpm import get_bpm_from_file
from modules.console_colors import ULTRASINGER_HEAD, gold_highlighted, red_highlighted
from modules.Image.image_helper import save_image
from modules.musicbrainz_client import search_musicbrainz
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


def strip_unmatched_suffixes(track: str, video_title: str) -> str:
    """Remove trailing parenthetical suffixes from the music service track name
    that do not appear in the actual video title.

    The music service metadata sometimes adds qualifiers like '(live)' or
    '(acoustic)' that do not match the uploaded video.  By cross-checking
    against the video title we can strip these misleading suffixes while
    keeping legitimate ones like '(feat. …)'.

    Only *trailing* parenthesised groups whose content matches a known
    set of media qualifiers are considered, so arbitrary metadata like
    '(feat. Guest)' or '(Part 2)' is never touched.
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

# Premium formats (higher-bitrate copies for subscribers) can come very slowly,
# with connections that stall again and again, even when the format picked in
# their place comes at a normal speed. So they are only taken when no other
# mp4 video is offered.
_VIDEO_FORMAT = "bestvideo[ext=mp4][format_note!*=?Premium]+bestaudio/bestvideo[ext=mp4]+bestaudio/best"


def get_video_title(url: str, cookiefile: str | None = None,
                     po_token: str | None = None) -> tuple[str, str, str, str]:
    """Get the title and language of the video.

    Returns:
        A 4-tuple of (artist, track, video_title, language).
        *video_title* is the raw title string from the video's page,
        useful for cross-checking metadata from other sources.
        *language* is the ISO 639-1 code from yt-dlp metadata (e.g. "en"),
        or empty string if unavailable.
    """

    ydl_opts = _apply_po_token({
        "cookiefile": cookiefile,
        "logger": _FilteredLogger(),
        # A watch URL that also names a playlist stands for its video only
        "noplaylist": True,
    }, po_token)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(
            url, download=False  # We just want to extract the info
        )

    video_title = (result.get("title") or "").strip()
    artist = (result.get("artist") or "").strip()
    track = (result.get("track") or "").strip()
    language = (result.get("language") or "").strip()
    if artist and track:
        return artist, strip_unmatched_suffixes(track, video_title), video_title, language
    channel = (result.get("channel") or "").strip()
    if "-" in video_title:
        parts = video_title.split("-", 1)
        return parts[0].strip(), parts[1].strip(), video_title, language
    # "ArtistVEVO" or "Artist - Topic" is no artist name the lookups would find
    return artist_from_channel(channel), video_title, video_title, language


def _output_template(output_path: str, clear_filename: str) -> str:
    """yt-dlp output template for the file ``clear_filename`` in ``output_path``.

    yt-dlp reads "%(...)" as a field of the video's data ("100%(Live)" raised a
    KeyError), so a "%" of the path is written as "%%".
    """
    return f"{output_path}/{clear_filename}".replace("%", "%%") + ".%(ext)s"


def __download_video_with_audio(url: str, clear_filename: str, output_path: str, cookiefile: str = None,
                                 po_token: str = None) -> str:
    """Download video with audio from the video platform and return the path of the file"""

    print(f"{ULTRASINGER_HEAD} Downloading Video with Audio")
    ydl_opts = _apply_po_token({
        "format": _VIDEO_FORMAT,
        "outtmpl": _output_template(output_path, clear_filename),
        # Only for formats that are merged: a single file keeps its own (e.g. webm)
        "merge_output_format": "mp4",
        "retries": _DOWNLOAD_RETRIES,
        "fragment_retries": _DOWNLOAD_RETRIES,
        "cookiefile": cookiefile,
        "logger": _FilteredLogger(),
        "noplaylist": True,  # as in get_video_title
    }, po_token)
    return __start_download(ydl_opts, url)


def __download_thumbnail(url: str, clear_filename: str, output_path: str, cookiefile: str = None,
                          po_token: str = None) -> str:
    """Download thumbnail from the video platform; its URL, or "" without one"""

    print(f"{ULTRASINGER_HEAD} Downloading thumbnail")
    ydl_opts = _apply_po_token({
        "skip_download": True,
        "writethumbnail": True,
        "cookiefile": cookiefile,
        "noplaylist": True,  # as in get_video_title
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
        # yt-dlp passes its screen messages here. Retries are shown, so that a
        # download that waits for a stalled connection does not look hung.
        if msg.startswith("[download] Got error"):
            print(f"WARNING: {msg}")
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


def __start_download(ydl_opts, url: str) -> str:
    """Start the download the ydl_opts; the path of the downloaded file"""
    ydl_opts.setdefault("logger", _FilteredLogger())
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    # The file as it is after merging, with the extension of its real container
    downloads = (info or {}).get("requested_downloads") or []
    path = downloads[-1].get("filepath") if downloads else None
    if not path or not os.path.isfile(path):
        raise Exception(f"Download failed: no downloaded file ({path})")
    return path


def download_from_video_platform(input_url: str, output_folder_path: str, cookiefile: str = None, *,
                                  keep_audio_in_video: bool = False,
                                  po_token: str = None) -> tuple[str, str, str, MediaInfo]:
    """Download from the video platform"""
    (artist, title, video_title, yt_language) = get_video_title(input_url, cookiefile, po_token)

    # Get additional data for song
    song_info = search_musicbrainz(title, artist)

    # MusicBrainz may re-introduce suffixes that were already stripped from
    # the yt-dlp metadata (e.g. "(live)" when the video is not actually live).
    # Cross-check the MusicBrainz title against the video title as well.
    song_info.title = strip_unmatched_suffixes(song_info.title, video_title)

    basename_without_ext = sanitize_filename(f"{song_info.artist} - {song_info.title}")
    song_output = os.path.join(output_folder_path, basename_without_ext)
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)

    print(f"{ULTRASINGER_HEAD} Downloading from the video platform")
    video_with_audio_path = __download_video_with_audio(
        input_url, basename_without_ext, song_output, cookiefile, po_token
    )

    # Separate audio and video
    audio_file_path, final_video_path, audio_ext, video_ext = separate_audio_video(
        video_with_audio_path, basename_without_ext, song_output,
        keep_audio_in_video=keep_audio_in_video,
    )

    if song_info.cover_url is not None and song_info.cover_image_data is not None:
        cover_url = song_info.cover_url
        save_image(song_info.cover_image_data, basename_without_ext, song_output)
    else:
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
