"""Ultrastar txt parser"""
import codecs
import math
import os
import re
import shutil

from modules import os_helper

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted, blue_highlighted
from modules.Ultrastar.coverter.ultrastar_converter import (
    get_end_time,
    get_start_time,
)
from modules.Ultrastar.ultrastar_txt import (
    UltrastarTxtValue,
    UltrastarTxtTag,
    UltrastarTxtNoteTypeTag,
    FILE_ENCODING,
    UltrastarNoteLine,
    get_note_type_from_string
)
from modules.os_helper import get_unused_song_output_dir

# A note line: its type (F : * R G), then a space or a tab
_NOTE_LINE_RE = re.compile("[" + re.escape("".join(t.value for t in UltrastarTxtNoteTypeTag)) + "][ \t]")


def is_note_line(line: str) -> bool:
    """Whether the line of a chart is a note: ``": 12 4 5 word"`` (also with a tab after the type)."""
    return _NOTE_LINE_RE.match(line) is not None

_ENCODING_TAG_RE = re.compile(rb"^#ENCODING:([^\r\n]*)", re.IGNORECASE | re.MULTILINE)


def chart_encoding(input_file: str) -> str:
    """The encoding to read an UltraStar txt file with.

    UTF-8 when the file starts with a byte order mark (which is then left out
    when reading), or when it is valid UTF-8 with non-ASCII characters (also
    under a stale #ENCODING tag, as an editor that saves in UTF-8 leaves it;
    text in a single-byte encoding is practically never valid UTF-8).
    Otherwise the encoding its #ENCODING tag names (older charts) if the file
    decodes with it, else UTF-8 or, when that fails, CP1252, the encoding of
    many older hand-made charts.
    """
    with open(input_file, "rb") as file:
        data = file.read()
    if data.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    try:
        data.decode(FILE_ENCODING)
        utf8 = True
    except UnicodeDecodeError:
        utf8 = False
    if utf8 and not data.isascii():
        return FILE_ENCODING
    tag = _ENCODING_TAG_RE.search(data)
    declared = _codec_of_encoding_tag(tag.group(1).decode("ascii", "replace")) if tag else None
    if declared:
        try:
            data.decode(declared)
            return declared
        except (UnicodeDecodeError, LookupError):
            pass
    return FILE_ENCODING if utf8 else "cp1252"


def _codec_of_encoding_tag(value: str) -> str | None:
    """Python codec of an #ENCODING tag value ("UTF8", "CP1252", ...); None for
    "AUTO", "LOCALE", unknown names and codecs that are no text encoding or do
    not read the tag itself (ASCII) as ASCII."""
    value = value.strip()
    if value.upper() in ("", "AUTO", "LOCALE"):
        return None
    try:
        codec = codecs.lookup(value).name
        return codec if b"#ENCODING:".decode(codec) == "#ENCODING:" else None
    except (LookupError, UnicodeDecodeError):
        return None


def read_chart_lines(input_file: str) -> list[str]:
    """The lines of an UltraStar txt file (decoded as ``chart_encoding`` says)."""
    with open(input_file, "r", encoding=chart_encoding(input_file), errors="replace") as file:
        return file.readlines()


def header_tag(line: str) -> str | None:
    """The tag of a header line in upper case (``"#Cover:x.jpg"`` -> ``"COVER"``), else None."""
    if not line.startswith("#"):
        return None
    tag, separator, _ = line[1:].partition(":")
    return tag.strip().upper() if separator else None


def is_relative_tag(line: str) -> bool:
    """Whether the line is ``#RELATIVE:YES``: the beats of each line count from the line's own start."""
    tag, separator, value = line.partition(":")
    return (bool(separator) and tag.strip().upper() == f"#{UltrastarTxtTag.RELATIVE.value}"
            and value.strip().upper() == "YES")


def relative_line_break(line: str) -> tuple[float, float] | None:
    """The numbers of a line break ``- <break beat> <shift>`` of a #RELATIVE:YES
    chart, both counted from the start of the current line: the break, and how
    far the start of the next line moves on (a break with one number moves the
    start there). None when the first value is not a finite number; a second
    value that is not a finite number counts as missing."""
    numbers = []
    for part in line[1:].split()[:2]:
        try:
            number = float(part)
        except ValueError:
            break
        if not math.isfinite(number):
            break
        numbers.append(number)
    if not numbers:
        return None
    return numbers[0], numbers[-1]


def parse(input_file: str) -> UltrastarTxtValue:
    """Parse ultrastar txt file to UltrastarTxt class"""
    print(f"{ULTRASINGER_HEAD} Parse ultrastar txt -> {input_file}")

    txt = read_chart_lines(input_file)

    ultrastar_class = UltrastarTxtValue()
    # #RELATIVE:YES - the beats of a line count from the line's own start, which
    # each line break "- <break beat> <shift>" moves on by <shift> beats
    relative = False
    line_start_beat = 0.0

    for line in txt:
        if line.startswith("#"):
            tag, separator, value = line[1:].partition(":")
            if not separator:
                continue
            tag = tag.strip().upper()
            value = value.replace("\n", "")
            if tag == UltrastarTxtTag.ARTIST.value:
                ultrastar_class.artist = value
            elif tag == UltrastarTxtTag.TITLE.value:
                ultrastar_class.title = value
            elif tag == UltrastarTxtTag.MP3.value:
                ultrastar_class.mp3 = value
            elif tag == UltrastarTxtTag.AUDIO.value:
                ultrastar_class.audio = value
            elif tag == UltrastarTxtTag.VIDEO.value:
                ultrastar_class.video = value
            elif tag == UltrastarTxtTag.GAP.value:
                ultrastar_class.gap = value
            elif tag == UltrastarTxtTag.BPM.value:
                ultrastar_class.bpm = value
            elif tag == UltrastarTxtTag.VIDEOGAP.value:
                ultrastar_class.videoGap = value
            elif tag == UltrastarTxtTag.COVERURL.value:
                ultrastar_class.coverUrl = value
            elif tag == UltrastarTxtTag.COVER.value:
                ultrastar_class.cover = value
            elif tag == UltrastarTxtTag.BACKGROUNDURL.value:
                ultrastar_class.backgroundUrl = value
            elif tag == UltrastarTxtTag.BACKGROUND.value:
                ultrastar_class.background = value
            elif tag == UltrastarTxtTag.VIDEOURL.value:
                ultrastar_class.videoUrl = value
            elif tag == UltrastarTxtTag.RELATIVE.value:
                relative = is_relative_tag(line)
        elif is_note_line(line):
            parts = line.split()
            # [0] F : * R G
            # [1] start beat
            # [2] duration
            # [3] pitch
            # [4] word
            start_beat = float(parts[1]) + line_start_beat

            ultrastar_note_line = UltrastarNoteLine(noteType=get_note_type_from_string(parts[0]),
                              startBeat=start_beat,
                              duration=float(parts[2]),
                              pitch=int(parts[3]),
                              word=parts[4] if len(parts) > 4 else "",
                            startTime=get_start_time(ultrastar_class.gap, ultrastar_class.bpm, start_beat),
                            endTime=get_end_time(ultrastar_class.gap, ultrastar_class.bpm, start_beat, float(parts[2])))

            ultrastar_class.UltrastarNoteLines.append(ultrastar_note_line)

            # todo: Progress?
        elif relative and line.startswith(UltrastarTxtTag.LINEBREAK.value):
            numbers = relative_line_break(line)
            if numbers:
                line_start_beat += numbers[1]

    return ultrastar_class


def parse_ultrastar_txt(input_file_path: str, output_folder_path: str) -> tuple[str, str, str, UltrastarTxtValue, str]:
    """Parse Ultrastar txt"""
    ultrastar_class = parse(input_file_path)

    if ultrastar_class.mp3:
        ultrastar_mp3_name = ultrastar_class.mp3
    elif ultrastar_class.audio:
        ultrastar_mp3_name = ultrastar_class.audio
    else:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error!')} The provided text file does not have a reference to "
            f"an audio file."
        )
        exit(1)
    _, audio_ext_with_dot = os.path.splitext(ultrastar_mp3_name)
    audio_ext = audio_ext_with_dot.lstrip('.')

    # Chart titles may contain characters that are invalid in paths (e.g. "?" or "/")
    basename_without_ext = os_helper.sanitize_filename(
        f"{ultrastar_class.artist.strip()} - {ultrastar_class.title.strip()}"
    )
    song_output = os.path.join(output_folder_path, basename_without_ext)

    # todo: get_unused_song_output_dir should be in the runner
    song_output = get_unused_song_output_dir(str(song_output))
    os_helper.create_folder(song_output)

    dirname = os.path.dirname(input_file_path)
    audio_file_path = os.path.join(dirname, ultrastar_mp3_name)

    # Copy audio file to output folder
    audio_output_path = _copy_audio_file(audio_file_path, ultrastar_mp3_name, song_output)

    # Copy referenced asset files (cover, background, video) to output folder
    _copy_txt_assets(ultrastar_class, dirname, song_output)

    return (
        basename_without_ext,
        song_output,
        audio_output_path,
        ultrastar_class,
        audio_ext
    )


def _copy_audio_file(
        audio_file_path: str,
        audio_filename: str,
        output_dir: str,
) -> str:
    """Copy the audio file referenced in the TXT to the output directory.

    Returns the path to the copied audio file in the output directory,
    or the original path if the source file was not found.
    """
    dst = os.path.join(output_dir, audio_filename)
    if os.path.isfile(audio_file_path):
        shutil.copy2(audio_file_path, dst)
        print(f"{ULTRASINGER_HEAD} Audio copied: {blue_highlighted(audio_filename)}")
        return dst
    else:
        print(f"{ULTRASINGER_HEAD} Audio not found: {red_highlighted(audio_file_path)}")
        return audio_file_path


def _copy_txt_assets(
        ultrastar_class: UltrastarTxtValue,
        source_dir: str,
        output_dir: str,
) -> None:
    """Copy cover, background, and video files from source to output directory.

    If a cover file is not found locally, attempts to fetch one from
    MusicBrainz using the artist/title metadata.
    """
    assets = [
        ("Cover", ultrastar_class.cover),
        ("Background", ultrastar_class.background),
        ("Video", ultrastar_class.video),
    ]

    for label, filename in assets:
        if not filename:
            continue
        src = os.path.join(source_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.isfile(src):
            # Ensure subdirectories exist (in case filename contains path separators)
            os.makedirs(os.path.dirname(dst) or output_dir, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"{ULTRASINGER_HEAD} {label} copied: {blue_highlighted(filename)}")
        else:
            print(f"{ULTRASINGER_HEAD} {label} not found: {red_highlighted(src)}")

    # If no cover was copied, try fetching from MusicBrainz
    cover_exists = ultrastar_class.cover and os.path.isfile(
        os.path.join(output_dir, ultrastar_class.cover)
    )
    if not cover_exists and ultrastar_class.artist and ultrastar_class.title:
        _fetch_musicbrainz_cover(ultrastar_class, output_dir)


def _fetch_musicbrainz_cover(
        ultrastar_class: UltrastarTxtValue,
        output_dir: str,
) -> None:
    """Try to download a cover image from MusicBrainz."""
    try:
        from modules.musicbrainz_client import search_musicbrainz
        from modules.Image.image_helper import save_image

        print(f"{ULTRASINGER_HEAD} No cover found — searching MusicBrainz...")
        song_info = search_musicbrainz(
            ultrastar_class.title.strip(),
            ultrastar_class.artist.strip(),
        )
        if song_info.cover_image_data:
            # A name that is valid as a file name, like the song folder's
            basename = os_helper.sanitize_filename(
                f"{ultrastar_class.artist.strip()} - {ultrastar_class.title.strip()}"
            )
            save_image(song_info.cover_image_data, basename, output_dir)
            cover_filename = basename + " [CO].jpg"
            ultrastar_class.cover = cover_filename
            if song_info.cover_url:
                ultrastar_class.coverUrl = song_info.cover_url
            print(
                f"{ULTRASINGER_HEAD} Cover downloaded from MusicBrainz: "
                f"{blue_highlighted(cover_filename)}"
            )
        else:
            print(f"{ULTRASINGER_HEAD} {red_highlighted('No cover found on MusicBrainz')}")
    except Exception as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted(f'MusicBrainz cover fetch failed: {e}')}")
