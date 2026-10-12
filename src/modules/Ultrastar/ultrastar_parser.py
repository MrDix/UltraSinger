"""Ultrastar txt parser"""
import codecs
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

NOTE_LINE_PREFIXES = (
    f"{UltrastarTxtNoteTypeTag.FREESTYLE.value} ",
    f"{UltrastarTxtNoteTypeTag.NORMAL.value} ",
    f"{UltrastarTxtNoteTypeTag.GOLDEN.value} ",
    f"{UltrastarTxtNoteTypeTag.RAP.value} ",
    f"{UltrastarTxtNoteTypeTag.RAP_GOLDEN.value} ",
)

_ENCODING_TAG_RE = re.compile(rb"^#ENCODING:([^\r\n]*)", re.IGNORECASE | re.MULTILINE)


def chart_encoding(input_file: str) -> str:
    """The encoding to read an UltraStar txt file with.

    UTF-8 when the file starts with a byte order mark (which is then left out
    when reading). Otherwise the encoding its #ENCODING tag names (older
    charts) or UTF-8, as long as the file decodes with it, else CP1252, the
    encoding of many older hand-made charts.
    """
    with open(input_file, "rb") as file:
        data = file.read()
    if data.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    tag = _ENCODING_TAG_RE.search(data)
    declared = _codec_of_encoding_tag(tag.group(1).decode("ascii", "replace")) if tag else None
    for encoding in (declared, FILE_ENCODING):
        if encoding is None:
            continue
        try:
            data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            pass
    return "cp1252"


def _codec_of_encoding_tag(value: str) -> str | None:
    """Python codec of an #ENCODING tag value ("UTF8", "CP1252", ...); None for
    "AUTO", "LOCALE" and unknown names."""
    value = value.strip()
    if value.upper() in ("", "AUTO", "LOCALE"):
        return None
    try:
        return codecs.lookup(value).name
    except LookupError:
        return None


def read_chart_lines(input_file: str) -> list[str]:
    """The lines of an UltraStar txt file (decoded as ``chart_encoding`` says)."""
    with open(input_file, "r", encoding=chart_encoding(input_file), errors="replace") as file:
        return file.readlines()


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
                relative = value.strip().upper() == "YES"
        elif line.startswith(NOTE_LINE_PREFIXES):
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
            parts = line[1:].split()
            if parts:
                # The shift is the second number; a break with only one moves the start there
                line_start_beat += float(parts[1] if len(parts) > 1 else parts[0])

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
            basename = f"{ultrastar_class.artist.strip()} - {ultrastar_class.title.strip()}"
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
