"""Ultrastar txt parser"""
import os
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


def parse(input_file: str) -> UltrastarTxtValue:
    """Parse ultrastar txt file to UltrastarTxt class"""
    print(f"{ULTRASINGER_HEAD} Parse ultrastar txt -> {input_file}")

    with open(input_file, "r", encoding=FILE_ENCODING) as file:
        txt = file.readlines()

    ultrastar_class = UltrastarTxtValue()
    count = 0

    # Strips the newline character
    for line in txt:
        count += 1
        if line.startswith("#"):
            if line.startswith(f"#{UltrastarTxtTag.ARTIST.value}"):
                ultrastar_class.artist = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.TITLE.value}"):
                ultrastar_class.title = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.MP3.value}"):
                ultrastar_class.mp3 = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.AUDIO.value}"):
                ultrastar_class.audio = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.VIDEO.value}"):
                ultrastar_class.video = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.GAP.value}"):
                ultrastar_class.gap = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.BPM.value}"):
                ultrastar_class.bpm = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.VIDEOGAP.value}"):
                ultrastar_class.videoGap = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.COVERURL.value}:"):
                ultrastar_class.coverUrl = line.split(":", 1)[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.COVER.value}:"):
                ultrastar_class.cover = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.BACKGROUNDURL.value}:"):
                ultrastar_class.backgroundUrl = line.split(":", 1)[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.BACKGROUND.value}:"):
                ultrastar_class.background = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.VIDEOURL.value}:"):
                ultrastar_class.videoUrl = line.split(":", 1)[1].replace("\n", "")
        elif line.startswith(
            (
                f"{UltrastarTxtNoteTypeTag.FREESTYLE.value} ",
                f"{UltrastarTxtNoteTypeTag.NORMAL.value} ",
                f"{UltrastarTxtNoteTypeTag.GOLDEN.value} ",
                f"{UltrastarTxtNoteTypeTag.RAP.value} ",
                f"{UltrastarTxtNoteTypeTag.RAP_GOLDEN.value} ",
            )
        ):
            parts = line.split()
            # [0] F : * R G
            # [1] start beat
            # [2] duration
            # [3] pitch
            # [4] word

            ultrastar_note_line = UltrastarNoteLine(noteType=get_note_type_from_string(parts[0]),
                              startBeat=float(parts[1]),
                              duration=float(parts[2]),
                              pitch=int(parts[3]),
                              word=parts[4] if len(parts) > 4 else "",
                            startTime=get_start_time(ultrastar_class.gap, ultrastar_class.bpm, float(parts[1])),
                            endTime=get_end_time(ultrastar_class.gap, ultrastar_class.bpm, float(parts[1]), float(parts[2])))

            ultrastar_class.UltrastarNoteLines.append(ultrastar_note_line)

            # todo: Progress?

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

    song_output = os.path.join(
        output_folder_path,
        ultrastar_class.artist.strip() + " - " + ultrastar_class.title.strip(),
    )

    # todo: get_unused_song_output_dir should be in the runner
    song_output = get_unused_song_output_dir(str(song_output))
    os_helper.create_folder(song_output)

    dirname = os.path.dirname(input_file_path)
    audio_file_path = os.path.join(dirname, ultrastar_mp3_name)
    basename_without_ext = f"{ultrastar_class.artist.strip()} - {ultrastar_class.title.strip()}"

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
