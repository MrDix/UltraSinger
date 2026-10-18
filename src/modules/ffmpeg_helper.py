"""FFmpeg helper module."""

import os
import shutil
import subprocess
import time


def __add_ffmpeg_to_path(ffmpeg_path) -> None:
    os.environ["PATH"] = os.pathsep.join([os.environ["PATH"], ffmpeg_path])


def get_ffmpeg_and_ffprobe_paths():
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not ffmpeg_path:
        raise FileNotFoundError("FFmpeg executable not found in the system path.")
    if not ffprobe_path:
        raise FileNotFoundError("FFprobe executable not found in the system path.")

    return ffmpeg_path, ffprobe_path


def is_ffmpeg_available(user_path: str = "") -> bool:
    if shutil.which("ffmpeg"):
        return True
    if (path := user_path) and path not in os.environ["PATH"]:
        __add_ffmpeg_to_path(path)
        if shutil.which("ffmpeg"):
            return True
    return False


def extract_audio(input_video_path: str, output_audio_path: str) -> None:
    """Extract audio from video file in original format without re-encoding"""
    ffmpeg_path, _ = get_ffmpeg_and_ffprobe_paths()

    cmd = [
        ffmpeg_path,
        "-i",
        input_video_path,
        # The first audio stream, whose codec get_audio_codec_and_extension names:
        # ffmpeg itself takes the default one, else the one with the most channels
        "-map",
        "0:a:0",
        "-vn",  # No video
        "-acodec",
        "copy",  # Copy audio codec without re-encoding
        "-fflags",
        "+bitexact",  # Same bytes every time (the cache keys of the steps after it follow the bytes)
        "-y",  # Overwrite output file if exists
        output_audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg audio extraction failed: {result.stderr}")


def remove_audio_from_video(input_video_path: str, output_video_path: str) -> None:
    """Remove audio from video file without re-encoding video"""
    ffmpeg_path, _ = get_ffmpeg_and_ffprobe_paths()

    cmd = [
        ffmpeg_path,
        "-i",
        input_video_path,
        "-an",  # No audio
        "-vcodec",
        "copy",  # Copy video codec without re-encoding
        "-y",  # Overwrite output file if exists
        output_video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg audio removal failed: {result.stderr}")


def is_video_file(file_path: str) -> bool:
    """Check if file contains a real video stream using ffprobe.

    Cover art embedded in audio files (e.g. an MP3 with an APIC frame) is
    reported by ffprobe as a video stream with the ``attached_pic``
    disposition. Such files are audio, not video, so those streams are
    ignored.
    """
    try:
        _, ffprobe_path = get_ffmpeg_and_ffprobe_paths()

        cmd = [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "v",
            "-show_entries", "stream=codec_type:stream_disposition=attached_pic",
            "-of", "csv=p=0",
            file_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False
        return _has_real_video_stream(result.stdout)
    except Exception:
        return False


def _has_real_video_stream(ffprobe_csv: str) -> bool:
    """Parse ``codec_type,attached_pic`` lines; True if any video stream is not cover art."""
    for line in ffprobe_csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or parts[0] != "video":
            continue
        attached_pic = parts[1] if len(parts) > 1 else "0"
        if attached_pic != "1":
            return True
    return False


def get_audio_codec_and_extension(video_file_path: str) -> str:
    """
    Detect audio codec from video file and return codec name and appropriate file extension.
    """
    try:
        _, ffprobe_path = get_ffmpeg_and_ffprobe_paths()

        cmd = [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_file_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            # Default to wav if detection fails
            return "wav"

        codec_name = result.stdout.strip()

        codec_to_extension = {
            "aac": "aac",
            "mp3": "mp3",
            "opus": "opus",
            "vorbis": "ogg",
            "flac": "flac",
            "pcm_s16le": "wav",
            "pcm_s24le": "wav",
            "pcm_s32le": "wav",
            "ac3": "ac3",
            "eac3": "eac3",
        }

        extension = codec_to_extension.get(codec_name, "wav")
        return extension

    except Exception:
        # Default to wav if detection fails
        return "wav"


def get_duration(file_path: str) -> float | None:
    """The length of an audio or video file in seconds, as ffprobe reads it
    from the file, or None when it cannot be read."""
    try:
        _, ffprobe_path = get_ffmpeg_and_ffprobe_paths()
        result = subprocess.run(
            [ffprobe_path, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True)
        duration = float(result.stdout.strip())
    except (OSError, ValueError):
        # No ffprobe, or no length: "N/A", or nothing for a file it cannot read
        return None
    return duration if duration > 0 else None


def convert_to_ultrastar_format(audio_file_path: str, basename_without_ext: str, output_folder: str, audio_ext: str) -> tuple[str, str]:
    """
    Convert audio to an UltraStar-compatible format (ogg) if needed.
    UltraStar Play only supports mp3, ogg, and wav.
    Returns the (possibly updated) audio file path and extension.
    """
    ULTRASTAR_COMPATIBLE_FORMATS = {"mp3", "ogg", "wav"}
    audio_ext = audio_ext.lstrip(".").lower()

    if audio_ext in ULTRASTAR_COMPATIBLE_FORMATS:
        return audio_file_path, audio_ext

    from modules.console_colors import ULTRASINGER_HEAD

    target_ext = "ogg"
    target_path = os.path.join(output_folder, f"{basename_without_ext}.{target_ext}")
    ffmpeg_path, _ = get_ffmpeg_and_ffprobe_paths()

    print(f"{ULTRASINGER_HEAD} Converting {audio_ext} to {target_ext} (UltraStar compatibility)")

    cmd = [
        ffmpeg_path,
        "-i", audio_file_path,
        "-y",
        "-loglevel", "error",
        "-codec:a", "libvorbis",
        "-q:a", "6",  # High quality VBR (~192 kbps)
        "-fflags", "+bitexact",  # Same bytes every time, as in extract_audio
        target_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg audio conversion to {target_ext} failed: {result.stderr}")

    # Remove the original incompatible audio file
    os.remove(audio_file_path)

    return target_path, target_ext


# A sync client or virus scanner may hold a file that was just written for a
# moment; renaming or removing it then fails with "file in use" on Windows.
_IN_USE_RETRIES = 10
_IN_USE_WAIT_S = 0.5


def _when_not_in_use(action, *args):
    """Run a file operation, retried for a few seconds while another process
    holds the file."""
    for attempt in range(_IN_USE_RETRIES):
        try:
            return action(*args)
        except PermissionError:
            if attempt == _IN_USE_RETRIES - 1:
                raise
            time.sleep(_IN_USE_WAIT_S)


def _same_path(first: str, second: str) -> bool:
    """Whether two paths name the same file, however they are written
    (separators, letter case on Windows, relative parts)."""
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def separate_audio_video(video_with_audio_path: str, basename_without_ext: str, output_folder: str, *, keep_audio_in_video: bool = False) -> tuple[str, str, str, str]:
    """
    Separate audio and video from a video file.
    Automatically detects the audio codec and uses the appropriate file extension.
    Converts to ogg if the audio format is not supported by UltraStar Play.

    If *keep_audio_in_video* is True the output video retains its original
    audio track (the full mix).  This makes it possible to re-import the
    video file into UltraSinger later without downloading it again.
    """
    from modules.console_colors import ULTRASINGER_HEAD

    # Get original video file extension without the dot
    _, video_ext_with_dot = os.path.splitext(video_with_audio_path)
    video_ext = video_ext_with_dot.lstrip('.')

    # Detect audio codec and get appropriate extension
    audio_ext = get_audio_codec_and_extension(video_with_audio_path)

    print(f"{ULTRASINGER_HEAD} Extracting audio from video")
    audio_file_path = os.path.join(output_folder, f"{basename_without_ext}.{audio_ext}")
    extract_audio(video_with_audio_path, audio_file_path)

    # Convert to UltraStar-compatible format if needed
    audio_file_path, audio_ext = convert_to_ultrastar_format(
        audio_file_path, basename_without_ext, output_folder, audio_ext
    )

    final_video_path = os.path.join(output_folder, f"{basename_without_ext}.{video_ext}")

    if keep_audio_in_video:
        print(f"{ULTRASINGER_HEAD} Keeping audio in video file")
        # Just rename the original to the final name (it already has audio).
        # The downloaded path and the output folder may write the same file
        # differently ("D:\x" vs "D:/x"): renaming a file onto itself fails
        # while a sync client or virus scanner holds it.
        if not _same_path(video_with_audio_path, final_video_path):
            _when_not_in_use(os.rename, video_with_audio_path, final_video_path)
    else:
        print(f"{ULTRASINGER_HEAD} Creating video without audio")
        video_only_path = os.path.join(output_folder, f"{basename_without_ext}_video.{video_ext}")
        remove_audio_from_video(video_with_audio_path, video_only_path)

        # Remove original video with audio
        _when_not_in_use(os.remove, video_with_audio_path)

        # Rename video without audio to final name
        _when_not_in_use(os.rename, video_only_path, final_video_path)

    return audio_file_path, final_video_path, audio_ext, video_ext

