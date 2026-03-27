"""Write metadata tags to audio output files (MP3, OGG, WAV, M4A).

Uses mutagen for cross-format tag writing. Non-critical — failures
are logged but never abort the pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

from modules.console_colors import ULTRASINGER_HEAD

logger = logging.getLogger(__name__)


def write_metadata_to_audio(
    audio_path: str,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    album: Optional[str] = None,
    year: Optional[str] = None,
    genre: Optional[str] = None,
    cover_image_path: Optional[str] = None,
) -> bool:
    """Write metadata tags to an audio file.

    Supports MP3 (ID3v2.4), OGG Vorbis, FLAC, M4A/AAC, and WAV.

    Returns True if tags were written successfully, False otherwise.
    """
    try:
        import mutagen

    except ImportError:
        print(f"{ULTRASINGER_HEAD} mutagen not installed, skipping metadata tags")
        return False

    if not Path(audio_path).exists():
        return False

    ext = Path(audio_path).suffix.lower()

    try:
        if ext == ".mp3":
            return _write_mp3_tags(audio_path, title, artist, album, year, genre, cover_image_path)
        elif ext == ".ogg":
            return _write_ogg_tags(audio_path, title, artist, album, year, genre, cover_image_path)
        elif ext == ".flac":
            return _write_flac_tags(audio_path, title, artist, album, year, genre, cover_image_path)
        elif ext in (".m4a", ".mp4", ".aac"):
            return _write_m4a_tags(audio_path, title, artist, album, year, genre, cover_image_path)
        elif ext == ".wav":
            return _write_wav_tags(audio_path, title, artist, album, year, genre)
        else:
            logger.debug("Unsupported format for metadata: %s", ext)
            return False
    except Exception as e:
        logger.warning("Failed to write metadata to %s: %s", audio_path, e)
        return False


def _write_mp3_tags(
    path: str, title, artist, album, year, genre, cover_path
) -> bool:
    """Write ID3v2.4 tags to MP3."""
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TCON, APIC, ID3NoHeaderError

    try:
        audio = MP3(path, ID3=ID3)
    except ID3NoHeaderError:
        audio = MP3(path)
        audio.add_tags()

    if audio.tags is None:
        audio.add_tags()

    if title:
        audio.tags.add(TIT2(encoding=3, text=[title]))
    if artist:
        audio.tags.add(TPE1(encoding=3, text=[artist]))
    if album:
        audio.tags.add(TALB(encoding=3, text=[album]))
    if year:
        audio.tags.add(TDRC(encoding=3, text=[str(year)]))
    if genre:
        audio.tags.add(TCON(encoding=3, text=[genre]))

    if cover_path and Path(cover_path).exists():
        with open(cover_path, "rb") as f:
            cover_data = f.read()
        mime = "image/jpeg" if cover_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        audio.tags.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=cover_data))

    audio.save()
    return True


def _write_ogg_tags(
    path: str, title, artist, album, year, genre, cover_path
) -> bool:
    """Write Vorbis comments to OGG."""
    from mutagen.oggvorbis import OggVorbis

    audio = OggVorbis(path)
    if title:
        audio["TITLE"] = [title]
    if artist:
        audio["ARTIST"] = [artist]
    if album:
        audio["ALBUM"] = [album]
    if year:
        audio["DATE"] = [str(year)]
    if genre:
        audio["GENRE"] = [genre]

    if cover_path and Path(cover_path).exists():
        _embed_cover_in_vorbis(audio, cover_path)

    audio.save()
    return True


def _write_flac_tags(
    path: str, title, artist, album, year, genre, cover_path
) -> bool:
    """Write Vorbis comments + picture to FLAC."""
    from mutagen.flac import FLAC, Picture

    audio = FLAC(path)
    if title:
        audio["TITLE"] = [title]
    if artist:
        audio["ARTIST"] = [artist]
    if album:
        audio["ALBUM"] = [album]
    if year:
        audio["DATE"] = [str(year)]
    if genre:
        audio["GENRE"] = [genre]

    if cover_path and Path(cover_path).exists():
        pic = Picture()
        pic.type = 3  # Front cover
        pic.mime = "image/jpeg" if cover_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        with open(cover_path, "rb") as f:
            pic.data = f.read()
        audio.clear_pictures()
        audio.add_picture(pic)

    audio.save()
    return True


def _write_m4a_tags(
    path: str, title, artist, album, year, genre, cover_path
) -> bool:
    """Write MP4 atoms to M4A/AAC."""
    from mutagen.mp4 import MP4, MP4Cover

    audio = MP4(path)
    if audio.tags is None:
        audio.add_tags()

    if title:
        audio.tags["\xa9nam"] = [title]
    if artist:
        audio.tags["\xa9ART"] = [artist]
    if album:
        audio.tags["\xa9alb"] = [album]
    if year:
        audio.tags["\xa9day"] = [str(year)]
    if genre:
        audio.tags["\xa9gen"] = [genre]

    if cover_path and Path(cover_path).exists():
        with open(cover_path, "rb") as f:
            cover_data = f.read()
        fmt = MP4Cover.FORMAT_JPEG if cover_path.lower().endswith((".jpg", ".jpeg")) else MP4Cover.FORMAT_PNG
        audio.tags["covr"] = [MP4Cover(cover_data, imageformat=fmt)]

    audio.save()
    return True


def _write_wav_tags(
    path: str, title, artist, album, year, genre
) -> bool:
    """Write ID3 tags to WAV (limited player support)."""
    from mutagen.wave import WAVE
    from mutagen.id3 import TIT2, TPE1, TALB, TDRC, TCON

    audio = WAVE(path)
    if audio.tags is None:
        audio.add_tags()

    if title:
        audio.tags.add(TIT2(encoding=3, text=[title]))
    if artist:
        audio.tags.add(TPE1(encoding=3, text=[artist]))
    if album:
        audio.tags.add(TALB(encoding=3, text=[album]))
    if year:
        audio.tags.add(TDRC(encoding=3, text=[str(year)]))
    if genre:
        audio.tags.add(TCON(encoding=3, text=[genre]))

    audio.save()
    return True


def _embed_cover_in_vorbis(audio, cover_path: str) -> None:
    """Embed cover art in OGG Vorbis using METADATA_BLOCK_PICTURE.

    Large cover images are resized to fit within the Vorbis comment header
    size limits of common players.  Karedi's JOrbis reader has a 64 KB
    mark limit for header parsing — a ~48 KB JPEG (≈500×500) stays safely
    below that after base64 encoding (~64 KB) plus other header overhead.
    """
    import base64
    from mutagen.flac import Picture

    cover_data = _load_and_resize_cover(cover_path, max_size=500, max_bytes=48_000)
    if cover_data is None:
        return

    pic = Picture()
    pic.type = 3  # Front cover
    pic.mime = "image/jpeg"
    pic.data = cover_data

    # Encode as base64 FLAC picture block (standard for Vorbis comments)
    audio["METADATA_BLOCK_PICTURE"] = [base64.b64encode(pic.write()).decode("ascii")]


# Maximum pixel dimension and file size for OGG-embedded covers.
# Keeps the base64 METADATA_BLOCK_PICTURE under ~64 KB so that
# Karedi (JOrbis 64 001-byte mark limit) can parse the headers.
_COVER_MAX_DIMENSION = 500
_COVER_MAX_BYTES = 48_000


def _load_and_resize_cover(
    cover_path: str,
    max_size: int = _COVER_MAX_DIMENSION,
    max_bytes: int = _COVER_MAX_BYTES,
) -> Optional[bytes]:
    """Load a cover image and resize/recompress if needed for OGG embedding.

    Returns JPEG bytes that fit within *max_bytes*, or None on failure.
    """
    import io

    try:
        from PIL import Image
    except ImportError:
        # Pillow not available — fall back to raw file, hope it's small enough
        with open(cover_path, "rb") as f:
            data = f.read()
        if len(data) <= max_bytes:
            return data
        logger.warning(
            "Cover image %d bytes exceeds %d byte limit but Pillow is not "
            "installed for resizing — skipping OGG cover embed",
            len(data), max_bytes,
        )
        return None

    try:
        img = Image.open(cover_path)
        img = img.convert("RGB")  # Ensure JPEG-compatible mode

        # Resize if larger than max_size
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)

        # Compress at decreasing quality until within budget
        for quality in (85, 75, 60, 45):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                return data

        # Last resort: even smaller
        img.thumbnail((300, 300), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=45, optimize=True)
        return buf.getvalue()

    except Exception as e:
        logger.warning("Failed to resize cover for OGG embedding: %s", e)
        # Try raw file as fallback
        try:
            with open(cover_path, "rb") as f:
                data = f.read()
            if len(data) <= max_bytes:
                return data
        except OSError:
            pass
        return None
