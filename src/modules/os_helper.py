"""OS helper functions."""

import hashlib
import os
import shutil

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted


def create_folder(folder_name: str) -> None:
    """Creates a folder if it doesn't exist."""
    print(f"{ULTRASINGER_HEAD} Creating output folder. -> {folder_name}")
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)


def move(original: str, target: str) -> None:
    """Moves a file from one location to another."""
    shutil.move(original, target)


def copy(original: str, target: str) -> None:
    """Copies a file from one location to another."""
    shutil.copy(original, target)


def rename(original: str, target: str) -> None:
    """Renames a file."""
    os.rename(original, target)


def current_executor_path() -> str:
    """Current executor path"""
    return os.getcwd()


def path_join(path1: str, path2: str) -> str:
    """Joins two paths together"""
    return os.path.join(path1, path2)


def check_file_exists(file_path: str) -> bool:
    """Checks if a file exists."""
    return os.path.isfile(file_path)


def check_if_folder_exists(song_output: str) -> bool:
    """Checks if a folder exists."""
    return os.path.isdir(song_output)


def remove_folder(cache_path):
    """Removes a folder."""
    shutil.rmtree(cache_path)


FILENAME_REPLACEMENTS = (('?:"', ""), ("<", "("), (">", ")"), ("/\\|*", "-"))

# Characters of a song name ("Artist - Title") in the names of its folder and
# files. The longest paths made from it are those of the separated stems, on
# Windows 177 characters plus the user name plus the song name:
#   %LOCALAPPDATA%\UltraSinger\cache\<cache folder>\separated\<model>\<song name>\no_vocals.wav
# and, in a cache kept in the song folder, which holds the name twice:
#   <output folder>\<song name> (2)\cache\separated\<model>\<song name>\no_vocals.wav
# With 60, both stay below the Windows path limit of 260 characters for a user
# name of up to 20 and an output folder of up to 50 characters.
MAX_NAME_CHARS = 60
# Hex characters of the hash that ends a shortened name
_NAME_HASH_CHARS = 6


def sanitize_filename(fname: str) -> str:
    """Sanitize filename: without the characters that paths cannot hold, and
    shortened to ``MAX_NAME_CHARS`` characters (see ``cap_name_length``)"""
    for old, new in FILENAME_REPLACEMENTS:
        for char in old:
            fname = fname.replace(char, new)
    # Windows silently drops trailing periods and spaces from folder names,
    # which would make the created path differ from the one used later on
    return cap_name_length(fname.rstrip(" ."))


def cap_name_length(name: str, max_chars: int = MAX_NAME_CHARS) -> str:
    """``name``, or when it is longer than ``max_chars`` characters its
    beginning (ended at a word) and a short hash of the whole name: a name is
    always shortened alike, and long names that begin alike stay apart."""
    if len(name) <= max_chars:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:_NAME_HASH_CHARS]
    head = name[:max_chars - _NAME_HASH_CHARS - 1]
    # End at a word, unless that drops more than a third
    word_end = head.rfind(" ")
    if name[len(head)] != " " and word_end >= len(head) * 2 // 3:
        head = head[:word_end]
    # No dangling separator, and no trailing period or space (see sanitize_filename)
    head = head.rstrip(" .-([")
    return f"{head} {digest}" if head else digest


def get_unused_song_output_dir(path: str) -> str:
    """``path``, or when that folder exists the first of ``path (1)``,
    ``path (2)``, ... that does not (only the last folder name changes)"""
    if not check_if_folder_exists(path):
        return path
    for i in range(1, 1000):
        candidate = f"{path} ({i})"
        if not check_if_folder_exists(candidate):
            return candidate
    print(
        f"{ULTRASINGER_HEAD} {red_highlighted('Error: Could not create output folder! (999) is the maximum number of tries.')}"
    )
    raise ValueError("Could not create output folder! (999) is the maximum number of tries.")
