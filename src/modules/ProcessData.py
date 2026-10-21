from dataclasses import dataclass, field
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Pitcher.pitched_data import PitchedData
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from modules.Midi.MidiSegment import MidiSegment
from typing import Optional, List, Tuple

@dataclass
class ProcessDataPaths:
    # Process data Paths
    processing_audio_path: Optional[str] = ""  # Muted audio for pitch detection
    whisper_audio_path: Optional[str] = ""  # Un-muted audio for Whisper transcription
    cache_folder_path: Optional[str] = ""
    audio_output_file_path: Optional[str] = "" # Output audio file path
    vocals_audio_file_path: Optional[str] = "" # Separated vocals audio file path
    instrumental_audio_file_path: Optional[str] = "" # Separated instrumental audio file path

@dataclass
class MediaInfo:
    """Media Info"""
    title: str
    artist: str
    bpm: Optional[float] = None
    year: Optional[str] = None
    genre: Optional[str] = None
    language: Optional[str] = None
    cover_url: Optional[str] = None
    cover_from_video: bool = False  # Whether the cover is the video's thumbnail (not the song database's)
    video_thumbnail_tried: bool = False  # Whether the video's thumbnail was downloaded or tried (once per run)
    video_thumbnail_url: Optional[str] = None  # The URL of the video's thumbnail, from the video's info
    video_url: Optional[str] = None
    music_key: Optional[str] = None  # (e.g., "C major", "A minor")
    video_extension: Optional[str] = None  # (e.g., "mp4", "mkv")
    audio_extension: Optional[str] = None  # (e.g., "mp3", "m4a")

@dataclass
class ProcessData:
    """Data for processing"""
    process_data_paths: ProcessDataPaths = field(default_factory=ProcessDataPaths)
    basename: Optional[str] = None
    media_info: Optional[MediaInfo] = None
    transcribed_data: Optional[List[TranscribedData]] = field(default_factory=list)
    # The transcription's words as (text, start s, end s), as the transcriber placed them: the
    # corrections of transcribed_data do not change them (they give plain lyrics their line times)
    transcription_words: List[Tuple[str, float, float]] = field(default_factory=list)
    pitched_data: Optional[PitchedData] = None
    midi_segments: Optional[List[MidiSegment]] = field(default_factory=list)
    parsed_file: Optional[UltrastarTxtValue] = None
    synced_lyrics: Optional[str] = None  # Synced LRC lyrics from LRCLIB (for reference-first pipeline)
    plain_lyrics: Optional[str] = None  # Plain text lyrics from LRCLIB (for plain-lyrics alignment and pitch_notes overlay)
    synced_lyrics_other_language: Optional[str] = None  # Language of found synced lyrics not used: another one than the singing
    synced_lyrics_length: Optional[float] = None  # Length (s) of the recording of the synced lyrics, when LRCLIB knows it
    lyrics_source: Optional[str] = None  # Where the found lyrics come from (LyricsInfo.source)
    # How many of the transcription's words the found lyrics have (and the ones of the second lyrics
    # database, when it was asked for them); None when they were not compared
    lyrics_heard: Optional[str] = None
    # The lyrics of the second lyrics database (LyricsInfo) that replaced them: a later lookup keeps them
    lyrics_instead: Optional[object] = None
    # Why found synced lyrics were taken for another recording's: their times were not used
    synced_lyrics_other_recording: Optional[str] = None
    plain_lyrics_from_synced: bool = False  # Whether the plain lyrics are the text of such synced lyrics
    # How the transcription's words checked the times of the synced lyrics (SyncedLyricsCheck), or why
    # they did not ("no (...)"); None when no check was tried
    synced_lyrics_check: Optional[object] = None
    # The transcription made to check them (TranscriptionResult): a later full transcription reuses it
    check_transcription: Optional[object] = None
