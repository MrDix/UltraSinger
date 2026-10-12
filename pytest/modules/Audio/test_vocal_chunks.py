"""Tests for the audio chunks of --create_audio_chunks (vocal_chunks.py)."""

import wave

import numpy as np

from src.modules.Audio import vocal_chunks
from src.modules.ProcessData import ProcessDataPaths
from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.Ultrastar.ultrastar_txt import UltrastarNoteLine, UltrastarTxtValue

RATE = 1000  # frames per second


def _stereo_wav(path, seconds=3):
    """A stereo wav whose frame n holds the sample value n in both channels."""
    samples = np.repeat(np.arange(seconds * RATE, dtype=np.int16), 2)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes(samples.tobytes())


def _frames(path):
    """The frame numbers in a chunk made from _stereo_wav."""
    with wave.open(str(path), "rb") as file:
        samples = np.frombuffer(file.readframes(file.getnframes()), dtype=np.int16)
    return samples[::2]


def _paths(tmp_path, **kwargs):
    wav = tmp_path / "vocals.wav"
    _stereo_wav(wav)
    return ProcessDataPaths(cache_folder_path=str(tmp_path / "cache"), whisper_audio_path=str(wav), **kwargs)


class TestAudioChunks:
    def test_chunks_of_transcribed_words(self, tmp_path):
        words = [TranscribedData(word="one ", start=0.5, end=0.75), TranscribedData(word="two", start=2.0, end=2.5)]
        vocal_chunks.create_audio_chunks_from_transcribed_data(_paths(tmp_path), words)
        chunks = tmp_path / "cache" / "audio-chunks"
        # Positions are frames, not samples of all channels: not from 1.0 s, 4.0 s on
        assert list(_frames(chunks / "chunk_0_one.wav")) == list(range(500, 750))
        assert list(_frames(chunks / "chunk_1_two.wav")) == list(range(2000, 2500))

    def test_chunks_of_a_txt_input_come_from_the_vocal_wav(self, tmp_path):
        song = tmp_path / "Artist - Title.mp3"  # no wav: the wave module cannot read it
        song.write_bytes(b"ID3")
        txt = UltrastarTxtValue()
        txt.bpm, txt.gap = "15", "500"  # a beat is a second, the first one at 0.5 s
        txt.UltrastarNoteLines = [UltrastarNoteLine(startBeat=0, startTime=0.5, endTime=1.5, duration=1,
                                                    pitch=0, word="one", noteType=":")]
        vocal_chunks.create_audio_chunks_from_ultrastar_data(_paths(tmp_path, audio_output_file_path=str(song)), txt)
        assert list(_frames(tmp_path / "cache" / "audio-chunks" / "chunk_0_one.wav")) == list(range(500, 1500))

    def test_word_after_the_end_of_the_audio(self, tmp_path):
        words = [TranscribedData(word="late", start=3.5, end=4.0)]
        vocal_chunks.create_audio_chunks_from_transcribed_data(_paths(tmp_path), words)
        assert len(_frames(tmp_path / "cache" / "audio-chunks" / "chunk_0_late.wav")) == 0
