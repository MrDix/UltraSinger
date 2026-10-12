"""Tests for the cache of the vocal separation (separation.py); no model is ever loaded."""

from pathlib import Path

import pytest

from src.modules.Audio import separation
from src.modules.Audio.separation import AudioSeparatorModel, DemucsModel, SeparatorBackend


def _audio_separator(calls):
    """A stand-in for the separation: stems that tell which audio they came from."""
    def separate(input_file_path, output_dir, model):
        calls.append(input_file_path)
        Path(output_dir, "vocals.wav").write_bytes(b"vocals of " + Path(input_file_path).read_bytes())
        Path(output_dir, "no_vocals.wav").write_bytes(b"instrumental")
    return separate


def _separate(cache: Path, song: Path, **kwargs) -> str:
    return separation.separate_vocal_from_audio(str(cache), str(song), True, False, "cpu",
                                                AudioSeparatorModel.MEL_BAND_ROFORMER, **kwargs)


@pytest.fixture
def song(tmp_path):
    path = tmp_path / "Artist - Title.ogg"
    path.write_bytes(b"mix A")
    return path


class TestSeparationCache:
    def test_stems_are_reused_for_the_same_audio_only(self, tmp_path, song, monkeypatch, capsys):
        calls = []
        monkeypatch.setattr(separation, "_separate_with_audio_separator", _audio_separator(calls))
        folder = _separate(tmp_path / "cache", song)
        assert _separate(tmp_path / "cache", song) == folder
        assert len(calls) == 1 and "reusing cached separated vocals" in capsys.readouterr().out
        # Another file of the same name, as a cache folder that is used again can hold
        song.write_bytes(b"mix B")
        assert _separate(tmp_path / "cache", song) == folder
        assert len(calls) == 2 and Path(folder, "vocals.wav").read_bytes() == b"vocals of mix B"

    def test_skip_cache_separates_again(self, tmp_path, song, monkeypatch):
        calls = []
        monkeypatch.setattr(separation, "_separate_with_audio_separator", _audio_separator(calls))
        _separate(tmp_path / "cache", song)
        _separate(tmp_path / "cache", song, skip_cache=True)
        assert len(calls) == 2

    def test_stems_of_an_interrupted_separation_are_not_reused(self, tmp_path, song, monkeypatch):
        calls = []
        monkeypatch.setattr(separation, "_separate_with_audio_separator", _audio_separator(calls))
        _separate(tmp_path / "cache", song)
        song.write_bytes(b"mix B")

        def interrupted(input_file_path, output_dir, model):
            Path(output_dir, "vocals.wav").write_bytes(b"half of the vocals of mix B")
            raise KeyboardInterrupt
        monkeypatch.setattr(separation, "_separate_with_audio_separator", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _separate(tmp_path / "cache", song)
        # Back to the first file: the stems in the folder are no longer its stems
        song.write_bytes(b"mix A")
        monkeypatch.setattr(separation, "_separate_with_audio_separator", _audio_separator(calls))
        folder = _separate(tmp_path / "cache", song)
        assert len(calls) == 2 and Path(folder, "vocals.wav").read_bytes() == b"vocals of mix A"

    def test_demucs_stems_are_keyed_alike(self, tmp_path, song, monkeypatch):
        calls = []

        def demucs(input_file_path, output_folder, model, device):
            # Demucs makes separated/<model>/<file name>/ in the folder it is given
            calls.append(input_file_path)
            out = Path(output_folder, "separated", model.value, Path(input_file_path).stem)
            out.mkdir(parents=True, exist_ok=True)
            (out / "vocals.wav").write_bytes(b"vocals")
            (out / "no_vocals.wav").write_bytes(b"instrumental")
        monkeypatch.setattr(separation, "_separate_with_demucs", demucs)
        for content in (b"mix A", b"mix A", b"mix B"):
            song.write_bytes(content)
            separation.separate_vocal_from_audio(str(tmp_path / "cache"), str(song), True, False, "cpu",
                                                 DemucsModel.HTDEMUCS, backend=SeparatorBackend.DEMUCS)
        assert len(calls) == 2

    def test_nothing_is_read_without_separation(self, tmp_path, song, monkeypatch):
        # Neither the separated vocal nor a karaoke file wanted: the audio is not even read
        monkeypatch.setattr(separation, "content_key", lambda path: pytest.fail("audio read"))
        separation.separate_vocal_from_audio(str(tmp_path / "cache"), str(song), False, False, "cpu")
