"""Tests for the cache of the vocal separation (separation.py); no model is ever loaded."""

import ntpath
import os
import weakref
from pathlib import Path

import pytest

from src.modules.Audio import separation
from src.modules.Audio.separation import AudioSeparatorModel, DemucsModel, SeparatorBackend
from src.modules.os_helper import MAX_NAME_CHARS, cap_name_length


def _audio_separator(calls):
    """A stand-in for the separation: stems that tell which audio they came from."""
    def separate(input_file_path, output_dir, model, force_cpu=False):
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

        def interrupted(input_file_path, output_dir, model, force_cpu=False):
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

        def demucs(input_file_path, output_folder, model, device, track_folder=None):
            # Demucs makes separated/<model>/<track folder>/ in the folder it is given
            calls.append(input_file_path)
            out = Path(output_folder, "separated", model.value, track_folder or Path(input_file_path).stem)
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

    def test_long_file_name_gets_a_short_stem_folder(self, tmp_path, monkeypatch):
        # The audio file of a txt input keeps its own name, however long
        song = tmp_path / ("Artist - " + "Long Title " * 10 + ".mp3")
        song.write_bytes(b"mix")
        monkeypatch.setattr(separation, "_separate_with_audio_separator", _audio_separator([]))
        folder = _separate(tmp_path / "cache", song)
        assert len(os.path.basename(folder)) <= MAX_NAME_CHARS
        assert os.path.basename(folder).startswith("Artist - Long Title")
        assert Path(folder, "vocals.wav").read_bytes() == b"vocals of mix"

    @pytest.mark.parametrize("name", ["Artist - " + "Long Title " * 10, "Artist - Title {Live}"])
    def test_demucs_stem_folder(self, tmp_path, monkeypatch, name):
        # A long file name gets a short stem folder with Demucs too; braces are no pattern
        import demucs.separate as demucs_separate
        song = tmp_path / (name + ".mp3")
        song.write_bytes(b"mix")

        def main(args):
            # Writes the stems where Demucs would, from its own arguments
            opts = demucs_separate.get_parser().parse_args(args)
            track = opts.tracks[0].name
            for stem in ("vocals", "no_vocals"):
                path = opts.out / opts.name / opts.filename.format(
                    track=track.rsplit(".", 1)[0], trackext=track.rsplit(".", 1)[-1], stem=stem, ext="wav")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"stem")
        monkeypatch.setattr(separation.demucs.separate, "main", main)
        monkeypatch.setattr(separation, "release_gpu_memory", lambda: None)
        folder = separation.separate_vocal_from_audio(str(tmp_path / "cache"), str(song), True, False, "cpu",
                                                      DemucsModel.HTDEMUCS, backend=SeparatorBackend.DEMUCS)
        assert os.path.basename(folder) == cap_name_length(os.path.splitext(song.name)[0])
        assert Path(folder, "vocals.wav").is_file() and Path(folder, "no_vocals.wav").is_file()


class _LibrarySeparator:
    """Stand-in for audio-separator's Separator: like the library's, its
    constructor picks the device, the GPU whenever PyTorch finds one."""

    devices = []  # device of each separation
    alive = []  # weak references to the instances

    def __init__(self, output_dir, **kwargs):
        self.output_dir = output_dir
        self.setup_torch_device(None)
        _LibrarySeparator.alive.append(weakref.ref(self))

    def setup_torch_device(self, system_info):
        self.torch_device = "cuda"

    def load_model(self, model_filename):
        pass

    def separate(self, path, custom_output_names=None):
        _LibrarySeparator.devices.append(str(self.torch_device))
        for name in custom_output_names.values():
            Path(self.output_dir, f"{name}.wav").write_bytes(b"stem")


@pytest.fixture
def library_separator(monkeypatch):
    import audio_separator.separator as sepmod
    monkeypatch.setattr(sepmod, "Separator", _LibrarySeparator)
    _LibrarySeparator.devices = []
    _LibrarySeparator.alive = []
    return _LibrarySeparator


class TestForceCpu:
    """--force_cpu keeps both separations on the CPU: audio-separator picks the GPU by itself."""

    @pytest.fixture
    def gpu_found(self, tmp_path, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        # The folder the library would load models from (none is loaded)
        monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path))

    def test_library_takes_the_gpu_by_itself(self, tmp_path, gpu_found):
        assert separation.create_separator(str(tmp_path / "out")).torch_device.type == "cuda"

    def test_force_cpu_overrides_the_library(self, tmp_path, gpu_found):
        separator = separation.create_separator(str(tmp_path / "out"), force_cpu=True)
        assert separator.torch_device.type == "cpu"
        assert separator.onnx_execution_provider == ["CPUExecutionProvider"]

    def test_library_version_that_picks_the_device_otherwise(self, tmp_path, monkeypatch):
        # The override then has no effect: an error, not a separation on the GPU
        import torch
        import audio_separator.separator as sepmod

        class RenamedSetup:
            def __init__(self, output_dir, **kwargs):
                self.choose_device()

            def choose_device(self):
                self.torch_device = torch.device("cuda")
        monkeypatch.setattr(sepmod, "Separator", RenamedSetup)
        with pytest.raises(RuntimeError, match="did not keep to the CPU"):
            separation.create_separator(str(tmp_path), force_cpu=True)
        assert separation.create_separator(str(tmp_path)).torch_device.type == "cuda"

    def test_vocal_separation(self, tmp_path, song, library_separator):
        _separate(tmp_path / "cache", song, force_cpu=True)
        _separate(tmp_path / "other cache", song)
        assert library_separator.devices == ["cpu", "cuda"]

    def test_lead_vocal_separation(self, tmp_path, library_separator, monkeypatch):
        from modules.Segmentation import lead_vocal
        # On the CPU the free GPU memory does not matter
        monkeypatch.setattr(lead_vocal, "free_gpu_memory", lambda: pytest.fail("GPU memory read"))
        vocals = tmp_path / "vocals.wav"
        vocals.write_bytes(b"vocals")
        lead_vocal.separate_lead_vocal(str(vocals), str(tmp_path), force_cpu=True)
        assert library_separator.devices == ["cpu"]

    def test_pipeline_passes_force_cpu(self, tmp_path, song, monkeypatch):
        import src.UltraSinger as us
        seen = {}

        def separate(*args, **kwargs):
            seen.update(kwargs)
            return str(tmp_path)
        monkeypatch.setattr(us, "separate_vocal_from_audio", separate)
        monkeypatch.setattr(us, "denoise_vocal_audio", lambda *args, **kwargs: None)
        monkeypatch.setattr(us, "convert_audio_to_mono_wav", lambda *args: None)
        monkeypatch.setattr(us, "mute_no_singing_parts", lambda *args: None)
        monkeypatch.setattr(us.settings, "use_separated_vocal", False)
        monkeypatch.setattr(us.settings, "force_cpu", True)
        process_data = us.ProcessData(basename="Artist - Title")
        process_data.process_data_paths.cache_folder_path = str(tmp_path / "cache")
        process_data.process_data_paths.audio_output_file_path = str(song)
        us.CreateProcessAudio(process_data)
        assert seen["force_cpu"] is True


class TestGpuMemoryAfterSeparation:
    """The separation hands its GPU memory back before the next model (the transcription) loads."""

    def test_released_when_the_separator_is_gone(self, tmp_path, song, library_separator, monkeypatch):
        released = []
        monkeypatch.setattr(separation, "release_gpu_memory",
                            lambda: released.append([ref() is None for ref in library_separator.alive]))
        _separate(tmp_path / "cache", song)
        assert released == [[True]]

    def test_released_after_a_failed_separation(self, tmp_path, song, library_separator, monkeypatch):
        released = []
        monkeypatch.setattr(separation, "release_gpu_memory", lambda: released.append(True))
        monkeypatch.setattr(library_separator, "separate", lambda *args, **kwargs: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            _separate(tmp_path / "cache", song)
        assert released == [True]

    def test_released_after_demucs(self, tmp_path, song, monkeypatch):
        released = []
        monkeypatch.setattr(separation, "release_gpu_memory", lambda: released.append(True))
        monkeypatch.setattr(separation.demucs.separate, "main", lambda args: None)
        separation._separate_with_demucs(str(song), str(tmp_path), DemucsModel.HTDEMUCS)
        assert released == [True]


class TestPathLength:
    """The paths made from a song name stay below the Windows limit of 260 characters
    (see os_helper.MAX_NAME_CHARS)."""

    # Windows counts UTF-16 units: an emoji takes two
    @pytest.mark.parametrize("title", ["x" * 200, "\U0001F3B5" * 100])
    def test_longest_paths(self, tmp_path, title):
        from src.modules import conversion_cache
        from src.modules.os_helper import sanitize_filename, utf16_len

        name = sanitize_filename("Artist - " + title)
        song_folder = name + " (99)"
        cache_folder = os.path.basename(conversion_cache.create(song_folder, str(tmp_path)))
        model = max((m.value for m in AudioSeparatorModel), key=len)
        stem = ntpath.join(name, "no_vocals.wav")
        # A user name of 20 characters
        cache = ntpath.join("C:\\Users", "u" * 20, "AppData", "Local", "UltraSinger", "cache",
                            cache_folder, "separated", model, stem)
        # An output folder of 50 characters, with the cache kept in the song folder
        kept_cache = ntpath.join("D:\\" + "o" * 47, song_folder, "cache", "separated", model, stem)
        assert utf16_len(cache) < 260 and utf16_len(kept_cache) < 260
