"""Tests for the vocal separation (separation.py): its cache, device, GPU memory and
errors; no model is ever loaded."""

import logging
import ntpath
import os
import weakref
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

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


def _separate_folder(cache: Path, song: Path) -> Path:
    """The folder of the stems of ``song`` that ``_separate`` makes."""
    return cache / "separated" / AudioSeparatorModel.MEL_BAND_ROFORMER.value / cap_name_length(song.stem)


def _write_stem(path: Path, seconds: float = 3.0) -> None:
    """A stem as the library writes it: a WAV file, here of silence."""
    sf.write(str(path), np.zeros((int(8000 * seconds), 2), dtype=np.float32), 8000)


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

    @pytest.mark.parametrize("name", ["Artist - " + "Long Title " * 10, "Artist - Title {Live}", "-intro"])
    def test_demucs_stem_folder(self, tmp_path, monkeypatch, name):
        # A long file name gets a short stem folder with Demucs too; braces are no
        # pattern, and a leading "-" no option
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
            _write_stem(Path(self.output_dir, f"{name}.wav"))


@pytest.fixture
def library_separator(monkeypatch):
    import audio_separator.separator as sepmod
    monkeypatch.setattr(sepmod, "Separator", _LibrarySeparator)
    _LibrarySeparator.devices = []
    _LibrarySeparator.alive = []
    return _LibrarySeparator


class _Model:
    """Stand-in for the library's model (the class of its architecture): demixes, then writes the stems."""

    def __init__(self, library):
        self.library = library

    def separate(self, path, custom_output_names=None):
        self.demix()
        self.library.events.append("write")
        for name in custom_output_names.values():
            # The library logs the error of writing a stem, and goes on with the
            # next; a near-silent stem it does not write at all
            if name != self.library.skipped:
                _write_stem(Path(self.library.output_dir, f"{name}.wav"), 0.5 if name == self.library.cut else 3.0)
        return [f"{name}.wav" for name in custom_output_names.values()]

    def demix(self):
        self.library.events.append("demix")
        if self.library.failures:
            raise self.library.failures.pop(0)


class _SwallowingSeparator:
    """Stand-in for audio-separator's Separator that, like the library's, logs
    the error of a separation and returns as if it had written the stems."""

    failures = []  # errors of the next demixings
    cut = None  # name of a stem whose writing fails halfway
    skipped = None  # name of a stem not written
    events = []

    def __init__(self, output_dir, **kwargs):
        self.output_dir = output_dir
        self.setup_torch_device(None)

    def setup_torch_device(self, system_info):
        self.torch_device = "cuda"

    def load_model(self, model_filename):
        self.events.append("load")
        self.model_instance = _Model(self)

    def separate(self, path, custom_output_names=None):
        try:
            return self.model_instance.separate(path, custom_output_names)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to process file {path}: {e}")
            return []


@pytest.fixture
def swallowing_separator(monkeypatch):
    import audio_separator.separator as sepmod
    monkeypatch.setattr(sepmod, "Separator", _SwallowingSeparator)
    _SwallowingSeparator.failures = []
    _SwallowingSeparator.cut = None
    _SwallowingSeparator.skipped = None
    _SwallowingSeparator.events = []
    monkeypatch.setattr(separation, "release_gpu_memory", lambda: _SwallowingSeparator.events.append("release"))
    return _SwallowingSeparator


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
        # After the model load, and at the end
        assert released == [[False], [True]]

    def test_released_after_a_failed_separation(self, tmp_path, song, library_separator, monkeypatch):
        released = []
        monkeypatch.setattr(separation, "release_gpu_memory", lambda: released.append(True))
        monkeypatch.setattr(library_separator, "separate", lambda *args, **kwargs: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            _separate(tmp_path / "cache", song)
        assert released == [True, True]

    def test_released_after_demucs(self, tmp_path, song, monkeypatch):
        released = []
        monkeypatch.setattr(separation, "release_gpu_memory", lambda: released.append(True))
        monkeypatch.setattr(separation.demucs.separate, "main", lambda args: None)
        separation._separate_with_demucs(str(song), str(tmp_path), DemucsModel.HTDEMUCS)
        assert released == [True]

    def test_released_after_the_model_load_and_after_the_demixing(self, tmp_path, song, swallowing_separator):
        # On Windows the GPU memory counts toward the memory the system can commit,
        # and the stems are written in the main memory
        _separate(tmp_path / "cache", song)
        assert swallowing_separator.events == ["load", "release", "demix", "release", "write", "release"]


class TestFailedSeparation:
    """audio-separator logs the error of a separation and returns as if it had written the stems."""

    def test_error_is_raised(self, tmp_path, song, swallowing_separator):
        swallowing_separator.failures = [ValueError("the file holds no audio")]
        with pytest.raises(separation.SeparationFailed, match="the file holds no audio") as raised:
            _separate(tmp_path / "cache", song)
        assert isinstance(raised.value.__cause__, ValueError) and not raised.value.out_of_memory
        assert not Path(_separate_folder(tmp_path / "cache", song), separation.SOURCE_KEY_FILE).exists()

    def test_stems_of_an_earlier_separation_are_not_taken(self, tmp_path, song, swallowing_separator):
        folder = _separate(tmp_path / "cache", song)
        song.write_bytes(b"mix B")
        swallowing_separator.failures = [ValueError("the file holds no audio")]
        with pytest.raises(separation.SeparationFailed):
            _separate(tmp_path / "cache", song)
        assert not any(Path(folder, name).exists()
                       for name in ("vocals.wav", "no_vocals.wav", separation.SOURCE_KEY_FILE))

    @pytest.mark.parametrize("error", [
        MemoryError("Unable to allocate 114. MiB for an array with shape (2, 14879936) and data type float32"),
        RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"),
    ])
    def test_out_of_memory(self, tmp_path, song, swallowing_separator, capsys, error):
        swallowing_separator.failures = [error]
        with pytest.raises(separation.SeparationFailed, match="ran out of memory") as raised:
            _separate(tmp_path / "cache", song)
        assert raised.value.out_of_memory and raised.value.__cause__ is error
        assert str(error) in str(raised.value)
        out = capsys.readouterr().out
        assert "Error:" in out and "close other programs" in out
        assert not Path(_separate_folder(tmp_path / "cache", song), separation.SOURCE_KEY_FILE).exists()

    def test_a_stem_cut_short(self, tmp_path, song, swallowing_separator):
        # The library logs the error of writing a stem (as a full disk) and goes on
        swallowing_separator.cut = "vocals"
        with pytest.raises(separation.SeparationFailed, match=r"cut short \(vocals.wav 0.5 s, no_vocals.wav 3.0 s\)"):
            _separate(tmp_path / "cache", song)
        assert not any(_separate_folder(tmp_path / "cache", song).iterdir())

    def test_a_stem_not_written(self, tmp_path, song, swallowing_separator):
        swallowing_separator.skipped = "vocals"
        with pytest.raises(separation.SeparationFailed, match="wrote no readable vocals.wav"):
            _separate(tmp_path / "cache", song)
        assert not any(_separate_folder(tmp_path / "cache", song).iterdir())

    def test_a_stem_that_cannot_be_removed(self, tmp_path, song, swallowing_separator, monkeypatch, capsys):
        # As one that another program holds open: the error of the separation goes on
        def remove(path):
            raise PermissionError(13, "The process cannot access the file", path)
        swallowing_separator.cut = "vocals"
        monkeypatch.setattr(separation.os, "remove", remove)
        with pytest.raises(separation.SeparationFailed, match="cut short"):
            _separate(tmp_path / "cache", song)
        assert "could not remove" in capsys.readouterr().out

    def test_a_separation_without_stems(self, tmp_path, song, monkeypatch):
        # A library that went on without its stems, and logged nothing to raise
        monkeypatch.setattr(separation, "_separate_with_audio_separator", lambda *args, **kwargs: None)
        with pytest.raises(separation.SeparationFailed, match="wrote no vocals.wav and no no_vocals.wav"):
            _separate(tmp_path / "cache", song)
        assert not Path(_separate_folder(tmp_path / "cache", song), separation.SOURCE_KEY_FILE).exists()

    def test_the_model_keeps_its_own_methods(self, tmp_path, song, swallowing_separator):
        separator = separation.create_separator(str(tmp_path))
        separator.load_model(model_filename="model.ckpt")
        separation.run_separation(separator, str(song), str(tmp_path), separation.STEM_NAMES)
        swallowing_separator.failures = [ValueError("the file holds no audio")]
        with pytest.raises(separation.SeparationFailed):
            separation.run_separation(separator, str(song), str(tmp_path), separation.STEM_NAMES)
        assert not {"separate", "demix"} & set(vars(separator.model_instance))

    def test_lead_vocal_separation(self, tmp_path, swallowing_separator):
        from modules.Audio.separation import SeparationFailed
        from modules.Segmentation import lead_vocal
        vocals = tmp_path / "vocals.wav"
        vocals.write_bytes(b"vocals")
        swallowing_separator.failures = [MemoryError("Unable to allocate 57.0 MiB")]
        with pytest.raises(SeparationFailed, match=r"ran out of memory \(Unable to allocate 57.0 MiB\)"):
            lead_vocal.separate_lead_vocal(str(vocals), str(tmp_path), force_cpu=True)

    def test_lead_stem_cut_short_is_not_cached(self, tmp_path, swallowing_separator):
        # The lead stem that is there counts as cached: one cut short must not stay
        from modules.Audio.separation import SeparationFailed
        from modules.Segmentation import lead_vocal
        vocals = tmp_path / "vocals.wav"
        vocals.write_bytes(b"vocals")
        swallowing_separator.cut = "backing"
        with pytest.raises(SeparationFailed, match="cut short"):
            lead_vocal.separate_lead_vocal(str(vocals), str(tmp_path), force_cpu=True)
        swallowing_separator.cut = None
        lead_path = lead_vocal.separate_lead_vocal(str(vocals), str(tmp_path), force_cpu=True)
        assert swallowing_separator.events.count("demix") == 2 and sf.info(lead_path).duration == 3.0


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
