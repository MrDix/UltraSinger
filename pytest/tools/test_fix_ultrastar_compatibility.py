"""Tests for tools/fix_ultrastar_compatibility.py -- audio, tag and encoding fixes.

Uses generic placeholder names only. FFmpeg and ffprobe never run: the
conversion and the length probe are replaced by fakes (a child Python
process stands in where the real decoding of tool output matters).
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import fix_ultrastar_compatibility as fx  # noqa: E402

# A Polish pangram: every letter here is in CP1250, and the CP1250 bytes
# decode as different (wrong) letters in Windows-1252.
PANGRAM = "Zażółć gęślą jaźń"


def _song(tmp_path: Path, lines: list[str], *, encoding: str = "utf-8", newline: str = "\n",
          files: tuple[str, ...] = ("song.mp3",)) -> Path:
    d = tmp_path / "Artist - Title"
    d.mkdir(parents=True, exist_ok=True)
    for name in files:
        (d / name).write_bytes(b"fake audio")
    txt = d / "song.txt"
    txt.write_bytes(newline.join(lines + [""]).encode(encoding))
    return txt


# ── #ENCODING tag ────────────────────────────────────────────────────────────

class TestEncodingTag:
    LINES = ["#TITLE:Title", "#ARTIST:Artist", "#ENCODING:CP1250", "#MP3:song.mp3",
             "#BPM:300", "#GAP:0", f": 0 4 60 {PANGRAM}", "E"]

    def test_read_in_the_declared_codepage(self, tmp_path):
        lines, enc = fx._read_txt_lines(_song(tmp_path, self.LINES, encoding="cp1250"))
        assert enc == "cp1250"
        assert lines[6].rstrip() == f": 0 4 60 {PANGRAM}"

    def test_normalize_writes_the_real_text_as_utf8(self, tmp_path):
        txt = _song(tmp_path, self.LINES, encoding="cp1250", newline="\r\n")
        assert fx.main([str(tmp_path), "--normalize-encoding"]) == 0
        text = txt.read_bytes().decode("utf-8")
        assert f": 0 4 60 {PANGRAM}\r\n" in text
        assert "#ENCODING" not in text

    def test_without_a_tag_legacy_text_is_read_as_cp1252(self, tmp_path):
        lines = [line for line in self.LINES if not line.startswith("#ENCODING")]
        lines[-2] = ": 0 4 60 schön"
        _, enc = fx._read_txt_lines(_song(tmp_path, lines, encoding="cp1252"))
        assert enc == "cp1252"

    def test_tag_rewrite_keeps_the_declared_codepage(self, tmp_path):
        """Without --normalize-encoding, a rewrite keeps the chart in CP1250."""
        lines = self.LINES[:2] + ["#COMMENT:remove me"] + self.LINES[2:]
        txt = _song(tmp_path, lines, encoding="cp1250")
        assert fx.remove_non_standard_tags(txt) is True
        assert txt.read_bytes() == "\n".join(self.LINES + [""]).encode("cp1250")


# ── TXT rewrites never lose the chart ────────────────────────────────────────

class TestAtomicRewrite:
    def test_failed_rewrite_keeps_the_chart(self, tmp_path):
        """A rewrite that fails half-way (here: a new file name the chart's
        codepage cannot hold) must leave the chart exactly as it was."""
        txt = _song(tmp_path, ["#TITLE:Title", "#MP3:song.flac", "#BPM:300", "#GAP:0",
                               ": 0 4 60 café", "E"], encoding="cp1252", newline="\r\n")
        original = txt.read_bytes()
        with pytest.raises(UnicodeEncodeError):
            fx.update_txt_tags(txt, {"song.flac": "łąka.ogg"})
        assert txt.read_bytes() == original
        assert sorted(p.name for p in txt.parent.iterdir()) == ["song.mp3", "song.txt"]

    def test_interrupted_replace_keeps_the_chart(self, tmp_path, monkeypatch):
        txt = _song(tmp_path, ["#TITLE:Title", "#COMMENT:x", "#MP3:song.mp3", "#BPM:300", ": 0 4 60 la", "E"])
        original = txt.read_bytes()

        def disk_full(src, dst):
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(fx.os, "replace", disk_full)

        with pytest.raises(OSError):
            fx.remove_non_standard_tags(txt)
        assert txt.read_bytes() == original
        assert sorted(p.name for p in txt.parent.iterdir()) == ["song.mp3", "song.txt"]

    def test_rewrite_keeps_line_endings_and_leaves_no_temporary_file(self, tmp_path):
        txt = _song(tmp_path, ["#TITLE:Title", "#COMMENT:remove me", "#MP3:song.mp3", "#BPM:300",
                               ": 0 4 60 la", "E"], newline="\r\n")
        assert fx.remove_non_standard_tags(txt) is True
        assert txt.read_bytes() == b"#TITLE:Title\r\n#MP3:song.mp3\r\n#BPM:300\r\n: 0 4 60 la\r\nE\r\n"
        assert sorted(p.name for p in txt.parent.iterdir()) == ["song.mp3", "song.txt"]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_file_mode_kept(self, tmp_path):
        txt = _song(tmp_path, ["#TITLE:Title", "#COMMENT:x", "#MP3:song.mp3", "#BPM:300", ": 0 4 60 la", "E"])
        txt.chmod(0o644)
        fx.remove_non_standard_tags(txt)
        assert stat.S_IMODE(txt.stat().st_mode) == 0o644


# ── OGG conversion ───────────────────────────────────────────────────────────

class TestConvertToOgg:
    def test_written_to_a_temporary_file_renamed_on_success(self, tmp_path, monkeypatch):
        calls = []

        def fake_ffmpeg(cmd, **kwargs):
            calls.append(cmd)
            Path(cmd[-1]).write_bytes(b"OggS complete")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        monkeypatch.setattr(fx.subprocess, "run", fake_ffmpeg)

        out = tmp_path / "song.ogg"
        fx.convert_to_ogg("ffmpeg", tmp_path / "song.flac", out)

        assert calls[0][-1] == str(out) + ".part" and calls[0][-3:-1] == ["-f", "ogg"]
        assert out.read_bytes() == b"OggS complete"
        assert [p.name for p in tmp_path.iterdir()] == ["song.ogg"]

    def test_failed_conversion_leaves_no_ogg(self, tmp_path, monkeypatch):
        def failing_ffmpeg(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"OggS half")
            return subprocess.CompletedProcess(cmd, 1, "", "Conversion failed!")
        monkeypatch.setattr(fx.subprocess, "run", failing_ffmpeg)

        with pytest.raises(RuntimeError, match="Conversion failed"):
            fx.convert_to_ogg("ffmpeg", tmp_path / "song.flac", tmp_path / "song.ogg")
        assert list(tmp_path.iterdir()) == []

    def test_non_ascii_error_message_survives(self, tmp_path, monkeypatch):
        """FFmpeg writes UTF-8. Decoded with the Windows console codepage, a
        file name with Ł in the error message used to lose the message."""
        child = ("import sys; sys.stderr.buffer.write('Error opening input \\u0141\\u201d'.encode('utf-8'));"
                 " sys.exit(1)")
        real_run = subprocess.run

        def fake_ffmpeg(cmd, **kwargs):
            return real_run([sys.executable, "-c", child], **kwargs)
        monkeypatch.setattr(fx.subprocess, "run", fake_ffmpeg)

        with pytest.raises(RuntimeError, match="Error opening input Ł”"):
            fx.convert_to_ogg("ffmpeg", tmp_path / "x.flac", tmp_path / "x.ogg")


class TestAudioSeconds:
    def test_container_length_when_the_stream_has_none(self, monkeypatch):
        out = '{"streams": [{"duration": "N/A"}], "format": {"duration": "212.4"}}'
        monkeypatch.setattr(fx.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, out, ""))
        assert fx._audio_seconds("ffprobe", Path("x.webm")) == 212.4

    def test_none_without_audio(self, monkeypatch):
        monkeypatch.setattr(fx.subprocess, "run",
                            lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, '{"streams": []}', ""))
        assert fx._audio_seconds("ffprobe", Path("x.ogg")) is None

    def test_none_when_ffprobe_fails(self, monkeypatch):
        monkeypatch.setattr(fx.subprocess, "run",
                            lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "Invalid data"))
        assert fx._audio_seconds("ffprobe", Path("x.ogg")) is None


class TestExistingOgg:
    """An OGG left by an aborted conversion must not become the song's audio."""

    LINES = ["#TITLE:Title", "#ARTIST:Artist", "#MP3:song.flac", "#BPM:300", "#GAP:0", ": 0 4 60 la", "E"]

    def _process(self, tmp_path, monkeypatch, *, ogg: bytes, seconds: dict | None,
                 lines: list[str] | None = None, files: tuple[str, ...] = ("song.flac",), fail: bool = False,
                 dry_run: bool = False):
        """process_song on a FLAC song with an existing song.ogg; *seconds* maps
        a suffix to the length ffprobe reports (None: there is no ffprobe)."""
        txt = _song(tmp_path, lines or self.LINES, files=files)
        (txt.parent / "song.ogg").write_bytes(ogg)
        converted = []

        def fake_convert(ffmpeg_path, input_path, output_path):
            converted.append(output_path.name)
            if fail:
                raise RuntimeError("FFmpeg conversion failed: broken input")
            output_path.write_bytes(b"OggS fresh")
        monkeypatch.setattr(fx, "convert_to_ogg", fake_convert)
        monkeypatch.setattr(fx, "check_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(fx, "find_ffprobe", lambda ffmpeg_path: None if seconds is None else "ffprobe")
        monkeypatch.setattr(fx, "_audio_seconds", lambda ffprobe_path, path: (seconds or {}).get(path.suffix))
        result = fx.process_song(fx.parse_song_info(txt), None, dry_run=dry_run)
        return txt, result, converted

    def test_empty_ogg_is_converted_again(self, tmp_path, monkeypatch):
        txt, result, converted = self._process(tmp_path, monkeypatch, ogg=b"", seconds={".flac": 200.0})
        assert converted == ["song.ogg"]
        assert (txt.parent / "song.ogg").read_bytes() == b"OggS fresh"
        assert not (txt.parent / "song.ogg.bak").exists()  # nothing to keep
        assert result.status == "converted"
        assert "#MP3:song.ogg" in txt.read_text(encoding="utf-8")

    def test_truncated_ogg_is_kept_aside_and_converted_again(self, tmp_path, monkeypatch):
        txt, result, converted = self._process(tmp_path, monkeypatch, ogg=b"OggS partial",
                                               seconds={".ogg": 61.0, ".flac": 200.0})
        assert converted == ["song.ogg"]
        assert (txt.parent / "song.ogg").read_bytes() == b"OggS fresh"
        assert (txt.parent / "song.ogg.bak").read_bytes() == b"OggS partial"
        assert any("song.ogg.bak" in w for w in result.warnings)

    def test_unreadable_ogg_is_kept_aside_and_converted_again(self, tmp_path, monkeypatch):
        (tmp_path / "Artist - Title").mkdir()
        (tmp_path / "Artist - Title" / "song.ogg.bak").write_bytes(b"older backup")
        txt, _, converted = self._process(tmp_path, monkeypatch, ogg=b"garbage", seconds={".flac": 200.0})
        assert converted == ["song.ogg"]
        assert (txt.parent / "song.ogg.bak").read_bytes() == b"older backup"
        assert (txt.parent / "song.ogg.bak.1").read_bytes() == b"garbage"

    def test_failed_conversion_puts_the_old_ogg_back(self, tmp_path, monkeypatch):
        txt, result, converted = self._process(tmp_path, monkeypatch, ogg=b"OggS partial", fail=True,
                                               seconds={".ogg": 61.0, ".flac": 200.0})
        assert converted == ["song.ogg"] and result.status == "error"
        assert (txt.parent / "song.ogg").read_bytes() == b"OggS partial"
        assert not (txt.parent / "song.ogg.bak").exists()
        assert "#MP3:song.flac" in txt.read_text(encoding="utf-8")

    def test_complete_ogg_is_reused(self, tmp_path, monkeypatch):
        txt, result, converted = self._process(tmp_path, monkeypatch, ogg=b"OggS complete",
                                               seconds={".ogg": 199.8, ".flac": 200.0})
        assert converted == []
        assert (txt.parent / "song.ogg").read_bytes() == b"OggS complete"
        assert result.status == "converted"
        assert "#MP3:song.ogg" in txt.read_text(encoding="utf-8")

    def test_ogg_not_trusted_without_ffprobe(self, tmp_path, monkeypatch):
        txt, _, converted = self._process(tmp_path, monkeypatch, ogg=b"OggS unknown", seconds=None)
        assert converted == ["song.ogg"]
        assert (txt.parent / "song.ogg.bak").read_bytes() == b"OggS unknown"

    @pytest.mark.parametrize("dry_run", [False, True])
    @pytest.mark.parametrize("ogg_seconds", [150.0, 200.0])
    def test_ogg_the_chart_uses_for_other_audio_is_never_touched(self, tmp_path, monkeypatch, ogg_seconds, dry_run):
        """song.ogg is the chart's instrumental, not a conversion of song.flac
        (a dry run reports the same as a real run)."""
        lines = self.LINES[:3] + ["#INSTRUMENTAL:song.ogg"] + self.LINES[3:]
        txt, result, converted = self._process(tmp_path, monkeypatch, ogg=b"OggS instrumental", lines=lines,
                                               seconds={".ogg": ogg_seconds, ".flac": 200.0}, dry_run=dry_run)
        assert converted == [] and result.status == "error"
        assert "used by #INSTRUMENTAL" in result.error_msg
        assert (txt.parent / "song.ogg").read_bytes() == b"OggS instrumental"
        assert not (txt.parent / "song.ogg.bak").exists()
        assert txt.read_bytes() == "\n".join(lines + [""]).encode()

    def test_tag_of_a_missing_file_does_not_claim_the_ogg(self, tmp_path, monkeypatch):
        """#INSTRUMENTAL names a missing file; song.ogg is only its stand-in (the
        single .ogg in the folder), so it is not the chart's instrumental."""
        lines = self.LINES[:3] + ["#INSTRUMENTAL:other.ogg"] + self.LINES[3:]
        txt, result, converted = self._process(tmp_path, monkeypatch, ogg=b"OggS complete", lines=lines,
                                               seconds={".ogg": 199.8, ".flac": 200.0})
        assert result.status == "converted" and converted == []
        text = txt.read_text(encoding="utf-8")
        assert "#MP3:song.ogg" in text and "#INSTRUMENTAL:other.ogg" in text

    def test_two_sources_for_one_ogg_name_are_refused(self, tmp_path, monkeypatch):
        lines = self.LINES[:3] + ["#VOCALS:song.m4a"] + self.LINES[3:]
        txt, result, converted = self._process(tmp_path, monkeypatch, ogg=b"", lines=lines,
                                               files=("song.flac", "song.m4a"), seconds={".flac": 200.0})
        assert converted == [] and result.status == "error"
        assert "would both become song.ogg" in result.error_msg
        assert "#MP3:song.flac" in txt.read_text(encoding="utf-8")

    def test_rewired_tag_keeps_the_line_ending(self, tmp_path, monkeypatch):
        txt = _song(tmp_path, self.LINES, files=("song.flac",), newline="\r\n")
        fx.update_txt_tags(txt, {"song.flac": "song.ogg"})
        assert txt.read_bytes() == "\r\n".join(self.LINES + [""]).replace("song.flac", "song.ogg").encode()


# ── One song never stops the run ─────────────────────────────────────────────

class TestRunSurvivesSongErrors:
    def test_one_failing_song_does_not_stop_the_run(self, tmp_path, monkeypatch, capsys):
        lines = ["#TITLE:Title", "#ARTIST:Artist", "#COMMENT:x", "#MP3:song.mp3", "#BPM:300", ": 0 4 60 la", "E"]
        txt_a = _song(tmp_path / "A", lines)
        txt_b = _song(tmp_path / "B", lines)
        real = fx.process_song

        def flaky(song_info, *a, **k):
            if song_info.txt_path == txt_a:
                raise TypeError("'NoneType' object has no attribute 'strip'")
            return real(song_info, *a, **k)
        monkeypatch.setattr(fx, "process_song", flaky)

        assert fx.main([str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "ERROR:" in out and "Unexpected error" in out
        assert "#COMMENT" not in txt_b.read_text(encoding="utf-8")
