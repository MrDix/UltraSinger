"""Chart review: look at a chart next to the singing, without singing it.

The game score says how well the vocal hits a chart, and the chart benchmark
(``tools/chart_benchmark.py``) needs a hand-made chart of the same song to
compare with. Neither shows *where* a chart goes wrong. This tool lays a
chart over the separated vocal of its song folder and writes

* **pages** (PNG): for every stretch of the song the spectrogram of the
  vocal with its pitch track and the notes, and below it the same stretch
  as a piano roll - the sung pitch in semitones against the notes, moved to
  the octave that is sung, with the lyrics. Harmonic singing shows as
  stacked lines in the spectrogram and a calm pitch track; harsh vocals as
  a cloud without lines and without pitch track; a run as steps of the
  pitch track that the notes follow or do not.
* **a report** (``review.md``): every line of the chart with how much of
  its notes is sung with a sure pitch, how much of that is on the notes'
  pitch, and what stands out - notes without a pitch that are scored, notes
  beside the sung pitch, sung pitches the chart leaves out, singing without
  notes. The lines that stand out are listed first.

    python tools/chart_review.py "D:/Songs/Artist - Title" D:/review/title
    python tools/chart_review.py chart.txt out_dir --vocals vocals.wav --from 60 --to 120

The vocal is the file of the chart's ``#VOCALS`` tag (UltraSinger writes it
next to the chart), or ``--vocals``. Without a separated vocal there is
nothing to compare with: the mix has the pitch of its instruments.

What stands out is found with UltraSinger's own passes, so the report says
what they would do: the freestyle detection (``unpitched_notes``) for notes
without a pitch, the plateaus of the vocal runs (``vocal_runs``) for sung
pitches that are not charted.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compare_ultrastar import _beat_to_ms, _parse_bpm, _parse_gap, _pitch_to_midi, absolute_chart_text, read_chart_text  # noqa: E402

SAMPLE_RATE = 16000
VOICED_CONFIDENCE = 0.7  # the pitch detection is sure of a pitch from here on
PITCHED_TYPES = (":", "*")
HIT_SEMITONES = 1.0  # a sung frame this close to its note (octaves folded) is on its pitch

# From when a line stands out
MIN_UNPITCHED_SHARE = 0.3  # of its scored notes are sung without a pitch
MIN_VOICED_SHARE = 0.4  # a line with less sure pitch than this has no pitch to be beside
MAX_ON_PITCH_SHARE = 0.6  # less of its sure pitch than this is on the notes
MIN_LEFT_OUT = 3  # sung pitches within its notes that have no note
MIN_UNCHARTED_S = 1.0  # seconds of sure pitch between its notes

SPECTROGRAM_HZ = (70.0, 5000.0)
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTE_COLOURS = {":": "#22aa22", "*": "#c8a000", "F": "#ff8c1a", "R": "#cc44aa", "G": "#cc44aa"}
UNPITCHED_COLOUR = "#e02020"  # a scored note the singing has no pitch in


@dataclass
class ReviewNote:
    kind: str
    start: float  # s
    end: float
    midi: int
    word: str
    line: int


@dataclass
class ReviewChart:
    headers: dict[str, str]
    notes: list[ReviewNote]


@dataclass
class LineReport:
    line: int
    start: float
    end: float
    text: str
    notes: int = 0  # scored notes (normal and golden)
    freestyle: int = 0
    voiced_share: float | None = None  # of the frames in its scored notes, those with a sure pitch
    on_pitch_share: float | None = None  # of those, the ones on the note's pitch
    unpitched: int = 0  # scored notes sung without a pitch
    left_out: int = 0  # sung pitches within its notes without a note of their own
    uncharted_s: float = 0.0  # sure pitch between its notes
    remarks: list[str] = field(default_factory=list)


def fold(semitones):
    """A difference in semitones as the games hear it: octaves folded (-6 ... +6)."""
    return (np.asarray(semitones, dtype=float) + 6.0) % 12.0 - 6.0


# A note line: type, start beat, length, pitch, and after one space the word
# (which may start or end with spaces of its own)
_NOTE_LINE_RE = re.compile(r"\s*([:*FRG])\s+(-?\d+)\s+(\d+)\s+(-?\d+)(?: (.*))?$")
# The line that starts the part of a singer in a duet: "P1", "P 2"
_SINGER_LINE_RE = re.compile(r"\s*P\s*(\d+)\s*$", re.IGNORECASE)


def load_chart(path: str | Path) -> ReviewChart:
    """The notes of an UltraStar TXT with their times in seconds and their line.

    Of a duet, the notes of the first singer: the vocal has both voices, but
    one after the other the two parts are no chart to lay over it.
    """
    headers: dict[str, str] = {}
    raw: list[tuple[str, int, int, int, str, int]] = []
    line = 0
    first_singer = True
    for text in absolute_chart_text(read_chart_text(path)).splitlines():
        singer = _SINGER_LINE_RE.match(text)
        note = _NOTE_LINE_RE.match(text)
        if text.startswith("#"):
            tag, _, value = text[1:].partition(":")
            headers[tag.strip().upper()] = value.strip()
        elif singer:
            first_singer = int(singer.group(1)) == 1
        elif text.strip().upper() == "E":
            break
        elif not first_singer:
            continue
        elif text.startswith("-"):
            line += 1
        elif note:
            raw.append((note.group(1), int(note.group(2)), int(note.group(3)), int(note.group(4)),
                        note.group(5) or "", line))
    bpm, gap = _parse_bpm(headers.get("BPM", "0")), _parse_gap(headers.get("GAP", "0"))
    version = headers.get("VERSION")
    notes = [ReviewNote(kind, _beat_to_ms(beat, bpm, gap) / 1000.0, _beat_to_ms(beat + length, bpm, gap) / 1000.0,
                        _pitch_to_midi(pitch, version), word, line)
             for kind, beat, length, pitch, word, line in raw]
    return ReviewChart(headers, notes)


def find_chart(song: Path) -> Path:
    """The chart of a song folder (or the given TXT)."""
    if song.is_file():
        return song
    charts = sorted(song.glob("*.txt"))
    if not charts:
        raise FileNotFoundError(f"no UltraStar TXT in {song}")
    return charts[0]


def find_vocals(chart_path: Path, headers: dict[str, str]) -> Path | None:
    """The separated vocal of the chart: the file of its #VOCALS tag."""
    name = headers.get("VOCALS")
    if name and (chart_path.parent / name).is_file():
        return chart_path.parent / name
    return None


def review_lines(chart: ReviewChart, times: np.ndarray, midi: np.ndarray, voiced: np.ndarray,
                 unpitched: list[bool] | None = None, left_out: list[int] | None = None) -> list[LineReport]:
    """What each line of the chart looks like against the singing.

    Args:
        chart: The chart.
        times: Times (s) of the pitch frames of the vocal.
        midi: Their pitch (fractional MIDI; any value where there is none).
        voiced: Which frames have a sure pitch.
        unpitched: For each note whether it is sung without a pitch.
        left_out: For each note the number of sung pitches in it without a note.
    """
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.016
    lines: dict[int, list[int]] = {}
    for index, note in enumerate(chart.notes):
        lines.setdefault(note.line, []).append(index)
    reports = []
    for line, indices in sorted(lines.items()):
        notes = [chart.notes[i] for i in indices]
        report = LineReport(line, notes[0].start, notes[-1].end,
                            "".join(note.word for note in notes).replace("~", "").strip())
        in_scored = np.zeros(len(times), dtype=bool)
        in_any = np.zeros(len(times), dtype=bool)
        on_pitch = np.zeros(len(times), dtype=bool)
        for index, note in zip(indices, notes):
            first, end = np.searchsorted(times, [note.start, note.end])
            in_any[first:end] = True
            if note.kind not in PITCHED_TYPES:
                report.freestyle += note.kind == "F"
                continue
            report.notes += 1
            in_scored[first:end] = True
            on_pitch[first:end] = np.abs(fold(midi[first:end] - note.midi)) <= HIT_SEMITONES
            if unpitched is not None and unpitched[index]:
                report.unpitched += 1
            if left_out is not None:
                report.left_out += left_out[index]
        frames = int(in_scored.sum())
        if frames:
            sure = in_scored & voiced
            report.voiced_share = float(sure.sum()) / frames
            if sure.any():
                report.on_pitch_share = float((sure & on_pitch).sum()) / float(sure.sum())
        first, end = np.searchsorted(times, [report.start, report.end])
        report.uncharted_s = float((voiced[first:end] & ~in_any[first:end]).sum()) * hop
        if report.notes and report.unpitched >= max(2.0, MIN_UNPITCHED_SHARE * report.notes):
            report.remarks.append(f"{report.unpitched} scored notes are sung without a pitch")
        if (report.voiced_share is not None and report.voiced_share >= MIN_VOICED_SHARE
                and report.on_pitch_share is not None and report.on_pitch_share < MAX_ON_PITCH_SHARE):
            report.remarks.append("notes beside the sung pitch")
        if report.left_out >= MIN_LEFT_OUT:
            report.remarks.append(f"{report.left_out} sung pitches without a note")
        if report.uncharted_s >= MIN_UNCHARTED_S:
            report.remarks.append(f"{report.uncharted_s:.1f} s of singing between the notes")
        reports.append(report)
    return reports


def _percent(share: float | None) -> str:
    return "-" if share is None else f"{100 * share:.0f} %"


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def format_report(title: str, chart: ReviewChart, lines: list[LineReport], pages: list[str]) -> str:
    """The report as Markdown."""
    scored = sum(line.notes for line in lines)
    unpitched = sum(line.unpitched for line in lines)
    out = [f"# Chart review: {title}", ""]
    out.append(f"{len(chart.notes)} notes in {len(lines)} lines; {scored} of them are scored, "
               f"{sum(line.freestyle for line in lines)} are freestyle.")
    if scored:
        out.append(f"{unpitched} scored notes ({100 * unpitched / scored:.0f} %) are sung without a pitch; "
                   f"{sum(line.left_out for line in lines)} sung pitches within notes have no note of their own.")
    standing_out = [line for line in lines if line.remarks]
    out += ["", f"## Lines that stand out ({len(standing_out)} of {len(lines)})", ""]
    out += [f"- {_clock(line.start)} \"{line.text[:60]}\": {'; '.join(line.remarks)}" for line in standing_out] or ["None."]
    out += ["", "## All lines", "",
            "| Time | Line | Notes | Freestyle | Sure pitch | On the notes | Remarks |", "|---|---|---|---|---|---|---|"]
    for line in lines:
        text = line.text[:50].replace("|", "/")
        out.append(f"| {_clock(line.start)} | {text} | {line.notes} | {line.freestyle} | "
                   f"{_percent(line.voiced_share)} | {_percent(line.on_pitch_share)} | {'; '.join(line.remarks)} |")
    if pages:
        out += ["", "## Pages", ""] + [f"- {Path(page).name}" for page in pages]
    return "\n".join(out) + "\n"


def analyse_vocals(vocals: Path, chart: ReviewChart):
    """Pitch frames of the vocal and, per note, what UltraSinger's passes find in it:
    ``(audio, times, midi, confidence, unpitched, left_out)``."""
    import librosa

    from modules.Midi.MidiSegment import MidiSegment
    from modules.Pitcher import unpitched_notes, vocal_runs
    from modules.Pitcher.Pitcher import get_pitch_with_swift_f0

    audio, _ = librosa.load(str(vocals), sr=SAMPLE_RATE, mono=True)
    pitched = get_pitch_with_swift_f0(audio, SAMPLE_RATE)
    times = np.asarray(pitched.times, dtype=float)
    hz = np.asarray(pitched.frequencies, dtype=float)
    confidence = np.asarray(pitched.confidence, dtype=float)
    midi = np.zeros(len(hz))
    midi[hz > 0] = librosa.hz_to_midi(hz[hz > 0])
    voiced = (confidence >= VOICED_CONFIDENCE) & (hz > 0)

    segments = [MidiSegment(note=librosa.midi_to_note(note.midi), start=note.start, end=note.end,
                            word=note.word, note_type=note.kind) for note in chart.notes]
    loud = unpitched_notes.loud_frames(unpitched_notes.loudness_db(audio, SAMPLE_RATE, times))
    has_pitch = voiced | (unpitched_notes.partial_prominence(audio, SAMPLE_RATE, times)
                          >= unpitched_notes.MIN_PROMINENCE_DB)
    unpitched = unpitched_notes.find_unpitched_notes(segments, times, has_pitch, loud)

    left_out = []
    for note in chart.notes:
        count = 0
        if note.kind in PITCHED_TYPES and note.end - note.start >= vocal_runs.MIN_STEP_NOTE_S:
            inside = voiced & (times >= note.start) & (times <= note.end)
            count = max(0, len(vocal_runs.find_plateaus(times[inside], midi[inside])) - 1)
        left_out.append(count)
    return audio, times, midi, confidence, unpitched, left_out


def render_pages(chart: ReviewChart, audio: np.ndarray, times: np.ndarray, midi: np.ndarray,
                 confidence: np.ndarray, unpitched: list[bool], out_dir: Path,
                 start: float, end: float, row_seconds: float = 10.0, rows: int = 3) -> list[str]:
    """Write the pages (spectrogram and piano roll of each stretch) and return their paths."""
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hop = 160
    spectrum = np.abs(librosa.stft(audio, n_fft=1024, hop_length=hop))
    level = 20 * np.log10(np.maximum(spectrum, 1e-6))
    top = float(np.percentile(level, 99.5))
    hz = np.fft.rfftfreq(1024, 1 / SAMPLE_RATE)
    band = (hz >= SPECTROGRAM_HZ[0]) & (hz <= SPECTROGRAM_HZ[1])
    frame_times = np.arange(level.shape[1]) * hop / SAMPLE_RATE
    voiced = (confidence >= VOICED_CONFIDENCE) & (midi > 0)
    unsure = (confidence >= 0.3) & (midi > 0) & ~voiced
    sung_hz = 440.0 * 2 ** ((midi - 69) / 12)
    colours = [UNPITCHED_COLOUR if flag else NOTE_COLOURS.get(note.kind, NOTE_COLOURS[":"])
               for note, flag in zip(chart.notes, unpitched)]
    breaks = [b.start for a, b in zip(chart.notes, chart.notes[1:]) if b.line != a.line]

    pages = []
    page_start = start
    while page_start < end:
        figure, axes = plt.subplots(2 * rows, 1, figsize=(17.5, 5.6 * rows), dpi=100,
                                    gridspec_kw={"height_ratios": [2, 3] * rows})
        for row in range(rows):
            a, b = page_start + row * row_seconds, page_start + (row + 1) * row_seconds
            shown = [(note, colour) for note, colour in zip(chart.notes, colours) if note.end > a and note.start < b]
            # Spectrogram with the pitch track and the notes
            axis = axes[2 * row]
            columns = (frame_times >= a) & (frame_times < b)
            if columns.any():
                axis.pcolormesh(frame_times[columns], hz[band], level[np.ix_(band, columns)], cmap="magma",
                                vmin=top - 60, vmax=top, shading="auto", rasterized=True)
            axis.set_yscale("log")
            axis.set_ylim(*SPECTROGRAM_HZ)
            axis.set_yticks([110, 220, 440, 880, 1760, 3520])
            axis.set_yticklabels(["A2", "A3", "A4", "A5", "A6", "A7"], fontsize=7)
            axis.minorticks_off()
            inside = (times >= a) & (times < b)
            axis.scatter(times[inside & unsure], sung_hz[inside & unsure], s=3, c="#66ccff", alpha=0.5, linewidths=0)
            axis.scatter(times[inside & voiced], sung_hz[inside & voiced], s=5, c="#00ffff", linewidths=0)
            for note, colour in shown:
                note_hz = 440.0 * 2 ** ((note.midi - 69) / 12)
                axis.plot([max(note.start, a), min(note.end, b)], [note_hz, note_hz], color=colour, lw=2.4,
                          solid_capstyle="butt")
            # Piano roll: the sung pitch against the notes, in the octave that is sung
            roll = axes[2 * row + 1]
            sure = inside & voiced
            centre = float(np.median(midi[sure])) if sure.sum() > 5 else 60.0
            low, high = (np.percentile(midi[sure], [2, 98]) if sure.sum() > 5 else (centre - 6, centre + 6))
            bottom, ceiling = np.floor(min(low, centre - 5) - 2.5), np.ceil(max(high, centre + 5) + 1.5)
            for pitch in range(int(bottom), int(ceiling) + 1):
                roll.axhline(pitch, color="#999999" if pitch % 12 == 0 else "#dddddd", lw=0.5, zorder=0)
            for note, colour in shown:
                sung = sure & (times >= note.start) & (times < note.end)
                target = float(np.median(midi[sung])) if sung.sum() >= 2 else centre
                pitch = note.midi + 12 * round((target - note.midi) / 12)
                roll.plot([max(note.start, a), min(note.end, b)], [pitch, pitch], color=colour, lw=5, alpha=0.75,
                          solid_capstyle="butt")
                roll.plot([note.start, note.start], [pitch - 0.45, pitch + 0.45], color=colour, lw=1.2)
                if note.start >= a:
                    roll.text(note.start, bottom + 0.2, note.word.strip() or "~", fontsize=8, rotation=55,
                              ha="left", va="bottom", clip_on=True)
            roll.scatter(times[inside & unsure], midi[inside & unsure], s=3, c="#bbbbbb", linewidths=0, zorder=3)
            roll.scatter(times[sure], midi[sure], s=6, c="#0033cc", linewidths=0, zorder=5)
            roll.set_ylim(bottom, ceiling)
            ticks = list(range(int(bottom), int(ceiling) + 1))
            roll.set_yticks(ticks)
            roll.set_yticklabels([f"{NOTE_NAMES[t % 12]}{t // 12 - 1}" for t in ticks], fontsize=6.5)
            for axis in (axis, roll):
                axis.set_xlim(a, b)
                axis.set_xticks(np.arange(np.ceil(a), b + 0.001, 1.0))
                axis.tick_params(axis="x", labelsize=7, pad=1)
                for line_break in breaks:
                    if a <= line_break < b:
                        axis.axvline(line_break, color="#808080", lw=0.8, ls="--")
        figure.subplots_adjust(left=0.03, right=0.995, top=0.99, bottom=0.03, hspace=0.18)
        path = out_dir / f"page_{int(page_start):04d}.png"
        figure.savefig(path)
        plt.close(figure)
        pages.append(str(path))
        page_start += row_seconds * rows
    return pages


def review(song: Path, out_dir: Path, vocals: Path | None = None, start: float | None = None,
           end: float | None = None, pages: bool = True) -> tuple[Path, list[LineReport]]:
    """Review the chart of *song* into *out_dir*; returns the report's path and the lines."""
    chart_path = find_chart(song)
    chart = load_chart(chart_path)
    if not chart.notes:
        raise ValueError(f"{chart_path} has no notes")
    vocals = vocals or find_vocals(chart_path, chart.headers)
    if vocals is None:
        raise FileNotFoundError(f"{chart_path} names no separated vocal (#VOCALS); give one with --vocals")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, times, midi, confidence, unpitched, left_out = analyse_vocals(vocals, chart)
    voiced = (confidence >= VOICED_CONFIDENCE) & (midi > 0)
    lines = review_lines(chart, times, midi, voiced, unpitched, left_out)
    written = []
    if pages:
        first = max(0.0, chart.notes[0].start - 1.0) if start is None else start
        last = chart.notes[-1].end + 1.0 if end is None else end
        written = render_pages(chart, audio, times, midi, confidence, unpitched, out_dir, first, last)
    report = out_dir / "review.md"
    title = " - ".join(filter(None, (chart.headers.get("ARTIST"), chart.headers.get("TITLE")))) or chart_path.stem
    report.write_text(format_report(title, chart, lines, written), encoding="utf-8")
    return report, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lay a chart over the separated vocal of its song: "
                                                 "pages to look at and a report of the lines that stand out.")
    parser.add_argument("song", type=Path, help="song folder or UltraStar TXT")
    parser.add_argument("out_dir", type=Path, help="folder for the pages and the report")
    parser.add_argument("--vocals", type=Path, help="separated vocal (default: the chart's #VOCALS file)")
    parser.add_argument("--from", dest="start", type=float, help="first second on the pages")
    parser.add_argument("--to", dest="end", type=float, help="last second on the pages")
    parser.add_argument("--no-pages", action="store_true", help="only write the report")
    args = parser.parse_args(argv)
    try:
        report, lines = review(args.song, args.out_dir, args.vocals, args.start, args.end, not args.no_pages)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    print(f"{sum(1 for line in lines if line.remarks)} of {len(lines)} lines stand out - see {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
