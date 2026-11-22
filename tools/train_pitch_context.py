"""Train a pitch context model on your own UltraStar song library.

The model moves single notes of a generated chart by up to two semitones,
from the pitch that was sung to the pitch a hand-made chart would write
(``--pitch_context`` with ``--pitch_context_model``, or in the GUI; see
src/modules/Pitcher/pitch_context.py). It learns this from the hand-made
charts of your library: every charted note gets the pitch its pitch track
gives it, the way the conversion pitches a note, and the model learns from
the note's pitch frames, its neighbours and the song's key by how many
semitones the charted pitch lies beside that one.

It trains on the examples that ``tools/train_segmentation.py extract`` makes
(the pitch track of the separated vocal and the hand-made chart per song),
so no audio is needed and no GPU: a few minutes on a CPU for a thousand
songs. The model file contains information derived from your library: keep
it outside this repository and do not publish it.

Workflow::

    # 1. once: separate the vocals and extract the examples (GPU), if not done
    #    already for the segmentation model
    python tools/train_segmentation.py extract D:/Songs D:/SegTrain

    # 2. train; a tenth of the songs is held out to show what the model gives
    python tools/train_pitch_context.py --data D:/SegTrain --out D:/SegTrain/pitch_context.npz

    # 3. use it
    python src/UltraSinger.py -i song.mp3 --segmentation_model D:/SegTrain/model.pt \\
        --pitch_context balanced --pitch_context_model D:/SegTrain/pitch_context.npz

Songs of a chart benchmark sample must not be trained on (``extract
--exclude`` of the segmentation tool, or ``--exclude`` here). Training needs
scikit-learn, which is installed with UltraSinger. See docs/pitch-context-model.md.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from modules.Pitcher import pitch_context  # noqa: E402
from modules.Pitcher.pitch_context import (  # noqa: E402
    N_OUTPUTS, SHIFTS, STRENGTHS, PitchContextModel, PitchTrack, choose_shifts, note_features, track_pitch,
)

FRAME_MS = 10  # the grid the charts are compared on (as tools/chart_benchmark.py)
# Note kinds of an example that are sung on a pitch: ":" and "*" (KIND_CODES of train_segmentation.py)
PITCHED_KIND_CODES = (0, 1)
MIN_NOTES = 50  # pitched notes a song needs to be learned from
_KEEP = SHIFTS.index(0)


def require_outside_repo(path: Path, what: str) -> Path:
    """Refuse locations inside the repository (library-derived data must stay private)."""
    try:
        path.resolve().relative_to(REPO.resolve())
    except ValueError:
        return path
    raise SystemExit(f"the {what} must be outside the repository (it holds data derived from your library)")


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

def fold(semitones):
    """A difference in semitones as the games hear it: without octaves (-6 ... 6)."""
    return (np.asarray(semitones, dtype=float) + 6.0) % 12.0 - 6.0


def reference_grid(starts_ms, ends_ms, pitches, n_frames: int) -> np.ndarray:
    """The charted pitch of each frame of a chart's pitched notes (NaN: no note)."""
    grid = np.full(n_frames, np.nan)
    for start, end, pitch in zip(starts_ms, ends_ms, pitches):
        a = max(int(start // FRAME_MS), 0)
        b = min(max(int(end // FRAME_MS), 0), n_frames)
        if b > a:
            grid[a:b] = pitch
    return grid


def shift_targets(starts_ms, ends_ms, pitches, grids: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What each shift of each note is worth, measured on a hand-made chart.

    For every note and every shift of ``SHIFTS``: the share of the note's time
    on which the shifted pitch is within one semitone of the charted pitch,
    and the share on which it is exactly on it (octaves ignored). ``grids``
    are the frames of the chart (see ``reference_grid``), one per voice; a
    shifted pitch agrees when it agrees with any of them. Returns both shares
    (notes x shifts) and, per note, whether the chart has a note under it.
    """
    n = len(pitches)
    within_one = np.zeros((n, len(SHIFTS)))
    exact = np.zeros((n, len(SHIFTS)))
    charted = np.zeros(n, dtype=bool)
    n_frames = min((len(g) for g in grids), default=0)
    for i in range(n):
        a = max(int(starts_ms[i] // FRAME_MS), 0)
        b = min(int(ends_ms[i] // FRAME_MS), n_frames)
        if b <= a:
            continue
        for k, shift in enumerate(SHIFTS):
            near = np.zeros(b - a, dtype=bool)
            on = np.zeros(b - a, dtype=bool)
            for grid in grids:
                under = grid[a:b]
                there = ~np.isnan(under)
                charted[i] |= bool(there.any())
                away = np.abs(fold(pitches[i] + shift - under[there]))
                near[there] |= away <= 1
                on[there] |= away == 0
            within_one[i, k] = near.mean()
            exact[i, k] = on.mean()
    return within_one, exact, charted


@dataclass
class SongRows:
    """The notes of one song to learn from: features, weights (note time) and targets."""
    features: np.ndarray
    weights: np.ndarray
    within_one: np.ndarray
    exact: np.ndarray


def load_example(path: Path) -> dict:
    """The number arrays of a training example that this tool needs.

    The examples also hold the words of the chart as Python objects; those are
    not read (nothing is unpickled here).
    """
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("f0_t", "f0_hz", "f0_conf", "notes", "ref_fit", "duration")}


def example_rows(example: dict, min_notes: int = MIN_NOTES) -> SongRows | None:
    """The training rows of one example; None for a song with too few usable notes.

    Every pitched note of the hand-made chart becomes a note with the pitch
    its pitch frames give it (as the conversion pitches the notes of the
    segmentation model); notes without enough confident frames are left out.
    """
    notes = np.asarray(example["notes"], dtype=float).reshape(-1, 4)
    duration_ms = float(example["duration"]) * 1000.0
    pitched = np.isin(notes[:, 3].astype(int), PITCHED_KIND_CODES) & (notes[:, 0] < duration_ms)
    notes = notes[pitched]
    notes = notes[np.argsort(notes[:, 0], kind="stable")]
    if len(notes) < min_notes:
        return None
    track = PitchTrack(example["f0_t"], example["f0_hz"], example["f0_conf"])
    sung = [track_pitch(track.frames(start, end)) for start, end in notes[:, :2]]
    kept = np.array([pitch is not None for pitch in sung])
    if kept.sum() < min_notes:
        return None
    starts, ends = notes[kept, 0], notes[kept, 1]
    pitches = np.array([pitch for pitch in sung if pitch is not None])
    features, _ = note_features(starts, ends, pitches, track)
    n_frames = int((float(notes[:, 1].max()) + 1000.0) // FRAME_MS) + 1
    grid = reference_grid(notes[:, 0], notes[:, 1], notes[:, 2], n_frames)
    within_one, exact, charted = shift_targets(starts, ends, pitches, [grid])
    return SongRows(features[charted], (ends - starts)[charted], within_one[charted], exact[charted])


def find_examples(data: Path) -> list[Path]:
    """The example files of ``data``: the work folder of train_segmentation.py or its data folder."""
    files = sorted(data.glob("*.npz"))
    if not files and (data / "data").is_dir():
        files = sorted((data / "data").glob("*.npz"))
    return files


def excluded_ids(values: list[str] | None) -> set[str]:
    """The example names of ``--exclude``: names, or JSON files that list names."""
    names: set[str] = set()
    for value in values or []:
        if value.lower().endswith(".json"):
            listed = json.loads(Path(value).read_text(encoding="utf-8"))
            if not isinstance(listed, list) or not all(isinstance(name, str) for name in listed):
                raise SystemExit(f"{value}: expected a JSON list of example names")
            names.update(listed)
        else:
            names.add(value)
    return {Path(name).stem if name.lower().endswith(".npz") else name for name in names}


def load_songs(files: list[Path], min_ref_fit: float) -> dict[str, SongRows]:
    """The usable songs of ``files`` by name; says how many were left out and why."""
    songs: dict[str, SongRows] = {}
    poor_fit = too_few = broken = 0
    for i, path in enumerate(files, 1):
        if i % 200 == 0:
            print(f"[{i}/{len(files)}] reading the examples", flush=True)
        try:
            example = load_example(path)
            if float(example["ref_fit"]) < min_ref_fit:
                poor_fit += 1
                continue
            rows = example_rows(example)
        except Exception as e:  # noqa: BLE001 - one broken example must not stop the run
            print(f"{path.name}: skipped ({e!r})", flush=True)
            broken += 1
            continue
        if rows is None or not len(rows.weights):
            too_few += 1
            continue
        songs[path.stem] = rows
    left_out = [f"{poor_fit} whose chart fits the vocal worse than {min_ref_fit:.0%}" if poor_fit else "",
                f"{too_few} with fewer than {MIN_NOTES} usable notes" if too_few else "",
                f"{broken} unreadable" if broken else ""]
    left_out = ", ".join(text for text in left_out if text)
    print(f"{len(songs)} usable songs of {len(files)}" + (f" (left out: {left_out})" if left_out else ""), flush=True)
    return songs


def stacked(songs: list[SongRows]) -> SongRows:
    return SongRows(np.vstack([s.features for s in songs]), np.concatenate([s.weights for s in songs]),
                    np.vstack([s.within_one for s in songs]), np.vstack([s.exact for s in songs]))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def export_trees(regressors: list, check_features: np.ndarray) -> PitchContextModel:
    """The trees of the trained regressors (one per model output) as plain arrays.

    The arrays are read from scikit-learn's own tree structures; the exported
    model must then predict what scikit-learn predicts, or the export fails
    (a later scikit-learn may keep its trees in another way).
    """
    if len(regressors) != N_OUTPUTS:
        raise ValueError(f"expected {N_OUTPUTS} regressors")
    parts = {name: [] for name in ("feature", "threshold", "left", "right", "missing_left", "value")}
    roots, outputs, baseline, total = [], [], [], 0
    for output, regressor in enumerate(regressors):
        baseline.append(float(np.ravel(regressor._baseline_prediction)[0]))
        for iteration in regressor._predictors:
            for predictor in iteration:
                nodes = predictor.nodes
                leaf = nodes["is_leaf"].astype(bool)
                if nodes["is_categorical"][~leaf].any():
                    raise ValueError("categorical splits are not supported")
                parts["feature"].append(np.where(leaf, -1, nodes["feature_idx"].astype(np.int64)))
                parts["threshold"].append(np.where(leaf, 0.0, nodes["num_threshold"]))
                parts["left"].append(np.where(leaf, 0, nodes["left"].astype(np.int64) + total))
                parts["right"].append(np.where(leaf, 0, nodes["right"].astype(np.int64) + total))
                parts["missing_left"].append(nodes["missing_go_to_left"].astype(bool) & ~leaf)
                parts["value"].append(np.where(leaf, nodes["value"], 0.0))
                roots.append(total)
                outputs.append(output)
                total += len(nodes)
    model = PitchContextModel(**{name: np.concatenate(arrays) for name, arrays in parts.items()},
                              tree_root=np.array(roots), tree_output=np.array(outputs),
                              baseline=np.array(baseline))
    sample = check_features[:2000]
    exported = np.hstack(model.predict(sample))
    trained = np.column_stack([regressor.predict(sample) for regressor in regressors])
    if not np.allclose(exported, trained, rtol=0, atol=1e-8):
        raise RuntimeError("the exported trees do not predict what scikit-learn predicts "
                           f"(largest difference {np.abs(exported - trained).max():.3g}); "
                           "this version of scikit-learn is not supported")
    return model


def train_model(rows: SongRows, trees: int = 300, seed: int = 0, log=print) -> PitchContextModel:
    """Fit one regressor per model output on ``rows`` and export their trees."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    targets = np.hstack([rows.within_one, rows.exact])
    regressors = []
    for output in range(N_OUTPUTS):
        regressor = HistGradientBoostingRegressor(max_iter=trees, learning_rate=0.08, max_depth=7,
                                                  l2_regularization=1.0, random_state=seed)
        regressor.fit(rows.features, targets[:, output], sample_weight=rows.weights)
        regressors.append(regressor)
        log(f"  output {output + 1}/{N_OUTPUTS} trained ({regressor.n_iter_} trees)")
    return export_trees(regressors, rows.features)


def evaluate(model: PitchContextModel, rows: SongRows) -> dict[str, dict[str, float]]:
    """Per strength, on ``rows``: the share of the note time exactly on the chart
    and within one semitone of it after the model's shifts, and the share of
    notes moved ("off": the notes as they are)."""
    predicted = model.predict(rows.features)
    index = np.arange(len(rows.weights))
    result = {}
    for strength in ("off", *STRENGTHS):
        shifts = np.zeros(len(index), dtype=int) if strength == "off" else choose_shifts(*predicted, strength)
        column = shifts + _KEEP  # SHIFTS are consecutive
        result[strength] = {
            "exact_pct": 100.0 * float(np.average(rows.exact[index, column], weights=rows.weights)),
            "within_one_pct": 100.0 * float(np.average(rows.within_one[index, column], weights=rows.weights)),
            "moved_pct": 100.0 * float(np.mean(shifts != 0)),
        }
    return result


def format_evaluation(result: dict[str, dict[str, float]]) -> str:
    lines = [f"  {'':10s} {'exact pitch':>12s} {'within one semitone':>20s} {'notes moved':>12s}"]
    for strength, row in result.items():
        name = "unchanged" if strength == "off" else strength
        lines.append(f"  {name:10s} {row['exact_pct']:10.1f} % {row['within_one_pct']:18.1f} % "
                     f"{row['moved_pct']:10.1f} %")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True,
                   help="folder of the training examples: the work folder of 'train_segmentation.py extract' "
                        "or its data folder")
    p.add_argument("--out", required=True, help="model file to write (outside the repository), e.g. pitch_context.npz")
    p.add_argument("--exclude", nargs="*", metavar="NAME",
                   help="examples to leave out (e.g. songs of a chart benchmark): their names, "
                        "or JSON files that list names")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="share of the songs held out to measure the model (0: no measurement); "
                        "the model that is written is trained on all songs")
    p.add_argument("--min-ref-fit", type=float, default=0.6,
                   help="skip songs whose chart fits the vocal worse than this (0-1)")
    p.add_argument("--trees", type=int, default=300, help="trees per model output at most")
    p.add_argument("--limit", type=int, default=0, help="only use the first N examples (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = Path(args.data)
    out = require_outside_repo(Path(args.out), "model file")
    if not data.is_dir():
        raise SystemExit(f"data folder not found: {data}")
    most_trees = pitch_context._MAX_TREES // N_OUTPUTS  # (a model file with more trees is refused)
    if not 1 <= args.trees <= most_trees:
        raise SystemExit(f"--trees must be at least 1 and at most {most_trees}")
    if not 0.0 <= args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be at least 0 and below 1")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise SystemExit("training needs scikit-learn (it is installed with UltraSinger: run this tool "
                         "with UltraSinger's Python)") from None

    files = find_examples(data)
    excluded = excluded_ids(args.exclude)
    if excluded:
        kept = [f for f in files if f.stem not in excluded]
        print(f"{len(files) - len(kept)} of {len(files)} examples left out (--exclude)", flush=True)
        files = kept
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no training examples (*.npz) in {data} - run 'train_segmentation.py extract' first")
    started = time.time()
    songs = load_songs(files, args.min_ref_fit)
    if len(songs) < 2:
        raise SystemExit(f"only {len(songs)} usable songs in {data}")
    names = sorted(songs)
    everything = stacked([songs[name] for name in names])
    already = evaluate_unchanged(everything)
    print(f"{len(everything.weights)} notes; before any correction {already['exact_pct']:.1f} % of their time "
          f"is exactly on the charted pitch, {already['within_one_pct']:.1f} % within one semitone", flush=True)

    info = {"songs": len(names), "notes": len(everything.weights), "trees": args.trees, "seed": args.seed,
            "exact_pct": round(already["exact_pct"], 2), "within_one_pct": round(already["within_one_pct"], 2)}
    n_val = min(len(names) - 1, round(len(names) * args.val_fraction)) if args.val_fraction > 0 else 0
    if n_val > 0:
        held_out = set(random.Random(args.seed).sample(names, n_val))
        print(f"measuring: training on {len(names) - n_val} songs, {n_val} held out", flush=True)
        probe = train_model(stacked([songs[n] for n in names if n not in held_out]), args.trees, args.seed)
        result = evaluate(probe, stacked([songs[n] for n in names if n in held_out]))
        print(f"held-out songs ({n_val}), share of the note time on the charted pitch:\n"
              f"{format_evaluation(result)}", flush=True)
        info["held_out"] = {"songs": n_val, **{k: {m: round(v, 2) for m, v in row.items()}
                                               for k, row in result.items()}}
    print(f"training on all {len(names)} songs", flush=True)
    model = train_model(everything, args.trees, args.seed)
    model.info = info
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"model written to {out} ({out.stat().st_size / 1e6:.1f} MB, {len(model.tree_root)} trees, "
          f"{time.time() - started:.0f} s)")
    return 0


def evaluate_unchanged(rows: SongRows) -> dict[str, float]:
    """Share of the note time exactly on / within one semitone of the chart, notes as they are."""
    return {"exact_pct": 100.0 * float(np.average(rows.exact[:, _KEEP], weights=rows.weights)),
            "within_one_pct": 100.0 * float(np.average(rows.within_one[:, _KEEP], weights=rows.weights))}


if __name__ == "__main__":
    raise SystemExit(main())
