"""Train a note segmentation model on your own UltraStar song library.

The model learns, frame by frame on the separated vocal, where notes start and
which passages are charted at all, from existing hand-made charts. Use the
resulting file with ``--segmentation_model`` (or in the GUI). Everything this
tool writes (extracted data, model files) contains information derived from
your library: keep it outside this repository and do not publish it.

Workflow::

    # 1. separate vocals and extract features + reference notes (GPU, ~20-40 s per song)
    python tools/train_segmentation.py extract D:/Songs D:/SegTrain

    # 2. train and tune the decoding thresholds on a held-out part of the data
    python tools/train_segmentation.py train D:/SegTrain --out D:/SegTrain/model.pt

    # 3. measure it end to end with the chart benchmark (tools/chart_benchmark.py)
    python tools/chart_benchmark.py convert D:/ChartBench --label model --args "--segmentation_model D:/SegTrain/model.pt"

Use ``extract --exclude D:/ChartBench/songs.json`` so that songs of a chart
benchmark sample are never trained on - otherwise the benchmark result is
meaningless. See docs/segmentation-model.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chart_benchmark as cb  # noqa: E402
from modules.Segmentation.features import FRAME_S, VocalAnalysis, analyse_vocal, model_input  # noqa: E402

KIND_CODES = {":": 0, "*": 1, "F": 2, "R": 3, "G": 4}
KIND_NAMES = {v: k for k, v in KIND_CODES.items()}
CHUNK_FRAMES = 625  # 10 s training windows
DECODE_GRID = {"onset_thr": [0.3, 0.4, 0.5, 0.6], "act_thr": [0.4, 0.5, 0.6],
               "min_note_frames": [3, 4, 6], "min_gap_frames": [0, 1, 2]}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def require_outside_repo(path: Path, what: str) -> Path:
    """Refuse locations inside the repository (library-derived data must stay private)."""
    try:
        path.resolve().relative_to(REPO.resolve())
    except ValueError:
        return path
    raise SystemExit(f"the {what} must be outside the repository (it holds data derived from your library)")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def save_example(path: Path, analysis: VocalAnalysis, notes: list, offset_ms: float, ref_fit: float) -> None:
    """One training example: vocal analysis + time-aligned reference notes.

    Written to a temporary file first: a resumed extraction skips existing
    examples, so an interrupted write must never leave a truncated one behind.
    """
    arr = np.array([[n.start_ms + offset_ms, n.end_ms + offset_ms, n.midi, KIND_CODES.get(n.kind, 0)]
                    for n in notes], dtype=np.float32).reshape(-1, 4)
    tmp = path.with_name(path.name + ".part")
    try:
        with open(tmp, "wb") as f:  # a file object, so numpy does not append ".npz" to the name
            np.savez_compressed(f, f0_t=analysis.f0_t, f0_hz=analysis.f0_hz, f0_conf=analysis.f0_conf,
                                logmel=analysis.logmel, rms=analysis.rms, notes=arr,
                                words=np.array([n.word for n in notes], dtype=object),
                                offset_ms=offset_ms, ref_fit=ref_fit, duration=analysis.duration)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_example(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def example_analysis(d: dict) -> VocalAnalysis:
    return VocalAnalysis(d["f0_t"], d["f0_hz"], d["f0_conf"], d["logmel"], d["rms"], float(d["duration"]))


def example_reference(d: dict) -> list:
    words = d["words"]
    return [cb.ChartNote(float(s), float(e), int(m), KIND_NAMES.get(int(k), ":"),
                         str(words[i]) if i < len(words) else "")
            for i, (s, e, m, k) in enumerate(d["notes"])]


def sung_from_analysis(a: VocalAnalysis) -> cb.SungPitch:
    keep = (a.f0_conf >= cb.CONFIDENCE) & (a.f0_hz > 40)
    return cb.SungPitch(a.f0_t[keep].astype(float), 69 + 12 * np.log2(a.f0_hz[keep].astype(float) / 440.0))


def cmd_extract(args) -> int:
    library = Path(args.library)
    workdir = require_outside_repo(Path(args.workdir), "workdir")
    if not library.is_dir():
        raise SystemExit(f"library folder not found: {library}")
    data = workdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    songs_path = workdir / "songs.json"
    excluded = set()
    for f in args.exclude or []:
        excluded |= {s["folder"] for s in json.loads(Path(f).read_text(encoding="utf-8"))}
    if songs_path.exists():
        # Resuming: apply --exclude here too, and drop examples already extracted
        # for excluded songs - otherwise a benchmark could silently be trained on.
        songs = json.loads(songs_path.read_text(encoding="utf-8"))
        dropped = [s for s in songs if s["folder"] in excluded]
        if dropped:
            for s in dropped:
                (data / f"{s['id']}.npz").unlink(missing_ok=True)
            songs = [s for s in songs if s["folder"] not in excluded]
            songs_path.write_text(json.dumps(songs, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"removed {len(dropped)} excluded songs (and their extracted data) from {songs_path}",
                  flush=True)
    else:
        cands = [c for c in cb.find_candidates(library) if c["folder"] not in excluded]
        songs = [{"id": f"tr_{i:05d}", **c} for i, c in enumerate(cands)]
        songs_path.write_text(json.dumps(songs, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{len(songs)} training songs ({len(excluded)} excluded folders)", flush=True)
    if args.limit:
        songs = songs[:args.limit]

    from audio_separator.separator import Separator
    from modules.Audio.separation import DEFAULT_AUDIO_SEPARATOR_MODEL
    from modules.Segmentation.features import load_vocal

    tmp = Path(tempfile.mkdtemp(prefix="segtrain_"))
    try:
        sep = Separator(output_dir=str(tmp), output_format="WAV", sample_rate=44100,
                        normalization_threshold=0.9, log_level=40)
        sep.load_model(model_filename=DEFAULT_AUDIO_SEPARATOR_MODEL.value)
        done, t_start, total = 0, time.time(), len(songs)
        already = sum(1 for s in songs if (data / f"{s['id']}.npz").exists())
        print(f"{total} songs, {already} already extracted", flush=True)
        # "[i/n]" prefixes are parsed by the GUI's training page for its progress bar
        for i, s in enumerate(songs, 1):
            out = data / f"{s['id']}.npz"
            if out.exists():
                continue
            try:
                for p in tmp.iterdir():
                    p.unlink()
                sep.separate(str(s.get("audio") or s["media"]),
                             custom_output_names={"Vocals": "vocals", "Instrumental": "no_vocals"})
                analysis = analyse_vocal(load_vocal(str(next(tmp.glob("vocals*.wav")))))
                ref = cb.load_chart(s["txt"])
                offset, fit = cb.fit_reference_offset(ref, sung_from_analysis(analysis))
                save_example(out, analysis, ref, offset, fit)
                done += 1
                print(f"[{i}/{total}] {s['id']}: ok fit={fit:.2f} offset={offset:+.0f} ms "
                      f"(avg {(time.time() - t_start) / done:.1f} s/song)", flush=True)
            except Exception as e:  # noqa: BLE001 - one broken song must not stop the run
                print(f"[{i}/{total}] {s['id']}: ERROR {e!r}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def frame_labels(d: dict, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame class (0 none / 1 pitched / 2 freestyle-rap) and onset target (+-1 frame soft)."""
    cls = np.zeros(n_frames, np.int64)
    onset = np.zeros(n_frames, np.float32)
    for s, e, _midi, kind in d["notes"]:
        a = max(int(round(s / 1000 / FRAME_S)), 0)
        b = min(int(round(e / 1000 / FRAME_S)), n_frames)
        if b <= a:
            continue
        if int(kind) in (2, 3, 4):
            cls[a:b] = 2
        else:
            cls[a:b] = 1
            onset[a] = 1.0
    soft = onset.copy()
    soft[1:] = np.maximum(soft[1:], 0.5 * onset[:-1])
    soft[:-1] = np.maximum(soft[:-1], 0.5 * onset[1:])
    return cls, soft


def load_dataset(workdir: Path, min_fit: float) -> list[tuple]:
    """Usable examples; only songs still listed in songs.json (if present) are used."""
    listed = None
    songs_path = workdir / "songs.json"
    if songs_path.exists():
        listed = {s["id"] for s in json.loads(songs_path.read_text(encoding="utf-8"))}
    items = []
    for p in sorted((workdir / "data").glob("*.npz")):
        if listed is not None and p.stem not in listed:
            continue
        d = load_example(p)
        if float(d["ref_fit"]) < min_fit:
            continue
        a = example_analysis(d)
        x = model_input(a)
        cls, onset = frame_labels(d, len(x))
        items.append((p.stem, x, cls, onset, a, example_reference(d)))
    return items


def _batches(items, batch_size, rng):
    while True:
        xs, cs, os_ = [], [], []
        for _ in range(batch_size):
            _, x, c, o, _, _ = items[rng.randrange(len(items))]
            a = rng.randrange(0, len(x) - CHUNK_FRAMES) if len(x) > CHUNK_FRAMES else 0
            x, c, o = x[a:a + CHUNK_FRAMES], c[a:a + CHUNK_FRAMES], o[a:a + CHUNK_FRAMES]
            pad = CHUNK_FRAMES - len(x)
            if pad:
                x = np.pad(x, ((0, pad), (0, 0)))
                c = np.pad(c, (0, pad))
                o = np.pad(o, (0, pad))
            xs.append(x); cs.append(c); os_.append(o)
        yield np.stack(xs), np.stack(cs), np.stack(os_)


def evaluate_split(model, items, device, decode_cfg) -> dict:
    """Median chart agreement / onset hit rate / note ratio of decoded notes on ``items``."""
    from modules.Segmentation.decode import decode_notes
    from modules.Segmentation.model import predict

    agree, onset, ratio = [], [], []
    for _, x, _, _, a, ref in items:
        probs, on = predict(model, x, device)
        notes = decode_notes(probs, on, a, **decode_cfg)
        gen = [cb.ChartNote(n.start * 1000, n.end * 1000, n.midi, "F" if n.freestyle else ":") for n in notes]
        m = cb.chart_metrics(ref, gen, sung_from_analysis(a))
        if m["chart_agreement_pct"] is not None:
            agree.append(m["chart_agreement_pct"])
            onset.append(m["onset_hit_100_pct"] or 0.0)
            ratio.append(m["note_count_ratio"] or 0.0)
    if not agree:
        return {"agreement": 0.0, "onset100": 0.0, "note_ratio": 0.0}
    return {"agreement": statistics.median(agree), "onset100": statistics.median(onset),
            "note_ratio": statistics.median(ratio)}


def tune_decoding(model, items, device) -> tuple[dict, dict]:
    """Grid-search decoding thresholds on the validation songs (predictions computed once)."""
    from modules.Segmentation.decode import decode_notes
    from modules.Segmentation.model import predict

    cached = [(predict(model, x, device), a, ref, sung_from_analysis(a)) for _, x, _, _, a, ref in items]
    best = (-1.0, None)
    for vals in itertools.product(*DECODE_GRID.values()):
        cfg = dict(zip(DECODE_GRID, vals))
        agree = []
        for (probs, on), a, ref, sung in cached:
            notes = decode_notes(probs, on, a, **cfg)
            gen = [cb.ChartNote(n.start * 1000, n.end * 1000, n.midi, "F" if n.freestyle else ":") for n in notes]
            m = cb.chart_metrics(ref, gen, sung)
            if m["chart_agreement_pct"] is not None:
                agree.append(m["chart_agreement_pct"])
        score = statistics.median(agree) if agree else 0.0
        if score > best[0]:
            best = (score, cfg)
    return best[1], {"agreement": best[0]}


def cmd_train(args) -> int:
    import torch
    import torch.nn.functional as F

    from modules.Segmentation.model import DEFAULT_DECODE, SegNet, save_model

    workdir = require_outside_repo(Path(args.workdir), "workdir")
    out = require_outside_repo(Path(args.out), "model file")
    if args.epochs < 1 or args.steps < 1:
        raise SystemExit("--epochs and --steps must be at least 1")
    device ="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    items = load_dataset(workdir, args.min_ref_fit)
    if len(items) < 4:
        raise SystemExit(f"only {len(items)} usable songs in {workdir / 'data'} - run 'extract' first")
    ids = sorted(i[0] for i in items)
    n_val = max(1, min(len(ids) // 2, round(len(ids) * args.val_fraction)))
    val_ids = set(random.Random(args.seed).sample(ids, n_val))
    train = [i for i in items if i[0] not in val_ids]
    val = [i for i in items if i[0] in val_ids]
    print(f"{len(train)} training / {len(val)} validation songs on {device}", flush=True)

    torch.manual_seed(args.seed)
    model = SegNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=args.epochs * args.steps)
    class_weights = torch.tensor([1.0, 1.0, 2.0], device=device)
    pos_weight = torch.tensor(8.0, device=device)
    batches = _batches(train, args.batch_size, random.Random(args.seed))
    best_state, best_agree = None, -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for _ in range(args.steps):
            x, c, o = (torch.from_numpy(t).to(device) for t in next(batches))
            logits, onset_logits = model(x)
            loss = (F.cross_entropy(logits.reshape(-1, 3), c.reshape(-1), weight=class_weights)
                    + 3.0 * F.binary_cross_entropy_with_logits(onset_logits, o, pos_weight=pos_weight))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        res = evaluate_split(model, val, device, DEFAULT_DECODE)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / args.steps:.3f} "
              f"validation agreement {res['agreement']:.1f} onset100 {res['onset100']:.1f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        if res["agreement"] > best_agree:
            best_agree = res["agreement"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    decode_cfg, tuned = tune_decoding(model, val, device)
    print(f"tuned decoding {decode_cfg}: validation agreement {tuned['agreement']:.1f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_model(out, model.cpu(), decode_cfg, {"train_songs": len(train), "val_songs": len(val),
                                             "val_agreement": tuned["agreement"]})
    print(f"model written to {out}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="separate vocals and extract training examples from a library")
    e.add_argument("library")
    e.add_argument("workdir")
    e.add_argument("--exclude", nargs="*", help="songs.json files (e.g. of a chart benchmark) whose songs are skipped")
    e.add_argument("--limit", type=int, default=0, help="only process the first N songs (0 = all)")
    e.set_defaults(func=cmd_extract)

    t = sub.add_parser("train", help="train a model on the extracted examples")
    t.add_argument("workdir")
    t.add_argument("--out", required=True, help="model file to write (outside the repository)")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--steps", type=int, default=400, help="training steps per epoch")
    t.add_argument("--batch-size", type=int, default=24)
    t.add_argument("--val-fraction", type=float, default=0.05)
    t.add_argument("--min-ref-fit", type=float, default=0.5,
                   help="skip songs whose reference chart fits the vocal worse than this (0-1)")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is available")
    t.set_defaults(func=cmd_train)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
