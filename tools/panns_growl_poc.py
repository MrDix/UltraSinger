"""POC: Compare Tier 1+2 growl detection with PANNs Tier 3 (AudioSet CNN).

This tool runs both detection approaches on a vocals audio file and
produces a side-by-side comparison for manual evaluation.

Requirements (install once):
    pip install panns-inference torch torchlibrosa

Usage:
    python tools/panns_growl_poc.py <vocals.ogg> [--ultrastar <song.txt>]
    python tools/panns_growl_poc.py <vocals.ogg> --csv out.csv

The --ultrastar flag loads UltraStar note timing to compare segment-level
Tier 1+2 results vs. PANNs frame-level predictions aligned to notes.
Without it, only PANNs frame-level analysis runs.

Example with test songs:
    python tools/panns_growl_poc.py ^
        "D:/UltraStar/Songs_UltraSinger/2026-03-G/May the Silence Fail - Come Alive/May the Silence Fail - Come Alive [Vocals].ogg" ^
        --ultrastar "D:/UltraStar/Songs_UltraSinger/2026-03-G/May the Silence Fail - Come Alive/May the Silence Fail - Come Alive.txt"
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

# ---------------------------------------------------------------------------
# PANNs Tier 3 - AudioSet sound event detection
# ---------------------------------------------------------------------------

# Relevant AudioSet class labels for vocal unpitchability
PANNS_VOCAL_CLASSES = {
    "Screaming": "scream",
    "Growling": "growl",
    "Shout": "shout",
    "Speech": "speech",
    "Rap": "rap",
    "Singing": "singing",
    "Yell": "yell",
    "Battle cry": "battle_cry",
    "Grunt": "grunt",
    "Whispering": "whisper",
}

# Classes that indicate NON-pitched/unpitchable vocals
UNPITCHABLE_CLASSES = {"Screaming", "Growling", "Shout", "Yell", "Battle cry", "Grunt", "Rap"}

# Classes that indicate normal pitched singing
PITCHED_CLASSES = {"Singing"}


@dataclass
class PANNsFrameResult:
    """Frame-level PANNs detection result."""
    time: float
    top_classes: list  # [(label, score), ...]
    is_unpitchable: bool
    unpitchable_score: float  # max score across unpitchable classes
    singing_score: float


@dataclass
class SegmentComparison:
    """Side-by-side comparison of Tier 1+2 vs Tier 3 for one segment."""
    word: str
    start: float
    end: float
    # Tier 1+2
    tier12_is_growl: bool
    tier12_median_conf: float
    tier12_pitch_stdev: float
    tier12_spectral_flat: float
    tier12_voiced_ratio: float
    # Tier 3 (PANNs)
    tier3_unpitchable_score: float
    tier3_singing_score: float
    tier3_top_class: str
    tier3_is_unpitchable: bool
    # Agreement
    agree: bool


def run_panns_sed(audio_path: str, device: str = "cpu") -> tuple:
    """Run PANNs SoundEventDetection and return frame-level results.

    Returns:
        (frame_times, framewise_output, label_list)
        framewise_output shape: (time_steps, 527)
    """
    try:
        from panns_inference import SoundEventDetection, labels
    except ImportError:
        print("ERROR: panns-inference not installed.")
        print("  Install with: pip install panns-inference torch torchlibrosa")
        sys.exit(1)

    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=32000, mono=True)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f}s, samples: {len(audio)}")

    audio_batch = audio[None, :]  # (1, num_samples)

    print(f"Running PANNs SoundEventDetection on {device}...")
    t0 = time.time()
    sed = SoundEventDetection(checkpoint_path=None, device=device)
    framewise_output = sed.inference(audio_batch)  # (1, time_steps, 527)
    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s")

    framewise = framewise_output[0]  # (time_steps, 527)
    n_frames = framewise.shape[0]

    # PANNs uses ~0.32s per frame (32ms hop * 10 pooling)
    frame_duration = duration / n_frames
    frame_times = np.arange(n_frames) * frame_duration

    print(f"  {n_frames} frames, {frame_duration:.3f}s per frame")

    return frame_times, framewise, labels


def analyze_panns_frames(
    frame_times: np.ndarray,
    framewise: np.ndarray,
    labels: list,
    threshold: float = 0.3,
) -> list[PANNsFrameResult]:
    """Analyze PANNs framewise output for vocal classes."""
    # Build index mapping for relevant classes
    class_indices = {}
    for label in PANNS_VOCAL_CLASSES:
        try:
            idx = labels.index(label)
            class_indices[label] = idx
        except ValueError:
            print(f"  Warning: '{label}' not found in AudioSet labels")

    results = []
    for i, t in enumerate(frame_times):
        frame = framewise[i]

        # Get scores for relevant classes
        class_scores = {}
        for label, idx in class_indices.items():
            class_scores[label] = float(frame[idx])

        # Sort by score
        top = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)

        # Compute unpitchable score (max across unpitchable classes)
        unpitchable_score = max(
            (class_scores.get(c, 0.0) for c in UNPITCHABLE_CLASSES), default=0.0
        )
        singing_score = class_scores.get("Singing", 0.0)

        is_unpitchable = unpitchable_score > threshold and unpitchable_score > singing_score

        results.append(PANNsFrameResult(
            time=float(t),
            top_classes=top[:5],
            is_unpitchable=is_unpitchable,
            unpitchable_score=unpitchable_score,
            singing_score=singing_score,
        ))

    return results


# ---------------------------------------------------------------------------
# Tier 1+2 analysis (standalone, no MidiSegment dependency)
# ---------------------------------------------------------------------------

def run_tier12_on_segments(
    segments: list[dict],
    audio_path: str,
    confidence_threshold: float = 0.35,
    pitch_stdev_threshold: float = 4.0,
    spectral_flatness_threshold: float = 0.25,
    min_voiced_ratio: float = 0.15,
) -> list[dict]:
    """Run Tier 1+2 analysis on time segments (standalone, no SwiftF0 needed).

    Uses pitch_crepe or librosa pyin as a lightweight pitch estimator
    for the POC (the real pipeline uses SwiftF0).

    Args:
        segments: List of {"word": str, "start": float, "end": float}
        audio_path: Path to vocals audio.

    Returns:
        List of dicts with tier12 analysis results per segment.
    """
    print("Running Tier 1+2 analysis (librosa pyin)...")
    t0 = time.time()

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Use pyin for pitch + voiced probability (approximation of SwiftF0)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=50, fmax=1000, sr=sr, hop_length=256
    )
    hop_length = 256
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
    sf_times = librosa.frames_to_time(
        np.arange(len(flatness)), sr=sr, hop_length=512
    )

    elapsed = time.time() - t0
    print(f"  Pitch analysis done in {elapsed:.1f}s")

    results = []
    for seg in segments:
        mask = (times >= seg["start"]) & (times <= seg["end"])
        seg_f0 = f0[mask]
        seg_prob = voiced_prob[mask]

        if len(seg_prob) == 0:
            results.append({
                "is_growl": False, "median_conf": 1.0,
                "pitch_stdev": 0.0, "spectral_flat": 0.0, "voiced_ratio": 1.0,
            })
            continue

        median_conf = float(np.nanmedian(seg_prob))

        # Voiced frames: probability > 0.1 and valid f0
        voiced = (~np.isnan(seg_f0)) & (seg_prob > 0.1) & (seg_f0 > 50.0)
        voiced_ratio = float(np.sum(voiced)) / len(seg_prob)

        # Pitch stdev in semitones
        voiced_f0 = seg_f0[voiced]
        if len(voiced_f0) >= 2:
            midi_notes = 12.0 * np.log2(voiced_f0 / 440.0) + 69.0
            pitch_stdev = float(np.std(midi_notes))
        else:
            pitch_stdev = 0.0

        # Spectral flatness for segment
        sf_mask = (sf_times >= seg["start"]) & (sf_times <= seg["end"])
        sf_seg = flatness[sf_mask]
        spectral_flat = float(np.median(sf_seg)) if len(sf_seg) > 0 else 0.0

        # Decision (same logic as growl_detector.py)
        tier1 = median_conf < confidence_threshold and pitch_stdev > pitch_stdev_threshold
        tier2 = spectral_flat > spectral_flatness_threshold

        is_growl = tier1 or (tier2 and median_conf < confidence_threshold * 1.3)

        if voiced_ratio < min_voiced_ratio:
            is_growl = False

        results.append({
            "is_growl": is_growl,
            "median_conf": median_conf,
            "pitch_stdev": pitch_stdev,
            "spectral_flat": spectral_flat,
            "voiced_ratio": voiced_ratio,
        })

    return results


# ---------------------------------------------------------------------------
# UltraStar TXT parser (minimal)
# ---------------------------------------------------------------------------

def parse_ultrastar_notes(txt_path: str, bpm: float = None) -> list[dict]:
    """Parse note lines from UltraStar TXT, return word segments with timing."""
    lines = Path(txt_path).read_text(encoding="utf-8", errors="replace").splitlines()

    # Extract BPM from header
    for line in lines:
        if line.startswith("#BPM:"):
            bpm = float(line.split(":")[1].replace(",", "."))
        elif not line.startswith("#"):
            break

    if bpm is None:
        print("  Warning: BPM not found in TXT, assuming 300")
        bpm = 300.0

    # UltraStar timing: beat = bpm/60 * 4 seconds per beat-unit
    beat_duration = 15.0 / bpm  # seconds per beat

    segments = []
    for line in lines:
        if not line or line.startswith("#") or line.startswith("E"):
            continue
        if line.startswith("- "):
            continue  # line break

        parts = line.split(maxsplit=4)
        if len(parts) < 5:
            continue

        note_type = parts[0]  # ":", "*", "F", "R"
        start_beat = int(parts[1])
        duration_beats = int(parts[2])
        # parts[3] = pitch
        word = parts[4] if len(parts) > 4 else ""

        start_s = start_beat * beat_duration
        end_s = (start_beat + duration_beats) * beat_duration

        segments.append({
            "word": word.strip(),
            "start": start_s,
            "end": end_s,
            "note_type": note_type,
            "original_type": note_type,
        })

    print(f"  Parsed {len(segments)} notes from UltraStar TXT (BPM={bpm})")
    return segments


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_tier12_vs_tier3(
    segments: list[dict],
    tier12_results: list[dict],
    panns_frames: list[PANNsFrameResult],
    panns_threshold: float = 0.15,
) -> list[SegmentComparison]:
    """Align PANNs frame results to segments and compare with Tier 1+2."""
    comparisons = []

    for seg, t12 in zip(segments, tier12_results):
        # Aggregate PANNs frames within segment
        seg_frames = [f for f in panns_frames if seg["start"] <= f.time <= seg["end"]]

        if seg_frames:
            avg_unpitchable = np.mean([f.unpitchable_score for f in seg_frames])
            avg_singing = np.mean([f.singing_score for f in seg_frames])

            # Dominant class across segment
            class_scores = {}
            for f in seg_frames:
                for label, score in f.top_classes:
                    class_scores.setdefault(label, []).append(score)
            avg_class_scores = {k: np.mean(v) for k, v in class_scores.items()}
            top_class = max(avg_class_scores, key=avg_class_scores.get) if avg_class_scores else "?"

            tier3_unpitchable = avg_unpitchable > panns_threshold and avg_unpitchable > avg_singing
        else:
            avg_unpitchable = 0.0
            avg_singing = 0.0
            top_class = "?"
            tier3_unpitchable = False

        comp = SegmentComparison(
            word=seg["word"],
            start=seg["start"],
            end=seg["end"],
            tier12_is_growl=t12["is_growl"],
            tier12_median_conf=t12["median_conf"],
            tier12_pitch_stdev=t12["pitch_stdev"],
            tier12_spectral_flat=t12["spectral_flat"],
            tier12_voiced_ratio=t12["voiced_ratio"],
            tier3_unpitchable_score=avg_unpitchable,
            tier3_singing_score=avg_singing,
            tier3_top_class=top_class,
            tier3_is_unpitchable=tier3_unpitchable,
            agree=t12["is_growl"] == tier3_unpitchable,
        )
        comparisons.append(comp)

    return comparisons


def print_comparison_table(comparisons: list[SegmentComparison]):
    """Print a formatted comparison table."""
    print()
    print("=" * 120)
    print(f"{'Word':<15} {'Time':>10} {'T12':>4} {'T3':>4} {'Agr':>4} "
          f"{'Conf':>5} {'PStd':>5} {'SFlat':>5} {'VRat':>5} "
          f"{'Unpit':>6} {'Sing':>6} {'TopClass':<12}")
    print("-" * 120)

    agree_count = 0
    total = len(comparisons)
    t12_growl = 0
    t3_growl = 0

    for c in comparisons:
        t12_mark = "G" if c.tier12_is_growl else "."
        t3_mark = "G" if c.tier3_is_unpitchable else "."
        agr_mark = "OK" if c.agree else "XX"

        if c.agree:
            agree_count += 1
        if c.tier12_is_growl:
            t12_growl += 1
        if c.tier3_is_unpitchable:
            t3_growl += 1

        word_display = c.word[:14] if len(c.word) > 14 else c.word
        time_str = f"{c.start:.1f}-{c.end:.1f}"

        print(f"{word_display:<15} {time_str:>10} {t12_mark:>4} {t3_mark:>4} {agr_mark:>4} "
              f"{c.tier12_median_conf:>5.2f} {c.tier12_pitch_stdev:>5.1f} "
              f"{c.tier12_spectral_flat:>5.3f} {c.tier12_voiced_ratio:>5.2f} "
              f"{c.tier3_unpitchable_score:>6.3f} {c.tier3_singing_score:>6.3f} "
              f"{c.tier3_top_class:<12}")

    print("-" * 120)
    agreement_pct = (agree_count / total * 100) if total > 0 else 0
    print(f"Summary: {total} segments | "
          f"Tier 1+2: {t12_growl} growl | "
          f"Tier 3: {t3_growl} growl | "
          f"Agreement: {agree_count}/{total} ({agreement_pct:.0f}%)")

    # Disagreements breakdown
    both_growl = sum(1 for c in comparisons if c.tier12_is_growl and c.tier3_is_unpitchable)
    only_t12 = sum(1 for c in comparisons if c.tier12_is_growl and not c.tier3_is_unpitchable)
    only_t3 = sum(1 for c in comparisons if not c.tier12_is_growl and c.tier3_is_unpitchable)
    neither = sum(1 for c in comparisons if not c.tier12_is_growl and not c.tier3_is_unpitchable)

    print(f"  Both growl: {both_growl} | Only T12: {only_t12} | "
          f"Only T3: {only_t3} | Neither: {neither}")

    # Top PANNs class distribution
    class_counts = {}
    for c in comparisons:
        class_counts[c.tier3_top_class] = class_counts.get(c.tier3_top_class, 0) + 1
    print(f"\nPANNs top-class distribution:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        print(f"  {cls:<20} {cnt:>4} ({pct:.0f}%)")

    # Show segments where PANNs sees unpitchable-class signal (even if below threshold)
    suspects = [(c, max(c.tier3_unpitchable_score, 0)) for c in comparisons
                if c.tier3_unpitchable_score > 0.02]
    if suspects:
        suspects.sort(key=lambda x: -x[1])
        print(f"\nTop PANNs unpitchable-signal segments (score > 0.02):")
        for c, score in suspects[:15]:
            print(f"  {c.start:>6.1f}-{c.end:<6.1f} {c.word:<15} "
                  f"unpit={c.tier3_unpitchable_score:.3f} sing={c.tier3_singing_score:.3f} "
                  f"top={c.tier3_top_class}")
    print("=" * 120)


def export_csv(comparisons: list[SegmentComparison], csv_path: str):
    """Export comparison results to CSV."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "word", "start", "end",
            "tier12_growl", "tier12_conf", "tier12_pitch_stdev",
            "tier12_spectral_flat", "tier12_voiced_ratio",
            "tier3_unpitchable", "tier3_unpitchable_score", "tier3_singing_score",
            "tier3_top_class", "agree",
        ])
        for c in comparisons:
            writer.writerow([
                c.word, f"{c.start:.3f}", f"{c.end:.3f}",
                c.tier12_is_growl, f"{c.tier12_median_conf:.3f}",
                f"{c.tier12_pitch_stdev:.2f}", f"{c.tier12_spectral_flat:.4f}",
                f"{c.tier12_voiced_ratio:.3f}",
                c.tier3_is_unpitchable, f"{c.tier3_unpitchable_score:.4f}",
                f"{c.tier3_singing_score:.4f}", c.tier3_top_class,
                c.agree,
            ])
    print(f"\nCSV exported to: {csv_path}")


def run_panns_only(audio_path: str, device: str = "cpu"):
    """Run PANNs-only analysis without UltraStar segments."""
    frame_times, framewise, labels = run_panns_sed(audio_path, device)
    panns_frames = analyze_panns_frames(frame_times, framewise, labels)

    # Print timeline with dominant vocal class per frame
    print(f"\n{'Time':>8} {'Top Vocal Class':<20} {'Score':>6} {'Unpit':>6} {'Sing':>6} {'Flag':>5}")
    print("-" * 60)

    for f in panns_frames:
        top = f.top_classes[0] if f.top_classes else ("?", 0.0)
        flag = "GROWL" if f.is_unpitchable else ""
        print(f"{f.time:>8.2f} {top[0]:<20} {top[1]:>6.3f} "
              f"{f.unpitchable_score:>6.3f} {f.singing_score:>6.3f} {flag:>5}")


def main():
    parser = argparse.ArgumentParser(
        description="POC: Compare Tier 1+2 growl detection with PANNs Tier 3"
    )
    parser.add_argument("audio", help="Path to vocals audio file (e.g. [Vocals].ogg)")
    parser.add_argument("--ultrastar", help="Path to UltraStar .txt for segment-level comparison")
    parser.add_argument("--csv", help="Export comparison to CSV file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="PyTorch device (default: cpu)")
    parser.add_argument("--panns-threshold", type=float, default=0.15,
                        help="PANNs unpitchable confidence threshold (default: 0.15)")
    args = parser.parse_args()

    audio_path = args.audio
    if not Path(audio_path).exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    if args.ultrastar:
        # Full comparison mode: Tier 1+2 vs Tier 3 on UltraStar segments
        if not Path(args.ultrastar).exists():
            print(f"ERROR: UltraStar file not found: {args.ultrastar}")
            sys.exit(1)

        print("=== PANNs Growl Detection POC - Segment Comparison Mode ===\n")

        # 1. Parse UltraStar segments
        segments = parse_ultrastar_notes(args.ultrastar)

        # 2. Run PANNs
        frame_times, framewise, labels = run_panns_sed(audio_path, args.device)
        panns_frames = analyze_panns_frames(
            frame_times, framewise, labels, threshold=args.panns_threshold
        )

        # 3. Run Tier 1+2
        tier12_results = run_tier12_on_segments(segments, audio_path)

        # 4. Compare
        comparisons = compare_tier12_vs_tier3(
            segments, tier12_results, panns_frames, panns_threshold=args.panns_threshold
        )
        print_comparison_table(comparisons)

        # 5. Export CSV if requested
        if args.csv:
            export_csv(comparisons, args.csv)

    else:
        # PANNs-only mode: frame-level timeline
        print("=== PANNs Growl Detection POC - Frame-Level Mode ===\n")
        print("(No --ultrastar given, running PANNs-only frame analysis)")
        run_panns_only(audio_path, args.device)


if __name__ == "__main__":
    main()
