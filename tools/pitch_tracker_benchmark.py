"""Benchmark pitch trackers (SwiftF0, FCPE, Penn) on isolated vocal audio.

Compares frame-level pitch output across trackers to identify which produces
the most stable, accurate, and musically meaningful pitch contours.

Usage:
    uv run python tools/pitch_tracker_benchmark.py <vocal_audio_file_or_dir>
    uv run python tools/pitch_tracker_benchmark.py <vocal_audio> --reference <ultrastar_txt>
"""

import argparse
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

# Add src to path for UltraSinger imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_audio_float32(filepath: str) -> tuple[np.ndarray, int]:
    """Load audio file as float32 mono, return (audio, sample_rate)."""
    audio, sr = librosa.load(filepath, sr=None, mono=True)
    return audio.astype(np.float32), sr


def load_audio_wav(filepath: str) -> tuple[np.ndarray, int]:
    """Load audio via scipy for SwiftF0 compatibility."""
    sr, audio = wavfile.read(filepath)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if audio.dtype in [np.int16, np.int32, np.int64]:
        dtype_info = np.iinfo(audio.dtype)
        max_val = max(abs(dtype_info.min), abs(dtype_info.max))
        audio = audio.astype(np.float32) / float(max_val)
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32) / 32768.0
    return audio, sr


def hz_to_midi(freq: float) -> float:
    """Convert Hz to MIDI note number. Returns NaN for 0 Hz."""
    if freq <= 0:
        return float("nan")
    return 12.0 * np.log2(freq / 440.0) + 69.0


def midi_to_note_name(midi: float) -> str:
    """Convert MIDI number to note name."""
    if np.isnan(midi):
        return "--"
    return librosa.midi_to_note(round(midi))


# ── SwiftF0 ──────────────────────────────────────────────────────────────────

def run_swiftf0(audio: np.ndarray, sr: int) -> dict:
    """Run SwiftF0 pitch detection."""
    from swift_f0 import SwiftF0

    detector = SwiftF0(fmin=46.875, fmax=2093.75, confidence_threshold=0.7)

    start = time.perf_counter()
    result = detector.detect_from_array(audio, sr)
    elapsed = time.perf_counter() - start

    times = np.array([float(t) for t in result.timestamps])
    freqs = np.array([float(f) for f in result.pitch_hz])
    confs = np.array([float(c) for c in result.confidence])

    return {"name": "SwiftF0", "times": times, "freqs": freqs, "confs": confs,
            "elapsed": elapsed}


# ── FCPE ──────────────────────────────────────────────────────────────────────

def run_fcpe(audio: np.ndarray, sr: int) -> dict:
    """Run FCPE (torchfcpe) pitch detection."""
    import torch
    from torchfcpe import spawn_bundled_infer_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = spawn_bundled_infer_model(device=device)

    # FCPE expects [batch, samples] tensor at 16kHz
    target_sr = 16000
    start = time.perf_counter()
    if sr != target_sr:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    else:
        audio_16k = audio

    audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).to(device)

    # hop_size in samples at target_sr, default 160 (10ms)
    hop_size = 160
    f0 = model.infer(audio_tensor, sr=target_sr, decoder_mode="local_argmax",
                     threshold=0.006)

    # f0 shape: [batch, frames, 1]
    f0_np = np.atleast_1d(f0.squeeze().cpu().numpy())

    # Generate timestamps
    n_frames = len(f0_np)
    times = np.arange(n_frames) * (hop_size / target_sr)
    freqs = np.maximum(f0_np, 0.0)  # Clip negatives

    # Derive confidence from energy and pitch stability (same as production)
    from modules.Pitcher.fcpe_pitcher import _compute_frame_confidence
    confs = np.array(_compute_frame_confidence(
        audio_16k, freqs.tolist(), hop_size
    ))
    elapsed = time.perf_counter() - start

    return {"name": "FCPE", "times": times, "freqs": freqs, "confs": confs,
            "elapsed": elapsed}


# ── Penn ──────────────────────────────────────────────────────────────────────

def run_penn(audio: np.ndarray, sr: int) -> dict:
    """Run Penn (FCNF0++) pitch detection."""
    import torch
    import penn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Penn expects [1, samples] tensor
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

    start = time.perf_counter()
    pitch, periodicity = penn.from_audio(
        audio_tensor, sr,
        hopsize=penn.HOPSIZE / penn.SAMPLE_RATE,
        fmin=penn.FMIN, fmax=penn.FMAX,
        batch_size=2048,
        interp_unvoiced_at=0.065,
        gpu=(0 if device == "cuda" else None)
    )
    elapsed = time.perf_counter() - start

    freqs = np.atleast_1d(pitch.squeeze().cpu().numpy())
    confs = np.atleast_1d(periodicity.squeeze().cpu().numpy())
    n_frames = len(freqs)
    hop_seconds = penn.HOPSIZE / penn.SAMPLE_RATE
    times = np.arange(n_frames) * hop_seconds

    return {"name": "Penn", "times": times, "freqs": freqs, "confs": confs,
            "elapsed": elapsed}


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_result(result: dict, conf_threshold: float = 0.4) -> dict:
    """Compute statistics for a pitch tracker result."""
    freqs = result["freqs"]
    confs = result["confs"]

    # Voiced frames (frequency > 0 and confidence above threshold)
    voiced_mask = (freqs > 0) & (confs >= conf_threshold)
    voiced_freqs = freqs[voiced_mask]
    voiced_confs = confs[voiced_mask]

    n_total = len(freqs)
    n_voiced = int(voiced_mask.sum())

    if n_voiced == 0:
        return {
            "n_frames": n_total, "n_voiced": 0, "voiced_pct": 0.0,
            "mean_conf": 0.0, "median_midi": 0.0, "pitch_range_st": 0.0,
            "pitch_std_st": 0.0, "frame_jumps_gt2st": 0,
            "frame_jumps_gt5st": 0,
        }

    midi_values = np.array([hz_to_midi(f) for f in voiced_freqs])
    midi_values = midi_values[~np.isnan(midi_values)]

    # Frame-to-frame pitch jumps (in semitones)
    midi_diffs = np.abs(np.diff(midi_values))
    jumps_gt2 = int((midi_diffs > 2.0).sum())
    jumps_gt5 = int((midi_diffs > 5.0).sum())

    return {
        "n_frames": n_total,
        "n_voiced": n_voiced,
        "voiced_pct": 100.0 * n_voiced / n_total,
        "mean_conf": float(np.mean(voiced_confs)),
        "median_midi": float(np.median(midi_values)) if len(midi_values) > 0 else 0.0,
        "median_note": midi_to_note_name(float(np.median(midi_values))) if len(midi_values) > 0 else "--",
        "pitch_range_st": float(np.ptp(midi_values)) if len(midi_values) > 0 else 0.0,
        "pitch_std_st": float(np.std(midi_values)) if len(midi_values) > 0 else 0.0,
        "frame_jumps_gt2st": jumps_gt2,
        "frame_jumps_gt5st": jumps_gt5,
    }


def compare_trackers(results: list[dict], ref_result_name: str = "SwiftF0",
                     conf_threshold: float = 0.4) -> None:
    """Compare pitch tracker results and print analysis."""
    print("\n" + "=" * 80)
    print("PITCH TRACKER BENCHMARK RESULTS")
    print("=" * 80)

    analyses = {}
    for r in results:
        name = r["name"]
        stats = analyze_result(r, conf_threshold)
        analyses[name] = stats

        print(f"\n-- {name} ({'%.2f' % r['elapsed']}s) --")
        print(f"  Frames: {stats['n_frames']:,}  |  Voiced: {stats['n_voiced']:,} ({stats['voiced_pct']:.1f}%)")
        print(f"  Mean Confidence: {stats['mean_conf']:.3f}")
        print(f"  Median Note: {stats.get('median_note', '--')} (MIDI {stats['median_midi']:.1f})")
        print(f"  Pitch Range: {stats['pitch_range_st']:.1f} ST  |  Std Dev: {stats['pitch_std_st']:.1f} ST")
        print(f"  Frame Jumps >2 ST: {stats['frame_jumps_gt2st']}  |  >5 ST: {stats['frame_jumps_gt5st']}")

    # Cross-comparison: pitch agreement
    if len(results) >= 2:
        print("\n-- CROSS-COMPARISON --")
        ref = None
        for r in results:
            if r["name"] == ref_result_name:
                ref = r
                break
        if ref is None:
            ref = results[0]

        for r in results:
            if r["name"] == ref["name"]:
                continue
            agreement = compute_pitch_agreement(ref, r, conf_threshold)
            print(f"  {ref['name']} vs {r['name']}:")
            print(f"    Overlapping voiced frames: {agreement['n_overlap']}")
            print(f"    Pitch agreement ±1 ST: {agreement['agree_1st']:.1f}%")
            print(f"    Pitch agreement ±2 ST: {agreement['agree_2st']:.1f}%")
            print(f"    Median difference: {agreement['median_diff_st']:.2f} ST")


def compute_pitch_agreement(ref: dict, test: dict,
                            conf_threshold: float = 0.4) -> dict:
    """Compute pitch agreement between two tracker results."""
    # Interpolate test to ref timebase
    ref_times = ref["times"]
    ref_freqs = ref["freqs"]
    ref_confs = ref["confs"]

    test_times = test["times"]
    test_freqs = test["freqs"]
    test_confs = test["confs"]

    if len(ref_times) == 0 or len(test_times) == 0:
        return {"n_overlap": 0, "agree_1st": 0.0, "agree_2st": 0.0,
                "median_diff_st": 0.0}

    # For each ref frame, find nearest test frame
    n_overlap = 0
    diffs = []

    for i, t in enumerate(ref_times):
        if ref_freqs[i] <= 0 or ref_confs[i] < conf_threshold:
            continue
        # Find nearest test frame
        j = np.searchsorted(test_times, t)
        j = min(j, len(test_times) - 1)
        if j > 0 and abs(test_times[j - 1] - t) < abs(test_times[j] - t):
            j = j - 1
        if abs(test_times[j] - t) > 0.05:  # Skip if >50ms apart
            continue
        if test_freqs[j] <= 0 or test_confs[j] < conf_threshold:
            continue

        ref_midi = hz_to_midi(ref_freqs[i])
        test_midi = hz_to_midi(test_freqs[j])
        if np.isnan(ref_midi) or np.isnan(test_midi):
            continue

        n_overlap += 1
        diffs.append(test_midi - ref_midi)

    if n_overlap == 0:
        return {"n_overlap": 0, "agree_1st": 0.0, "agree_2st": 0.0,
                "median_diff_st": 0.0}

    diffs = np.array(diffs)
    abs_diffs = np.abs(diffs)

    return {
        "n_overlap": n_overlap,
        "agree_1st": 100.0 * np.mean(abs_diffs <= 1.0),
        "agree_2st": 100.0 * np.mean(abs_diffs <= 2.0),
        "median_diff_st": float(np.median(diffs)),
    }


def parse_ultrastar_notes(txt_path: str) -> list[dict]:
    """Parse UltraStar .txt file and extract note timing + pitch."""
    notes = []
    gap_ms = 0.0
    bpm = 120.0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#GAP:"):
                gap_ms = float(line.split(":")[1].replace(",", "."))
            elif line.startswith("#BPM:"):
                bpm = float(line.split(":")[1].replace(",", "."))
            elif line and line[0] in ":*F":
                parts = line.split(None, 4)
                if len(parts) >= 4:
                    note_type = parts[0]
                    start_beat = int(parts[1])
                    duration = int(parts[2])
                    pitch = int(parts[3])
                    # Convert beats to seconds
                    beat_duration_s = 60.0 / (bpm * 4)  # UltraStar quarter-beats
                    start_s = gap_ms / 1000.0 + start_beat * beat_duration_s
                    end_s = start_s + duration * beat_duration_s
                    midi = pitch + 48  # UltraStar to MIDI
                    notes.append({
                        "start": start_s, "end": end_s,
                        "midi": midi, "type": note_type,
                        "word": parts[4] if len(parts) > 4 else "",
                    })
    return notes


def evaluate_against_reference(results: list[dict], ref_notes: list[dict],
                               conf_threshold: float = 0.4) -> None:
    """Evaluate tracker results against UltraStar reference notes."""
    print("\n-- REFERENCE COMPARISON --")
    print(f"  Reference: {len(ref_notes)} notes")

    for r in results:
        name = r["name"]
        correct_1st = 0
        correct_2st = 0
        total = 0

        for note in ref_notes:
            if note["type"] == "F":  # Skip freestyle
                continue
            # Find frames in note time range
            mask = (r["times"] >= note["start"]) & (r["times"] <= note["end"])
            mask &= (r["freqs"] > 0) & (r["confs"] >= conf_threshold)
            note_freqs = r["freqs"][mask]
            if len(note_freqs) == 0:
                total += 1
                continue

            # Weighted median pitch for this note
            note_confs = r["confs"][mask]
            sort_idx = np.argsort(note_freqs)
            sorted_freqs = note_freqs[sort_idx]
            sorted_confs = note_confs[sort_idx]
            cum_weight = np.cumsum(sorted_confs)
            median_idx = np.searchsorted(cum_weight, cum_weight[-1] / 2)
            median_freq = sorted_freqs[min(median_idx, len(sorted_freqs) - 1)]
            detected_midi = hz_to_midi(median_freq)

            if not np.isnan(detected_midi):
                diff = abs(detected_midi - note["midi"])
                # Allow octave equivalence
                diff_chroma = min(diff, abs(diff - 12), abs(diff + 12))
                if diff_chroma <= 1.0:
                    correct_1st += 1
                if diff_chroma <= 2.0:
                    correct_2st += 1
            total += 1

        if total > 0:
            print(f"  {name}: ±1 ST = {100.0 * correct_1st / total:.1f}%  |  "
                  f"±2 ST = {100.0 * correct_2st / total:.1f}%  |  "
                  f"({correct_2st}/{total} notes)")


def process_file(audio_path: str, ref_path: str | None,
                 trackers: list[str],
                 conf_threshold: float = 0.4) -> list[dict]:
    """Run all trackers on a single audio file."""
    print(f"\nProcessing: {os.path.basename(audio_path)}")
    duration = librosa.get_duration(path=audio_path)
    print(f"  Duration: {duration:.1f}s")

    # Load audio
    audio_librosa, sr_librosa = load_audio_float32(audio_path)

    # For SwiftF0: need to convert to WAV first if not WAV
    import tempfile
    import soundfile as sf
    tmp_wav = None
    if not audio_path.lower().endswith(".wav"):
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, audio_librosa, sr_librosa)
        tmp_wav.close()

    results = []

    try:
        if "swiftf0" in trackers:
            try:
                if tmp_wav:
                    audio_sf, sr_sf = load_audio_wav(tmp_wav.name)
                else:
                    audio_sf, sr_sf = load_audio_wav(audio_path)
                result = run_swiftf0(audio_sf, sr_sf)
                results.append(result)
                print(f"  SwiftF0: {len(result['freqs'])} frames in {result['elapsed']:.2f}s")
            except Exception as e:
                print(f"  SwiftF0 FAILED ({type(e).__name__}): {e}")

        if "fcpe" in trackers:
            try:
                result = run_fcpe(audio_librosa, sr_librosa)
                results.append(result)
                print(f"  FCPE: {len(result['freqs'])} frames in {result['elapsed']:.2f}s")
            except Exception as e:
                print(f"  FCPE FAILED ({type(e).__name__}): {e}")

        if "penn" in trackers:
            try:
                result = run_penn(audio_librosa, sr_librosa)
                results.append(result)
                print(f"  Penn: {len(result['freqs'])} frames in {result['elapsed']:.2f}s")
            except Exception as e:
                print(f"  Penn FAILED ({type(e).__name__}): {e}")
    finally:
        if tmp_wav and os.path.exists(tmp_wav.name):
            os.unlink(tmp_wav.name)

    # Analysis
    compare_trackers(results, conf_threshold=conf_threshold)

    # Reference comparison if available
    if ref_path and os.path.exists(ref_path):
        ref_notes = parse_ultrastar_notes(ref_path)
        if ref_notes:
            evaluate_against_reference(results, ref_notes, conf_threshold)

    return results


def find_vocal_files(path: str) -> list[tuple[str, str | None]]:
    """Find vocal audio files and their reference .txt files."""
    entries = []
    p = Path(path)

    if p.is_file():
        # Single file
        txt = p.with_suffix(".txt")
        ref = str(txt) if txt.exists() else None
        entries.append((str(p), ref))
    elif p.is_dir():
        # Look for [Vocals] files in subdirectories
        for vocal in sorted(p.rglob("*")):
            if "[Vocals]" not in vocal.name:
                continue
            if vocal.suffix.lower() in (".ogg", ".wav", ".mp3", ".flac"):
                # Find corresponding .txt
                txt_candidates = list(vocal.parent.glob("*.txt"))
                ref = None
                for t in txt_candidates:
                    if t.name != "ultrasinger_parameter.info":
                        ref = str(t)
                        break
                entries.append((str(vocal), ref))
        # Limit to avoid excessive processing
        if len(entries) > 5:
            print(f"Found {len(entries)} vocal files, limiting to first 5")
            entries = entries[:5]
    return entries


def main():
    parser = argparse.ArgumentParser(description="Benchmark pitch trackers")
    parser.add_argument("audio", help="Vocal audio file or directory")
    parser.add_argument("--reference", "-r", help="UltraStar .txt reference file")
    parser.add_argument("--trackers", "-t", default="swiftf0,fcpe,penn",
                        help="Comma-separated tracker names (default: swiftf0,fcpe,penn)")
    parser.add_argument("--conf-threshold", type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    args = parser.parse_args()

    trackers = [t.strip().lower() for t in args.trackers.split(",")]
    print(f"Trackers: {', '.join(trackers)}")

    entries = find_vocal_files(args.audio)
    if not entries:
        print(f"No vocal audio files found in {args.audio}")
        sys.exit(1)

    if args.reference:
        entries = [(entries[0][0], args.reference)]

    all_results = []
    for audio_path, ref_path in entries:
        results = process_file(audio_path, ref_path, trackers,
                                    args.conf_threshold)
        all_results.append((audio_path, results))

    # Summary across all files
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("AGGREGATE SUMMARY")
        print("=" * 80)
        for tracker_name in trackers:
            speeds = []
            voiced_pcts = []
            stds = []
            jumps = []
            for _, results in all_results:
                for r in results:
                    if r["name"].lower().replace(" ", "") == tracker_name.replace(" ", ""):
                        stats = analyze_result(r, args.conf_threshold)
                        speeds.append(r["elapsed"])
                        voiced_pcts.append(stats["voiced_pct"])
                        stds.append(stats["pitch_std_st"])
                        jumps.append(stats["frame_jumps_gt2st"])
            if speeds:
                print(f"\n  {tracker_name.upper()}:")
                print(f"    Avg Speed: {np.mean(speeds):.2f}s  |  "
                      f"Avg Voiced: {np.mean(voiced_pcts):.1f}%  |  "
                      f"Avg Pitch Std: {np.mean(stds):.1f} ST  |  "
                      f"Avg Jumps>2ST: {np.mean(jumps):.0f}")


if __name__ == "__main__":
    main()
