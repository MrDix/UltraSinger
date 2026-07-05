# Library Triage Tool (Experimental)

> ⚠️ **Experimental.** This tool **moves files on disk**. Always start with a
> dry run (the default), keep a backup of anything important, and read this
> whole page before using `--apply`.

`tools/library_triage.py` scans an UltraStar song library and moves songs that
are **broken** — or that score badly against their own extracted vocals — out
of the library into a separate "corrupt" directory, so you can review them
away from your working collection.

It answers two questions for every song:

1. **Is it corrupt?** (Stage 1, fast, no GPU) — can the chart even be parsed and
   the audio decoded?
2. **Is it any good?** (Stage 2, optional, GPU) — does the chart actually match
   the singing, measured with the same ptAKF engine the karaoke games use to
   score you?

---

## Quick start

```commandline
# 1) DRY RUN over the whole library (reads only, moves nothing):
uv run python tools/library_triage.py "D:\UltraStar\Songs" "D:\UltraStar\Corrupt"

# 2) Review the report + the CSV, then actually move the obviously broken ones:
uv run python tools/library_triage.py "D:\UltraStar\Songs" "D:\UltraStar\Corrupt" --apply

# 3) Optionally, the deeper vocal-score test on a chosen sub-folder (uses the GPU):
uv run python tools/library_triage.py "D:\UltraStar\Songs\Some Collection" "D:\UltraStar\Corrupt" --stage2 --apply
```

Usage:

```
library_triage.py <source_dir> <corrupt_dir> [options]
```

- `<source_dir>` — the library (or any sub-folder) to scan.
- `<corrupt_dir>` — where broken songs are moved. **Created automatically if it
  does not exist.** Must **not** be inside `<source_dir>`.

---

## What counts as a "song"

A folder is treated as a song when it directly contains at least one `.txt`
that looks like an UltraStar chart (has `#TITLE` / `#BPM` / note lines). Plain
collection folders, or folders whose only `.txt` is something like a readme,
are ignored — never moved.

---

## Stage 1 — corruption check (always runs, no GPU)

A song is flagged **corrupt** in Stage 1 only when it is *unambiguously* broken.
This is deliberately conservative: when in doubt, the song is kept.

| Verdict | Meaning |
| --- | --- |
| `unparsable_txt` | The chart cannot be parsed (e.g. a non-numeric `#BPM`). |
| `no_notes` | The chart parses but has zero notes. |
| `no_audio` | Neither the `#MP3` / `#AUDIO` file nor any audio file (`.mp3/.ogg/.opus/.m4a/.wav/.flac`) is present in the folder. |
| `audio_undecodable` | `ffprobe` cannot read the audio, or it has no audio stream / zero duration. |

> Note on garbled text: mojibake (wrong-encoding lyrics) alone is **not** treated
> as corruption — the song still plays. Truly unreadable charts fall into
> `no_notes` instead.

---

## Stage 2 — vocal-score check (optional, `--stage2`, uses the GPU)

For songs that pass Stage 1, Stage 2 measures whether the chart matches the
actual singing:

1. Separate the vocals into a **temporary** directory (Mel-Band-Roformer by
   default; the model is loaded once and reused across songs).
2. Score the chart at **Medium** difficulty against those vocals using the real
   ptAKF game-scoring engine.
3. If the score is **below `--stage2-threshold` (default 40 %)**, the song fails
   and is moved. The temporary vocals are always deleted afterwards.

The default threshold is intentionally low: it targets clearly broken or
badly-desynced charts (wrong audio, large timing offset), not merely mediocre
ones. Raise it if you want a stricter cull, lower it to be even more cautious.

> **Cost:** separation is the slow part (~1–2 min/song on a mid-range CUDA GPU).
> Running Stage 2 over a very large library takes a long time — prefer running
> it on one collection or sub-folder at a time.

---

## How songs are moved

A song keeps its full relative path **including the source folder's own name**.
For example, with:

```
library_triage.py  C:\Songs\Foo   C:\Trash
```

a broken song at `C:\Songs\Foo\Party2\Some Artist - Some Song\` is moved to:

```
C:\Trash\Foo\Party2\Some Artist - Some Song\
```

The intermediate folders are recreated, so different source roots stay separated
under one trash directory and you can see where each song came from.

---

## Safety

This tool is built to fail safe:

- **Dry run is the default.** Without `--apply` it only prints what *would* move
  and writes the CSV log; nothing on disk is changed.
- **Errors never move a song.** If separation or scoring throws for any reason,
  the song is recorded as `error` and **kept** — a move happens only when a real
  score was computed and found to be below the threshold.
- **Hard guards:** the run aborts if `<corrupt_dir>` is inside `<source_dir>`
  (which would re-scan moved songs) or equal to it; the source root itself is
  never moved.
- **No overwriting:** if a destination already exists, the song is skipped and
  logged instead of overwritten.
- **Isolated failures:** a single failed move (locked file, permissions) is
  logged and the run continues.

---

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--apply` | off (dry run) | Actually move songs. Without it, nothing is moved. |
| `--stage2` | off | Run the GPU vocal-scoring stage on Stage-1 survivors. |
| `--stage2-threshold FLOAT` | `40.0` | Medium game-score % below which a song fails Stage 2. |
| `--log PATH` | `<corrupt_dir>/triage_log.csv` | Progress/verdict CSV (see below). |
| `--limit N` | — | Process at most N songs (handy for a first test). |
| `--separator-model NAME` | Mel-Band-Roformer | audio-separator model used in Stage 2. |

---

## The CSV log & resuming

Every processed song is written to the CSV (`rel_path, stage1_verdict,
stage2_score, action, reason`). Re-running with the **same log** resumes:
songs already resolved as `moved` or `kept` are skipped. Two useful details:

- A song `kept` by a **Stage-1-only** run is **re-checked** if you later add
  `--stage2` (so the first cheap pass and a later deep pass can share one log).
- `error` rows are retried on the next run.

---

## Recommended workflow

1. **Dry run** the whole library (Stage 1). Read the summary and the CSV.
2. `--apply` Stage 1 to clear the obviously broken songs.
3. Run **`--stage2 --apply`** selectively — one collection at a time, or
   overnight in batches — to catch charts that parse fine but don't match the
   vocals.
4. Review the `<corrupt_dir>` and decide what to re-convert, fix, or delete.

---

*This is an experimental helper tool and is not part of the normal conversion
pipeline. See the main [README](../README.md#-experimental-features) for the
rest of UltraSinger's features.*
