# Model-Based Note Segmentation (Experimental)

By default UltraSinger builds notes from the **word timing** of the lyrics: every
word (or syllable) becomes a note, which is then split where the pitch changes.
Where words are mistimed, missing or not sung as written, the notes end up in the
wrong place.

With `--segmentation_model`, a small neural network looks at the **separated
vocal** instead and predicts, frame by frame (16 ms):

* whether a note is sung here, a freestyle/spoken passage, or nothing to chart,
* where a new note starts — also between notes that follow each other without a gap.

The note pitch comes from the pitch tracker (SwiftF0) over each predicted note:
the 60th percentile of its confident frames, a little above their median,
because sung notes sag below their written pitch (scoops into the note, a
falling end, vibrato). The lyrics from the usual sources (synced lyrics or
Whisper, with forced alignment) are split into syllables and placed onto the
predicted notes in order of time; a syllable held over several notes gets `~`
continuations.

> **No model is shipped.** You train your own on your UltraStar song library.
> The model and the extracted training data are derived from your library —
> keep them private and outside this repository. The training tool refuses to
> write into the repository.

---

## Quick start

```commandline
# 1) Extract training examples: separates the vocals of every song and stores
#    pitch, spectrogram and the chart's notes (GPU recommended, ~20-40 s per song)
uv run python tools/train_segmentation.py extract "D:\Karaoke\Songs" "D:\SegTrain"

# 2) Train (GPU: about 30 minutes for ~1000 songs) and tune the decoding thresholds
uv run python tools/train_segmentation.py train "D:\SegTrain" --out "D:\SegTrain\segmentation.pt"

# 3) Use it
uv run python src/UltraSinger.py -i "Artist - Title.mp4" --segmentation_model "D:\SegTrain\segmentation.pt"
```

`extract` is resumable — run it again to continue after an interruption. Use
`--limit N` for a quick test with the first N songs.

In the GUI: Settings → Experimental Features → **Segmentation Model**.

### Training in the GUI

The **Training** page (sidebar) runs the same two steps: choose the song library, a work
folder for the extracted data (outside the UltraSinger folder), optionally a `songs.json`
of songs to exclude (e.g. a chart benchmark sample) and where to save the model, then
**Start Training**. The page shows the extraction and training progress and the log;
**Cancel** stops it, and starting again continues the extraction where it stopped. When
the model is ready, **Use This Model** sets it as Segmentation Model in the settings.

### Using a model from a GitHub repository

To use the same model on several computers, put it into a GitHub repository —
a **private** one, since it is derived from your library — and let UltraSinger
download it:

```commandline
uv run python src/UltraSinger.py -i "Artist - Title.mp4" --segmentation_model_repo owner/models-repo
```

* `--segmentation_model_repo owner/repo` uses the file `segmentation.pt` in the
  repository root; `owner/repo/path/to/model.pt` picks another file. Examples:
  `my-account/my-models` or `my-account/my-models/models/v2.pt`. The address of
  the repository or the file as the browser shows it on github.com works as well
  (`https://github.com/my-account/my-models/blob/main/models/v2.pt`; the branch
  in it is ignored, the file always comes from the default branch). A raw
  download link does not work.
* A private repository needs a GitHub access token with read access to the
  repository's contents (for a fine-grained token: *Contents: Read-only* on that
  repository). Set it as the `ULTRASINGER_MODEL_TOKEN` environment variable;
  `--segmentation_model_token <token>` works as well, but command-line arguments
  are visible to other local processes. Public repositories need no token.
* The model is cached per user (`%LOCALAPPDATA%\UltraSinger\models` on Windows,
  `~/.cache/ultrasinger/models` elsewhere) and only downloaded again when the
  file in the repository changed. Without network access the cached copy is used.
* A local `--segmentation_model` file takes precedence.

In the GUI: **Model Repository** and **Repository Token** below Segmentation
Model. The token is kept in the system keyring, not in the settings file, and is
passed to the conversion through its environment, not its command line.

---

## Which songs make good training data

`extract` uses the same rules as the [chart benchmark](chart-benchmark.md): one
solo `.txt` per folder (no duets), at least 100 pitched notes, and the audio file
it refers to. Each chart is aligned to the sung pitch of the separated vocal; how
well it fits is stored per song, and `train` skips songs below `--min-ref-fit`
(default 0.5). `train` also skips charts that go on after the end of the audio
(notes starting more than a second after it) — a duet flattened into one track
sometimes lists the second voice after the song, so the passages that voice
sings would be labelled "no note"; a chart of a longer version of the song is
skipped as well (it reports how many songs it skipped). Charts with sloppy timing teach the model sloppy timing, so the
model can only be as good as the charts it learns from. Several hundred
well-timed songs are a good start; more data keeps helping.

The model learns the **style** of your charts too — how long held notes are,
whether runs are split into many notes, what is marked as freestyle.

---

## Measuring a model honestly

A model always looks good on the songs it was trained on. Measure it on songs it
has never seen, with the [Chart Benchmark Tool](chart-benchmark.md):

```commandline
# Sample a benchmark set first and exclude it from training
uv run python tools/chart_benchmark.py sample "D:\Karaoke\Songs" "D:\ChartBench" --count 100
uv run python tools/train_segmentation.py extract "D:\Karaoke\Songs" "D:\SegTrain" --exclude "D:\ChartBench\songs.json"
uv run python tools/train_segmentation.py train "D:\SegTrain" --out "D:\SegTrain\segmentation.pt"

# Compare word-based notes (baseline) with the model
uv run python tools/chart_benchmark.py convert "D:\ChartBench" --label baseline
uv run python tools/chart_benchmark.py convert "D:\ChartBench" --label model --args "--segmentation_model D:\SegTrain\segmentation.pt"
uv run python tools/chart_benchmark.py evaluate "D:\ChartBench" --label baseline
uv run python tools/chart_benchmark.py evaluate "D:\ChartBench" --label model
uv run python tools/chart_benchmark.py compare "D:\ChartBench" baseline model
```

---

## Options

`extract LIBRARY WORKDIR`

| Option | Meaning |
|---|---|
| `--exclude FILE ...` | `songs.json` files (e.g. a chart benchmark sample) whose songs are skipped — also when resuming; data already extracted for them is removed |
| `--limit N` | only process the first N songs |

`train WORKDIR --out MODEL.pt`

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 30 | training epochs (the best epoch on the validation songs is kept) |
| `--steps` | 400 | training steps per epoch |
| `--batch-size` | 24 | 10-second windows per step |
| `--val-fraction` | 0.05 | share of songs held out for validation and threshold tuning |
| `--min-ref-fit` | 0.5 | skip songs whose chart fits the vocal worse than this |
| `--seed` | 42 | random seed (validation split, training order) |
| `--cpu` | off | train on the CPU (much slower) |

The model file contains the network weights and the decoding thresholds tuned
on the validation songs.

---

## Notes

* The model sees the vocal exactly as the pipeline separates it. Train and
  convert with the same separation model (the default is Mel-Band-Roformer).
* If the model file is missing, no vocal separation is available, or the step
  fails, UltraSinger keeps the word-based notes and prints a warning. The
  settings info file shows whether the model was applied.
* Pitch-change splitting, syllable merging, the timing refinement (which
  snaps note starts to audio onsets) and the GAP sweep (which shifts all notes
  together to where the game scores them best) are skipped when the model is
  used, because the model already decides where notes start and end, and more
  precisely than the onset snapping or the shift would.
* Note pitches come from a lead-vocal stem (an extra karaoke separation of the
  vocal stem, cached per song) whenever that stem kept at least 80 % of the
  singing; otherwise from the full vocal stem. This keeps harmonies and backing
  vocals out of the chart. The later steps that compare the note pitches with
  the singing (pitch refinement, ptAKF refit, game score) then use the lead stem
  as well, so they do not pull the pitches back to a louder backing voice; notes
  in which the game's pitch detection finds no tone in the lead stem are checked
  against the full vocal stem.
  `--disable_lead_vocal_pitch` (GUI: Lead Vocal Pitch) skips the extra
  separation.
* The pitch refinement (which moves badly scoring notes to the pitch the game's
  own pitch detection hears) keeps the pitch of model notes that the pitch
  tracker followed through at least half of the note, unless a second pitch
  track measures the corrected pitch too: the full vocal stem for pitches from
  the lead stem, otherwise the lead stem (when it was made). Where the pitch
  tracker is sure, the game's detection is more often the one that is wrong,
  e.g. a fourth or fifth off on a harmonic.
