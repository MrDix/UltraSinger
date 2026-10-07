# Model-Based Note Segmentation (Experimental)

By default UltraSinger builds notes from the **word timing** of the lyrics: every
word (or syllable) becomes a note, which is then split where the pitch changes.
Where words are mistimed, missing or not sung as written, the notes end up in the
wrong place.

With `--segmentation_model`, a small neural network looks at the **separated
vocal** instead and predicts, frame by frame (16 ms):

* whether a note is sung here, a freestyle/spoken passage, or nothing to chart,
* where a new note starts — also between notes that follow each other without a gap.

The note pitch is the median of the pitch tracker (SwiftF0) over each predicted
note. The lyrics from the usual sources (synced lyrics or Whisper, with forced
alignment) are split into syllables and placed onto the predicted notes in order
of time; a syllable held over several notes gets `~` continuations.

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

In the GUI: Settings → Experimental Features → **Segmentation Model**.

`extract` is resumable — run it again to continue after an interruption. Use
`--limit N` for a quick test with the first N songs.

---

## Which songs make good training data

`extract` uses the same rules as the [chart benchmark](chart-benchmark.md): one
solo `.txt` per folder (no duets), at least 100 pitched notes, and the audio file
it refers to. Each chart is aligned to the sung pitch of the separated vocal; how
well it fits is stored per song, and `train` skips songs below `--min-ref-fit`
(default 0.5). Charts with sloppy timing teach the model sloppy timing, so the
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
* Pitch-change splitting and syllable merging are skipped when the model is
  used, because the model already decides where notes start and end.
