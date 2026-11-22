# Note Pitches from Their Musical Context (Experimental)

A note of the [segmentation model](segmentation-model.md) gets the pitch that
was **sung**: the 60th percentile of the confident pitch frames inside the
note. A hand-made chart writes the pitch the melody **means**. Mostly that is
the same, but not always: a note sung flat, scooped into from below or left
early still is the note of the scale it belongs to, and a charter writes that
one. Of the generated notes that sit one to one on a note of a hand-made
chart, 71 % have exactly its pitch and 16 % are one semitone off.

With `--pitch_context`, a second small model looks at every pitched note and
decides whether the charted pitch is one or two semitones beside the sung one.
The step is **off by default** and meant for comparing charts with and
without it.

> **No model is shipped.** You train your own on your UltraStar song library.
> The model is derived from your library — keep it private and outside this
> repository. The training tool refuses to write into the repository.

---

## Quick start

```commandline
# 1) Once, if not done already for the segmentation model: separate the vocals of
#    every song and store the pitch track and the chart's notes (GPU recommended)
uv run python tools/train_segmentation.py extract "D:\Karaoke\Songs" "D:\SegTrain"

# 2) Train the pitch context model (CPU, a few minutes for ~1000 songs)
uv run python tools/train_pitch_context.py --data "D:\SegTrain" --out "D:\SegTrain\pitch_context.npz"

# 3) Use it together with the segmentation model
uv run python src/UltraSinger.py -i "song.mp3" --segmentation_model "D:\SegTrain\model.pt" ^
    --pitch_context balanced --pitch_context_model "D:\SegTrain\pitch_context.npz"
```

In the GUI: Settings → Experimental Features → **Pitch Context Model** (the
file) and **Pitch Context** (Off / Balanced / Strong). Both are also in the
settings of a single song, so one song can be converted with and without it.

## What the model sees and decides

For every pitched note the model gets 40 numbers:

* **the note's pitch frames** relative to its pitch: their 10th to 90th
  percentiles, mean and spread, how many there are and how much of the note
  they cover, the share of them within half a semitone of each candidate pitch
  (two below to two above), and whether the pitch rises or falls over the note;
* **the note's length**;
* **the two pitched notes before and after it**: how many semitones away they
  are and how long the pause in between is;
* **the song**: how much of the note time is sung on the pitch class of each
  candidate pitch, whether each candidate lies in the major scale that holds
  most of the song's note time, how well and how clearly that scale fits, the
  note's step in it, and how far the singing is tuned off the semitones.

The notes around a note and the song are made of the notes that have at least
two confident pitch frames in the pitch track. A note without them has its
pitch from elsewhere (the full vocal stem, or the note before it): the
training leaves such notes out, and so does the conversion when it describes
the others.

For each shift of −2 to +2 semitones the model predicts two things: on what
share of the note's time the shifted pitch would be **within one semitone** of
a hand-made chart, and on what share it would be **exactly** on it. A note
takes the shift that promises most, when that is more than staying promises:

| Strength | Counts the exact pitch | Moves a note when a shift promises |
|---|---|---|
| `balanced` | 0.3 times as much as "within one semitone" | over 0.1 more than staying |
| `strong` | as much as "within one semitone" | anything more than staying |

The strengths are applied when a chart is converted, not when the model is
trained: one model file serves both.

## Where it runs

The step is the last one that changes pitches: after the octave corrections,
the pitch refinement and the optional split at vocal runs, before the golden
notes, the chart and the game score. It compares the notes with the pitch
track of the stem their pitches come from (the lead vocal when the
[lead-vocal pitch](segmentation-model.md) is used, else the vocal stem).

* Only the notes of a **segmentation model** are moved: the step is trained
  and measured on notes pitched that way. With word-based notes it does not run.
* **Freestyle and rap notes** are never touched, and they do not count as
  neighbours of a note.
* A note **keeps its octave**: it moves by at most two semitones from where it is.
* A note without at least two confident pitch frames stays as it is, and it
  is no neighbour of the notes around it.
* A note that is **not at the pitch of its frames** stays as it is: more than
  1.5 semitones away from them, octaves aside. Another step put it there: the
  pitch refinement moves a note to the tone the games' pitch detection hears
  in it. The model has seen no such note (in training every note has the
  pitch of its frames) and would mostly pull it back to the sung pitch, which
  costs game score and agreement with hand-made charts alike (see below).
* Skipped with the ptAKF refit (`--ptakf_refit`), which the `score` chart
  style switches on: its pitches are meant to be the ones the games' pitch
  detection hears. The `score` chart style with `--disable_ptakf_refit` has
  no refit, so the step runs there.
* It **fails open**: without a model file, with a file that is no model, without
  a pitch track or on any error, the conversion prints the reason and keeps
  the note pitches as they are. The settings info file
  (`--write_settings_info`) names the strength, the model file and how many
  note pitches moved.

## What it gives

Measured on 100 songs with hand-made reference charts, with a model trained
on 1255 other songs (none of the 100, and no other edition of them). The
values are means over the songs; the change is against the same generated
charts without the step, in points.

| | without | `balanced` | `strong` |
|---|---|---|---|
| Reference note time with exactly the reference pitch | 68.6 % | +3.1 | +5.8 |
| ... within one semitone of it | 82.7 % | +0.1 | −0.7 |
| ... within two semitones of it | 87.2 % | ±0.0 | ±0.0 |
| Sung pitch inside the notes exactly on the note | 68.6 % | −0.5 | −1.4 |
| ... within one semitone of it | 88.6 % | −0.4 | −0.9 |
| ... within two semitones of it | 94.2 % | −0.3 | −0.4 |
| Notes moved | | 8.5 % | 14.8 % |

Songs that gain / lose more than 0.05 points:

| | `balanced` | `strong` |
|---|---|---|
| Exactly the reference pitch | 97 / 3 | 97 / 3 |
| Within one semitone | 56 / 35 | 17 / 78 |
| Within two semitones | 42 / 34 | 39 / 38 |
| Sung pitch exactly on the note | 21 / 74 | 13 / 85 |
| Sung pitch within one semitone | 15 / 77 | 1 / 97 |
| Sung pitch within two semitones | 3 / 86 | 3 / 88 |

"Sung pitch inside the notes" is the share of the confident pitch frames of
the separated vocal inside the notes that lie within the tolerance of their
note (octaves ignored): what the original singer would score, as the pitch
tracker hears the singing.

The **game score** of the original singing, as UltraSinger prints it (the
games' own pitch detection on the vocal), was measured on 25 other songs
that have no reference chart:

| | without | `balanced` | `strong` |
|---|---|---|---|
| Easy | 87.9 % | −0.1 | −0.1 |
| Medium | 82.1 % | −0.2 | −0.6 |
| Hard | 68.4 % | −0.1 | −0.2 |

It is a trade, not a pure gain. The chart agrees more often with what a
charter writes, most of all in the exact pitch, which is what a chart looks
like on screen. A moved note leaves the pitch that was actually sung, so the
original singing scores a little less on it. Single songs can lose: of the
100 songs, the worst three lost 5.9, 1.6 and 0.5 points of exact agreement
with `balanced` and 8.8, 3.6 and 0.5 with `strong`; the best gained 12.8 and
17.1. Nine in ten songs gain about half a point or more with `balanced` and
about 1.2 points or more with `strong`.

Without the rule that leaves notes alone which are not at the pitch of their
frames, the same model gains less exact agreement (+2.0 and +5.1) and fares
worse within one semitone (−0.1 and −0.9), 23 and 14 songs lose exact
agreement instead of 3 and 3, and the game score drops by up to 1.1 points
(`balanced`, Hard) and 1.2 points (`strong`, Hard) instead of 0.2 and 0.6 at
most.

## Training

```commandline
uv run python tools/train_pitch_context.py --data "D:\SegTrain" --out "D:\SegTrain\pitch_context.npz"
```

`--data` is the work folder of `train_segmentation.py extract` (or its `data`
folder). The tool needs neither the audio nor a GPU: it reads the pitch track
and the hand-made chart of each example. Every charted note gets the pitch its
pitch frames give it, the way the conversion pitches a note, and the model
learns by how many semitones the charted pitch lies beside that one.

It prints what it learned from:

```text
1255 usable songs of 1300 (left out: 45 whose chart fits the vocal worse than 60%)
451256 notes; before any correction 76.4 % of their time is exactly on the charted pitch, 90.8 % within one semitone
measuring: training on 1129 songs, 126 held out
held-out songs (126), share of the note time on the charted pitch:
              exact pitch  within one semitone  notes moved
  unchanged        77.3 %               91.8 %        0.0 %
  balanced         81.9 %               92.7 %        9.7 %
  strong           85.5 %               91.9 %       17.2 %
training on all 1255 songs
model written to D:\SegTrain\pitch_context.npz (1.7 MB, 3000 trees, 334 s)
```

The held-out numbers come from a model trained without the held-out songs;
the model that is written is then trained on all songs. They are higher than
the gain on generated charts above: the held-out notes have the boundaries of
the hand-made chart, generated notes do not.

| Option | Default | Meaning |
|---|---|---|
| `--data` | | Folder of the training examples |
| `--out` | | Model file to write (outside the repository) |
| `--exclude` | | Examples to leave out: their names (`tr_00012`), or JSON files that list names |
| `--val-fraction` | 0.1 | Share of the songs held out for the measurement (0: no measurement) |
| `--min-ref-fit` | 0.6 | Skip songs whose chart fits the vocal worse than this |
| `--trees` | 300 | Trees per model output at most |
| `--limit` | 0 | Only use the first N examples (0 = all) |
| `--seed` | 0 | Seed of the held-out choice and of the training |

A small library is enough to start with: a model trained on 80 songs gave
most of the gain of the one trained on 1255 (exact pitch +2.5 and +5.3
instead of +3.1 and +5.8 in the table above). More songs make it somewhat
better and more careful; it then moves fewer notes for a larger gain. Songs
whose chart is poorly timed against the recording teach the model wrong
pitches; `--min-ref-fit` leaves them out.

### Measuring a model honestly

Songs of a [chart benchmark](chart-benchmark.md) sample must not be trained
on. `train_segmentation.py extract --exclude D:\ChartBench\songs.json` keeps
them out of the examples by their folder. A library often holds the same
chart a second time (another edition of the song, a duet version); such an
example can be left out by name with `--exclude` here.

## The model file

The model is a set of gradient-boosted regression trees, stored as plain
arrays of numbers in a numpy `.npz` archive of about 2 MB. Loading a model
file never runs code: nothing in it is unpickled, and every array is checked
before it is read: numbers of the expected kind, packed the way numpy packs
them, and exactly as much data as its header names. A model file may unpack
to 64 MB and hold two million nodes in twenty thousand trees at most (a real
one unpacks to about 5 MB and has about 200 000 nodes in 3000 trees), so a
crafted file can make the loader hold a few hundred megabytes at the very
most, and predicting with a model that loads stays within bounds. A pickle, an archive of another program or arrays that are no trees
are refused, and the conversion then goes on without the step. Predicting
needs only numpy, and the same notes and pitch track always give the same
result.

Training uses scikit-learn, which is installed with UltraSinger (librosa
depends on it), and exports its trees into that file. The
export is checked: the exported model must predict what scikit-learn
predicts, or no file is written.

A model file names the note features it was trained on. When a later version
of UltraSinger changes the features, an older model is refused with the hint
to train a new one.

## Options

| CLI | GUI | Default | Meaning |
|---|---|---|---|
| `--pitch_context off\|balanced\|strong` | Pitch Context | off | Strength of the step |
| `--pitch_context_model <file>` | Pitch Context Model | | The trained model file (`.npz`) |
