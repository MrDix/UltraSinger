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

# 2) Once, if you have no segmentation model yet: train it from the same examples
#    (GPU: about 30 minutes for ~1000 songs, see segmentation-model.md)
uv run python tools/train_segmentation.py train "D:\SegTrain" --out "D:\SegTrain\model.pt"

# 3) Train the pitch context model (CPU, a few minutes for ~1000 songs)
uv run python tools/train_pitch_context.py --data "D:\SegTrain" --out "D:\SegTrain\pitch_context.npz"

# 4) Use both: the pitch context step only runs for the notes of a segmentation model
uv run python src/UltraSinger.py -i "song.mp3" --segmentation_model "D:\SegTrain\model.pt" ^
    --pitch_context balanced --pitch_context_model "D:\SegTrain\pitch_context.npz"
```

Step 2 is only for those without a segmentation model. Any segmentation model
you already use will do (`--segmentation_model` or
`--segmentation_model_repo`, see [segmentation-model.md](segmentation-model.md));
it does not have to be trained from the same examples as the pitch context
model.

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
* A note **the pitch refinement moved** stays where that put it. The
  refinement moves a note that scores badly to the tone the games' pitch
  detection hears in it, in about half of the cases by one semitone, and marks
  it. The model only knows notes at the pitch they were sung on: it would move
  about half of the notes the refinement moved by one semitone, most of them
  back to the sung pitch, and undo what the refinement was for (see below).
* Any other note that is **not at the pitch of its frames** stays as well:
  more than 1.5 semitones away from them, octaves aside.
* Skipped when the ptAKF refit (`--ptakf_refit`, which the `score` chart
  style switches on) made the notes: their pitches are meant to be the ones
  the games' pitch detection hears. The `score` chart style with
  `--disable_ptakf_refit` has no refit, so the step runs there; it also runs
  when the refit was asked for but could not run and kept the notes as they
  were (it says so in the console).
* It **fails open**: without a model file, with a file that is no model, without
  a pitch track or on any error, the conversion prints the reason and keeps
  the note pitches as they are. The settings info file
  (`--write_settings_info`) names the strength, the model file, how many note
  pitches moved and how many notes of the pitch refinement were left alone.

## What it gives

Measured on 100 songs with hand-made reference charts, with a model trained
on 1255 other songs (none of the 100, and no other edition of them). The
values are means over the songs; the change is against the same generated
charts without the step, in points.

| | without | `balanced` | `strong` |
|---|---|---|---|
| Reference note time with exactly the reference pitch | 68.6 % | +2.3 | +4.5 |
| ... within one semitone of it | 82.7 % | ±0.0 | −0.5 |
| ... within two semitones of it | 87.2 % | ±0.0 | ±0.0 |
| Game score of the original singing, Easy | 85.3 % | −0.1 | −0.2 |
| ... Medium | 79.0 % | −0.3 | −0.6 |
| ... Hard | 59.8 % | −0.4 | −1.0 |
| Notes moved | | 5.6 % | 11.1 % |

Songs that gain / lose more than 0.05 points:

| | `balanced` | `strong` |
|---|---|---|
| Exactly the reference pitch | 92 / 7 | 96 / 4 |
| Within one semitone | 47 / 38 | 10 / 84 |
| Within two semitones | 30 / 29 | 34 / 35 |
| Game score, Easy | 14 / 62 | 12 / 71 |
| Game score, Medium | 14 / 77 | 3 / 92 |
| Game score, Hard | 22 / 74 | 6 / 87 |

The **game score** is the one UltraSinger prints: the games' own pitch
detection on the separated vocal, scoring the original singing on the chart.
On 25 other songs, which have no reference chart, it changes like this:

| | without | `balanced` | `strong` |
|---|---|---|---|
| Easy | 87.9 % | −0.1 | −0.1 |
| Medium | 82.1 % | −0.2 | −0.4 |
| Hard | 68.4 % | −0.1 | −0.2 |

It is a trade, not a pure gain. The chart agrees more often with what a
charter writes, most of all in the exact pitch, which is what a chart looks
like on screen. A moved note leaves the pitch that was actually sung, so the
original singing scores a little less on it. Single songs can lose: of the
100 songs, the worst three lost 4.1, 0.8 and 0.3 points of exact agreement
with `balanced` and 7.5, 3.8 and 0.2 with `strong`; the best gained 10.8 and
13.2. Nine in ten songs gain 0.1 points or more with `balanced` and about
0.7 or more with `strong`.

The step was applied to the finished charts of these songs, with the notes
the pitch refinement had moved told from each chart and its vocal stem. For
78 of the 100 songs the note pitches come from the lead vocal, whose stem
was not kept: the vocal stem stands in for it there, also for the game
score. On the 22 songs where everything is as in a conversion, the numbers
are much the same: exact pitch +2.3 and +4.2, within one semitone +0.1 and
−0.4, game score on Medium −0.2 and −0.5.

### Why the notes of the pitch refinement stay

About one note in ten is moved by the pitch refinement, a good half of them
by one semitone. By its distance to the sung pitch, such a note looks like
one that was sung a little off, and the model moved about half of them, four
in five of those back to the pitch that was sung. Hand-made charts have these
notes on the sung pitch more often than on the refinement's, so that raised
the agreement (exact pitch +3.1 and +5.8 instead of +2.3 and +4.5). But it
undid what the refinement is for: the game score dropped by 0.2, 0.3 and 0.6
points (Easy, Medium, Hard) with `balanced` and by 0.3, 0.8 and 1.1 with
`strong`, instead of 0.1, 0.3 and 0.4 and 0.2, 0.6 and 1.0. A step that is
there to raise the game score should not be undone by one that runs after
it, so its notes stay. (With the pitch refinement switched off,
`--disable_refine_pitch`, no note carries its mark.)

Without any of these rules, the notes the refinement moved further away are
pulled back as well: 23 and 14 songs lose exact agreement, and on the 25
songs the game score on Hard drops by 1.1 and 1.2 points.

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
most of the gain of the one trained on 1255 (exact pitch +1.9 and +4.2
instead of +2.3 and +4.5 in the table above). More songs make it somewhat
better and more careful: it then moves fewer notes for a larger gain, and
costs less game score (on Hard 0.4 and 1.0 points instead of 0.8 and 1.3). Songs
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
