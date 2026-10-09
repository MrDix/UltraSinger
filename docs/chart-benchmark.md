# Chart Benchmark Tool (Experimental)

`tools/chart_benchmark.py` measures how close UltraSinger's generated charts come
to **hand-made reference charts** of the same songs. It answers the question
"did my pipeline change make the charts better?" without anybody having to sing.

Why not just use the game score? The game score only tells how well a voice hits
a chart. A chart that traces every wobble of the extracted vocal scores just as
high as a carefully charted one, yet is far harder to read and sing. Comparing
against reference charts measures what actually matters: are the notes in the
right place, with the right length and pitch, and are the right passages charted
at all?

> **Your data stays yours.** The tool works on your own UltraStar library. All
> samples, converted songs, plots and reports go into a work directory that must
> be **outside** this repository. It contains file names from your library and
> copies of your audio — never commit or share it.

---

## Quick start

```commandline
# 1) Pick a reproducible sample of songs from your library
uv run python tools/chart_benchmark.py sample "D:\Karaoke\Songs" "D:\ChartBench" --count 100

# 2) Convert every sampled song with the current pipeline (uses the GPU, ~1 min per song)
uv run python tools/chart_benchmark.py convert "D:\ChartBench" --label baseline

# 3) Measure the generated charts against the reference charts (+ piano-roll images)
uv run python tools/chart_benchmark.py evaluate "D:\ChartBench" --label baseline --plots

# 4) After a code or settings change: convert + evaluate under a new label, then compare
uv run python tools/chart_benchmark.py convert "D:\ChartBench" --label new --args "--chart_style score"
uv run python tools/chart_benchmark.py evaluate "D:\ChartBench" --label new
uv run python tools/chart_benchmark.py compare "D:\ChartBench" baseline new
```

`convert` is resumable: songs that already have a result are skipped. A song that
takes longer than `--timeout` seconds (default 3600, `0` = no limit) is recorded as
failed and the batch continues. All commands refuse a work directory inside the
repository. By default
only the generated TXT and the cached pitch data are kept per song (a few hundred
KB); pass `--keep-audio` to keep stems, audio and video as well.

---

## Which songs are used

`sample` walks the library and keeps every song folder that has

* exactly one UltraStar `.txt` (solo — duets are skipped),
* at least 100 pitched notes,
* the audio (`#AUDIO`/`#MP3`) or video (`#VIDEO`) file the chart refers to.

The sample is drawn with a fixed `--seed`, so the same library always yields the
same songs. `--prefer-video` converts from the video file when one exists, which
mimics converting a downloaded music video.

Each input file is **copied** into the work directory before conversion, so the
reference chart lying next to the original audio can never influence the result.
The copy is named `Artist - Title` after the chart's `#ARTIST`/`#TITLE`, like a
properly named download, because UltraSinger derives its metadata and lyrics
lookup from the input file name.

Extra UltraSinger arguments are passed with `--args`, e.g.
`--args --syllable_split` or `--args "--chart_style score"`. To forward a name the
benchmark uses itself (such as `--keep-audio` or `--timeout`), write `--args=...`.

The quality of your reference charts matters. `evaluate` aligns each reference
chart to the sung pitch of the separated vocal (it searches a time offset of up
to ±1.2 s) and reports how well the two fit (`ref_fit_pct`). Songs whose
reference fits worse than `--min-ref-fit` (default 50 %) are marked
`unreliable_reference` and left out of the summary — typically charts with bad
timing, a different song version, or a voice the pitch tracker cannot follow.

Some charts are duets flattened into one track that list the second voice
**after** the end of the song (the notes go on for about another song length
after the audio ends). `evaluate` detects this (at least 10 % of the pitched note
time starting after the end of the audio), fits the appended part onto the song
on its own and then scores the generated chart against **either voice**: a frame
counts as agreeing when the generated note matches one of them. A part that fits
the vocal worse than `--min-ref-fit` is dropped instead, as notes after the end
of the audio cannot be measured. Such songs carry `appended_voice`
(`folded`/`dropped`), `appended_offset_ms` and `appended_fit_pct` in the report.

---

## Metrics

Each song gets one row; the summary reports the **median** over reliable songs.
Octaves are folded everywhere, as the games ignore the octave when scoring.

| Metric | Meaning |
|---|---|
| `chart_agreement_pct` | **Primary metric.** Share of the reference's pitched note time on which the generated chart also has a note within ±1 semitone. Equals the score a singer who sings the reference perfectly would get on the generated chart (Medium). Generated notes where the reference has none do not lower it — see the precision. |
| `chart_precision_pct` | Share of the generated pitched note time that agrees with the reference (±1 semitone). Notes that run past the reference notes, or that chart backing vocals, lower the precision but not the agreement. |
| `chart_f1_pct` | Harmonic mean of agreement and precision — rewards charts that cover the reference *without* padding it. |
| `onset_hit_50_pct` / `onset_hit_100_pct` | Reference note starts that have a generated note start within 50 / 100 ms. |
| `onset_precision_100_pct` | Generated note starts that have a reference note start within 100 ms. Low values mean extra or split notes. |
| `note_count_ratio` | Generated / reference pitched notes (1.0 = same number). |
| `median_note_ms`, `short_notes_pct` | Generated note lengths (short = under 150 ms). |
| `pitch_agree_pct` | Where both charts have a note: pitch within ±1 semitone. |
| `ref_coverage_pct` | Reference note time covered by any generated note. |
| `extra_time_pct` | Generated note time where the reference has no pitched note (backing vocals, ad-libs, echoes). |
| `freestyle_charted_pct` | Reference freestyle/rap time the generated chart covers with pitched notes. |
| `oracle_pitch_pct` | Pitch accuracy the pitch tracker would reach with the reference's own note boundaries. If this is high while `chart_agreement_pct` is low, the loss is in note segmentation, not in pitch detection. |
| `vocal_hits_ref_pct` / `vocal_hits_gen_pct` | Share of sung frames inside notes that hit the reference / generated chart. Similar values do **not** mean similar chart quality — that is exactly why the game score is not used as the target. |

Reports are written to `<workdir>/reports/<label>.json` and `<label>.md`. Song IDs
in reports are anonymous (`song_001`, …); the mapping to library folders is kept
only in `<workdir>/songs.json`.

---

## Piano-roll images

With `--plots`, every song gets game-like images in
`<workdir>/runs/<label>/<song>/plots/`: the reference chart as filled green bars,
the generated chart as red outlines (dotted = freestyle) and the sung pitch of the
separated vocal as black dots, each row showing 12 seconds. Generated notes that
were folded by an octave are labelled `+12`/`-12`.
