# Getting Started — KMTNet ToO Pipeline (new-member guide)

This `tutorial/` folder is a **friendly, step-by-step entry point** to the
KMTNet Target-of-Opportunity (ToO) reduction pipeline. It is built *on top of*
the production code in `pipe/` and does **not** modify any of it.

If you are new to the project, start here.

---

## 1. What's in this folder

| File | Purpose |
|------|---------|
| `GETTING_STARTED.md` | This guide. |
| `steps.py` | A thin runner that exposes each of the 10 pipeline stages as a single function (`run_ampcom`, `run_astrom`, …). Importable **and** runnable from the command line. |
| `KMTNet_ToO_tutorial.ipynb` | A guided notebook that runs every stage on a tiny example dataset, with explanations and quick-look images. |

The notebook is the best place to *learn*; `steps.py` is what you'll *reuse* in
your own scripts.

---

## 2. One-time setup

Every command below is run from the **repository root** — the directory holding
`pipe/`, `config/` and `data/`. Nothing in the pipeline depends on where that
directory lives, so `cd` to your own checkout:

```bash
cd /path/to/KMTNet_ToO       # your clone; on the SNU reduction server: /data8/kmtntoo
```

### 2.1 Python environment
On the SNU reduction server the `kmtnet` conda environment already exists:

```bash
conda activate kmtnet
```

Elsewhere, create one from the pinned list at the repository root:

```bash
conda create -n kmtnet python=3.9 && conda activate kmtnet
pip install -r requirements.txt
```

It provides astropy, astroquery, numpy, scipy, matplotlib, astroscrappy, etc.

### 2.2 External astronomy software
These are separate C programs, not pip packages. They must be on your `PATH`
(check with `which <name>`):

| Program | Used by | Needed for |
|---------|---------|------------|
| `source-extractor` (SExtractor) | stages 1–4, 7–9 | source detection / photometry |
| `scamp` | stage 2 | astrometric solution |
| `swarp` | stage 6 | image co-addition |
| `hotpants` | stage 9 | image subtraction |
| `psfex` *(optional)* | stage 9 | PSF modelling |

### 2.3 Reference data

| What | Where | Set up by |
|------|-------|-----------|
| SExtractor / SCAMP / SWarp configs | `config/` | shipped with the repo |
| Gaia-XP catalogues | `catalog/gaiaxp/` | downloaded on demand per field (see below) |
| Reference templates | `data/tmpl/` | **you** — see below |

`data/tmpl/` is the one path that usually points outside the repository. Make it
a symlink to wherever your reference stacks live:

```bash
ln -s /data8/KS4/database/stack data/tmpl        # SNU reduction server
```

If a symlink is not an option, drop a `config/local_settings.py` (git-ignored)
instead:

```python
path_tmpl = '/somewhere/else/stacks/'
```

Astrometry prefers a local Gaia-XP catalogue and downloads one automatically for
any field that has none, falling back to a UCAC-4 network query only if that
fails. Nothing to do by hand.

You never need to create output directories — `steps` makes them.

---

## 3. The 10 stages at a glance

```
 1. ampcom       raw 32-amp frame  ->  4 chip images (kk/mm/tt/nn)
 2. astrom       solve WCS per chip (SExtractor + SCAMP, Gaia-XP)
 3. astromqa     validate astrometry + build CR / bleed / weight masks
 4. zpscale      scale chips to a common zero-point  ->  data/scaled/
 5. bpmask       merge all defect maps into one bad-pixel mask
 6. stacking     co-add dithers into field stacks (SWarp)  ->  data/stack/
 7. stackqa      re-run QA on the stacked images
 8. catalog      calibrated source catalogue from each stack
 9. subtraction  difference imaging (HOTPANTS)  ->  transient candidates
10. rbclass      Real/Bogus ML classification of cutouts (optional)
```

Each stage reads the output of the previous one, so run them in order.

Data flows through these directories (all under the git-ignored `data/`):

```
data/raw/<run>/       raw frames + combined chips + masks    (stages 1-3)
data/scaled/<run>/    zero-point-scaled chips                (stages 4-5)
data/stack/<run>/     field stacks + catalogues              (stages 6-8)
data/subt/<run>/      difference images + transient catalogue (stage 9)
data/subt/<run>/snap/ candidate cutouts                      (stages 9-10)
```

`<run>` is simply the name of a directory under `data/raw/` (e.g. `example`).
It is the **only argument** every stage takes.

---

## 4. Quick start (5 minutes)

All three options assume you are at the repository root with `kmtnet` activated.

### Option A — Notebook (recommended for learning)
```bash
jupyter lab    # then open tutorial/KMTNet_ToO_tutorial.ipynb and run top to bottom
```
The notebook builds a 2-frame subset of field 0292 (which has a template) so the
whole chain finishes in a few minutes.

### Option B — Command line
```bash
# Build a small test set (2 frames of field 0292) and run everything on it:
python - <<'PY'
import sys; sys.path.insert(0, "tutorial")
import steps
steps.make_subset("quicktest0292", "example", ("062875", "062876"))
PY

python tutorial/steps.py quicktest0292            # run all 10 stages
```

### Option C — From Python / your own script
```python
import sys; sys.path.insert(0, "tutorial")
import steps

DATE = steps.make_subset("quicktest0292")   # or just DATE = "example"
steps.run_ampcom(DATE)
steps.run_astrom(DATE)
steps.run_astromqa(DATE)
# ... or simply:
steps.run_all(DATE)
```

---

## 5. `steps.py` cheat-sheet

```python
import steps

# --- run individual stages (1 argument: the run name) ---
steps.run_ampcom(date)
steps.run_astrom(date)
steps.run_astromqa(date)
steps.run_zpscale(date)
steps.run_bpmask(date)
steps.run_stacking(date)
steps.run_stackqa(date)
steps.run_catalog(date)
steps.run_subtraction(date, known_obj=None)   # optional known-target CSV
steps.run_rbclass(date)

# --- run several / all ---
steps.run_all(date)                      # stages 1-10
steps.run_all(date, start_from="stacking")

# --- helpers ---
steps.make_subset("quicktest0292", "example", ("062875", "062876"))
steps.list_outputs(date)     # what each stage has produced so far
steps.show_fits(path)        # quick-look display (matplotlib, zscale)
steps.list_steps()           # print the ordered stage list
```

Command-line form:
```bash
python tutorial/steps.py <run>                 # all stages
python tutorial/steps.py <run> --step ampcom   # one stage
python tutorial/steps.py <run> --from zpscale  # resume from a stage
python tutorial/steps.py <run> --known-obj S250206dm/S250206dm.csv
python tutorial/steps.py --list <run>          # list stages
```

---

## 6. How this relates to the production pipeline

The production entry point is `pipe/KMTNet_ToO_pipeline.py`, which runs all
stages inside one `ToO_pipeline()` call and is also the real-time `AUTO`
upload-watcher. Every stage defaults to on; pass a flag to skip one:

```python
from KMTNet_ToO_pipeline import ToO_pipeline
ToO_pipeline('250212_CTIO')                                  # everything
ToO_pipeline('250212_CTIO', ampcompro=False)                 # chips already built
ToO_pipeline('250212_CTIO', ampcompro=False, astrompro=False,
             astromqapro=False)                              # resume at zpscale
```

An unknown flag name raises `TypeError` immediately rather than being silently
ignored. On the command line:

```bash
python pipe/KMTNet_ToO_pipeline.py 250212_CTIO
python pipe/KMTNet_ToO_pipeline.py AUTO            # watch data/raw/ for uploads
python pipe/KMTNet_ToO_pipeline.py                 # interactive directory picker
```

`steps.py` calls the **exact same** core functions
(`pipe/KMTNet_ToO_functions.py`) with the **same arguments and file-name
patterns** — it just unbundles them into one clean function per stage and hides
the plumbing, so you can run and inspect each step independently. Behaviour is
identical; only the ergonomics differ.

For production / real-time use, keep using `pipe/KMTNet_ToO_pipeline.py`. For
learning, debugging, and re-running a single stage, use `tutorial/steps.py`.

---

## 7. Cleaning up a test run

Test runs are disposable. To remove one:

```python
import shutil, os, steps
DATE = "quicktest0292"
for sub in (steps.path_raw, steps.path_scale, steps.path_stack, steps.path_subt):
    shutil.rmtree(os.path.join(sub, DATE), ignore_errors=True)
```

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `command not found: source-extractor / scamp / swarp / hotpants` | External software not on `PATH`; activate `kmtnet` or load the right module. |
| `import steps` fails in the notebook | Run the first code cell — it adds the `tutorial/` folder to `sys.path`. |
| A stage prints `*** ... failed for <chip>: ... Skipping. ***` | One chip failed but the run continues by design. Check that chip's inputs; the rest still process. |
| `stacking` produces 0 stacks | A field needs a complete `(kk, mm, tt, nn)` set and a matching template in `data/tmpl/`. Check that `data/tmpl` is the symlink from §2.3 and not an empty directory — `create_directories()` will happily make one if the symlink is missing. |
| `subtraction` finds 0 candidates | Normal for a clean field/subset — inspect `data/subt/<run>/snap/` and the `*.transient.cat`. |
| `rbclass` fails on a missing checkpoint | The trained weights (~196 MB) are **not** in the repository. Copy `pipe/rbclass_kmtnet/ckpt/` from an existing deployment, or skip the stage with `rbclasspro=False`. |
| SExtractor aborts with `cannot open flag.fits` or `Incompatible FLAG-map size` | A `.param` file is requesting `IMAFLAGS_ISO` for a call that passes no `-FLAG_IMAGE`. Only `kmtnet_imask.param` and `kmtnet_psf.param` should request it; `kmtnet.param` must not. |

For the scientific detail of each stage, read the repository `README.md` and the
docstrings in `pipe/KMTNet_ToO_functions.py`.
