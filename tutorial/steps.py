"""
KMTNet ToO Pipeline -- step-by-step runner (onboarding layer)
=============================================================

This module is a thin, **non-destructive** friendly layer on top of the core
pipeline (``pipe/KMTNet_ToO_functions.py``).  It does NOT modify any of the
original code -- it simply exposes each of the 10 pipeline stages as a single,
clearly named function that you call with just the raw-data directory name.

Why this exists
---------------
The production entry point ``pipe/KMTNet_ToO_pipeline.py`` runs all 10 stages
inside one big ``ToO_pipeline()`` function.  To run a single stage you have to
pass toggle flags (``ampcompro=True, astrompro=False, ...``) and the file-name
regexes / path plumbing are hidden inside that function.  That makes it hard for
a newcomer to run, inspect, and *understand* one stage at a time.

Here, every stage is a one-liner:

    >>> import steps
    >>> steps.run_ampcom("example")      # stage 1
    >>> steps.run_astrom("example")      # stage 2
    >>> ...                              # and so on

The 10 stages, in order
-----------------------
    1.  run_ampcom       Amplifier (32 ext) -> 4 chip images (kk/mm/tt/nn)
    2.  run_astrom       Astrometric calibration (SExtractor + SCAMP)
    3.  run_astromqa     1st QA on chips (+ cosmic-ray / bleed / weight masks)
    4.  run_zpscale      Photometric zero-point scaling -> data/scaled/
    5.  run_bpmask       Bad-pixel-map update on scaled chips
    6.  run_stacking     Co-add dithers into field stacks (SWarp) -> data/stack/
    7.  run_stackqa      2nd QA on the stacked science images
    8.  run_catalog      Build calibrated source catalogue from each stack
    9.  run_subtraction  Difference imaging (HOTPANTS) -> transient candidates
    10. run_rbclass      Real/Bogus ML classification of the candidate cutouts

Convenience helpers
--------------------
    run_all(date)            run stages 1-10 in order
    make_subset(...)         build a small raw subset for a fast test
    list_outputs(date)       show what each stage produced
    show_fits(path)          quick-look display of a FITS image (matplotlib)
    list_steps()             print the ordered stage list

Command line
------------
    # run every stage on the "example" directory
    python steps.py example

    # run a single stage
    python steps.py example --step ampcom

    # resume from a stage to the end
    python steps.py example --from zpscale

    # forward a known-object list to the subtraction stage
    python steps.py example --known-obj S250206dm/S250206dm.csv
"""

from __future__ import annotations

import os
import re
import sys
import glob
import time
import contextlib

# --------------------------------------------------------------------------- #
# Locate the repository and make the core pipeline importable, no matter where
# this file is called from (notebook cwd, shell, etc.).
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_PIPE_DIR = os.path.join(_REPO_ROOT, "pipe")
for _p in (_REPO_ROOT, _PIPE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.working_directory_structure import (  # noqa: E402
    create_directories,
    path_base,
    path_cfg,
    path_cat,
    path_raw,
    path_scale,
    path_stack,
    path_subt,
    path_tmpl,
    path_plot,
)
import KMTNet_ToO_functions as pipe  # noqa: E402

# Default field/tiling table used everywhere in the pipeline.
GRID = "kmtnet_grid.fits"

# Ensure the base directory tree exists (safe to call repeatedly).
create_directories()

# Regexes copied verbatim from the production pipeline so behaviour matches.
_CHIP_RE = re.compile(r"(?P<serial>\d{6})\.(?P<chip>kk|mm|tt|nn)\.fits")
_SCALED_RE = re.compile(
    r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\."
    r"(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\."
    r"(?P<type>scaled)\.fits"
)
_STACK_RE = re.compile(
    r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\."
    r"(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.(?P<type>stack)\.fits"
)
_STACK_CAT_RE = re.compile(
    r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\."
    r"(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.(?P<type>stack)\.fits\.cat"
)
_STACKING_PATTERN = (
    r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\."
    r"(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\."
    r"(?P<type>scaled|mask)\.fits"
)


# --------------------------------------------------------------------------- #
# Small infrastructure helpers
# --------------------------------------------------------------------------- #
class _Dirs:
    """Resolved (and created) output directories for one observation run."""

    def __init__(self, date: str):
        self.date = date
        self.raw = os.path.join(path_raw, f"{date}/")     # chip images live here
        self.scaled = os.path.join(path_scale, f"{date}/")
        self.stack = os.path.join(path_stack, f"{date}/")
        self.subt = os.path.join(path_subt, f"{date}/")
        self.snap = os.path.join(self.subt, "snap/")
        for d in (self.raw, self.scaled, self.stack, self.subt, self.snap):
            os.makedirs(d, exist_ok=True)
            try:
                os.chmod(d, 0o777)
            except PermissionError:
                pass


@contextlib.contextmanager
def _workdir(path: str):
    """Run a block with cwd set to *path*, then restore.

    The core functions copy/link a few scratch files into the current working
    directory; we anchor that to the repository root so behaviour matches the
    production pipeline (and the scratch files are already git-ignored).
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _banner(n: int, name: str, date: str):
    print("=" * 70)
    print(f"  STAGE {n}/10  |  {name}  |  run = '{date}'")
    print("=" * 70)


def _chip_images(dirs: _Dirs):
    files = sorted(glob.glob(f"{dirs.raw}*.fits"))
    return [f for f in files if _CHIP_RE.match(os.path.basename(f))]


def _done(name: str, t0: float, n=None):
    extra = "" if n is None else f"  ({n} file(s))"
    print(f"--> {name} done in {time.time() - t0:.1f}s{extra}\n")


# --------------------------------------------------------------------------- #
# The 10 stages -- each takes only the run name (raw-data directory under
# data/raw/) and returns the directory holding its products.
# --------------------------------------------------------------------------- #
def run_ampcom(date: str, grid: str = GRID):
    """Stage 1 - Amplifier-to-chip combination.

    Reads the raw 32-extension ``kmt[asc].YYYYMMDD.NNNNNN.fits`` frames in
    ``data/raw/<date>/`` and writes four combined chip images per frame
    (``NNNNNN.kk/mm/tt/nn.fits``) into the same directory.  Frames that fail
    basic CCD / seeing / tracking checks are moved into ``bad*`` sub-folders.
    """
    _banner(1, "ampcom (amp -> chip)", date)
    dirs = _Dirs(date)
    t0 = time.time()
    with _workdir(path_base):
        pipe.ampcom(dirs.raw, path_cfg)
    _done("ampcom", t0, len(_chip_images(dirs)))
    return dirs.raw


def run_astrom(date: str, radius: float = 0.73, ithresh: int = 10, grid: str = GRID):
    """Stage 2 - Astrometric calibration (SExtractor + SCAMP).

    Solves the WCS of every chip image against the local Gaia-XP catalogue
    (UCAC-4 over the network as fallback) and writes the TPV solution into each
    chip header (plus ``.astrom.head`` side-cars).
    """
    _banner(2, "astrom (WCS solution)", date)
    dirs = _Dirs(date)
    t0 = time.time()
    with _workdir(path_base):
        pipe.astrom(dirs.raw, path_cfg, path_cat, radius=radius, ithresh=ithresh, gridcat=grid)
    _done("astrom", t0)
    return dirs.raw


def run_astromqa(date: str, grid: str = GRID):
    """Stage 3 - First quality assurance on the chips.

    For each chip this validates the astrometric solution (edge-focused QA),
    and builds the cosmic-ray / saturation-bleed / weight masks that later
    stages rely on.  A failure on one chip is isolated and never aborts the run.
    """
    _banner(3, "astromqa (chip QA + masks)", date)
    dirs = _Dirs(date)
    imgs = _chip_images(dirs)
    t0 = time.time()
    with _workdir(path_base):
        for img in imgs:
            try:
                pipe.qatest(
                    img, configdir=path_cfg, refcatdir=path_cat, refcatname="gaiaxp",
                    gridcat=grid, crreject=True, bleedreject=True, weightmap=True,
                    imtype="chip",
                )
            except Exception as e:
                print(f"*** astromqa failed for {os.path.basename(img)}: {e}. Skipping. ***")
            os.system(f"chmod 777 {dirs.raw}*mask.fits 2>/dev/null")
    _done("astromqa", t0, len(imgs))
    return dirs.raw


def run_zpscale(date: str, zpscaled: float = 30.0, grid: str = GRID):
    """Stage 4 - Photometric zero-point scaling.

    Scales every chip to a common zero-point (default 30 mag) and writes the
    results, with their masks, into ``data/scaled/<date>/``.
    """
    _banner(4, "zpscale (photometric scaling)", date)
    dirs = _Dirs(date)
    imgs = _chip_images(dirs)
    t0 = time.time()
    with _workdir(path_base):
        for img in imgs:
            try:
                outname = pipe.zpscale(
                    img, dirs.scaled, path_cfg, path_cat, path_plot,
                    zpscaled=zpscaled, figure=False, start=t0, gridcat=grid,
                )
            except Exception as e:
                print(f"*** zpscale failed for {os.path.basename(img)}: {e}. Skipping. ***")
                continue
            mask = img.replace(".fits", ".mask.fits")
            if outname is not None and os.path.exists(mask):
                os.rename(mask, os.path.join(dirs.scaled, outname.replace(".scaled.", ".mask.")))
    _done("zpscale", t0, len(glob.glob(f"{dirs.scaled}*.scaled.fits")))
    return dirs.scaled


def run_bpmask(date: str):
    """Stage 5 - Bad-pixel-map update on the scaled chips.

    Merges cosmic-ray, bad-pixel and bad-amplifier information into a single
    mask per scaled chip.
    """
    _banner(5, "bpmask (bad-pixel map)", date)
    dirs = _Dirs(date)
    files = sorted(glob.glob(f"{dirs.scaled}*.fits"))
    imgs = [f for f in files if _SCALED_RE.match(os.path.basename(f))]
    t0 = time.time()
    with _workdir(path_base):
        for img in imgs:
            try:
                pipe.BPM_update(img, path_cfg)
            except Exception as e:
                print(f"*** BPM update failed for {os.path.basename(img)}: {e}. Skipping. ***")
    _done("bpmask", t0, len(imgs))
    return dirs.scaled


def run_stacking(date: str, combinetype: str = "MEDIAN", grid: str = GRID):
    """Stage 6 - Image stacking (SWarp).

    Groups complete (kk, mm, tt, nn) dither sets by KMTNet grid field and
    co-adds them, re-projected onto the field's reference frame, into
    ``data/stack/<date>/`` (science + mask stacks).
    """
    _banner(6, "stacking (co-addition)", date)
    dirs = _Dirs(date)
    t0 = time.time()
    with _workdir(path_base):
        total = pipe.stacking(
            _STACKING_PATTERN, dirs.scaled, dirs.stack, path_cfg, path_tmpl,
            combinetype=combinetype, start=t0, gridcat=grid,
        )
    _done("stacking", t0, total)
    return dirs.stack


def run_stackqa(date: str, grid: str = GRID):
    """Stage 7 - Second QA, run on the stacked science images."""
    _banner(7, "stackqa (stack QA)", date)
    dirs = _Dirs(date)
    files = sorted(glob.glob(f"{dirs.stack}*.fits"))
    stacks = [f for f in files if _STACK_RE.match(os.path.basename(f))]
    t0 = time.time()
    with _workdir(path_base):
        for simg in stacks:
            try:
                pipe.qatest(
                    simg, configdir=path_cfg, refcatdir=path_cat, refcatname="gaiaxp",
                    gridcat=grid, crreject=False, bleedreject=False, weightmap=True,
                    imtype="stack",
                )
            except Exception as e:
                print(f"*** stack QA failed for {os.path.basename(simg)}: {e}. Skipping. ***")
    _done("stackqa", t0, len(stacks))
    return dirs.stack


def run_catalog(date: str):
    """Stage 8 - Calibrated source-catalogue generation from each stack."""
    _banner(8, "catalog (source catalogue)", date)
    dirs = _Dirs(date)
    cats = [c for c in sorted(glob.glob(f"{dirs.stack}*.cat"))
            if _STACK_CAT_RE.match(os.path.basename(c))]
    t0 = time.time()
    with _workdir(path_base):
        for cat in cats:
            try:
                pipe.catalogmaker(
                    cat, path_output=dirs.stack, path_cat=path_cat,
                    figure=False, start=t0, path_plot=path_plot,
                )
            except Exception as e:
                print(f"*** catalogmaker failed for {os.path.basename(cat)}: {e}. Skipping. ***")
    _done("catalog", t0, len(cats))
    return dirs.stack


def run_subtraction(date: str, detect: float = 1.5, known_obj: str | None = None):
    """Stage 9 - Difference imaging (HOTPANTS) and transient detection.

    Subtracts the reference template from each field stack, detects sources on
    the difference image, applies the artifact-flag filtering, and writes the
    transient catalogue plus candidate cutouts into ``data/subt/<date>/`` (and
    ``snap/``).

    *known_obj* is an optional CSV of targets (resolved under ``catalog/`` if
    not absolute) whose matching detections are always snapshotted.
    """
    _banner(9, "subtraction (difference imaging)", date)
    dirs = _Dirs(date)
    known_path = None
    if known_obj:
        known_path = known_obj if os.path.isabs(known_obj) else os.path.join(path_cat, known_obj)
        if not os.path.isfile(known_path):
            print(f"*** known-obj CSV not found: {known_path}. Continuing without it. ***")
            known_path = None
    files = sorted(glob.glob(f"{dirs.stack}*.fits"))
    stacks = [f for f in files if _STACK_RE.match(os.path.basename(f))]
    t0 = time.time()
    with _workdir(path_base):
        for simg in stacks:
            try:
                pipe.subtraction(
                    simg, path_ref=path_tmpl, path_cat=dirs.stack, path_refcat=path_tmpl,
                    path_output=dirs.subt, path_config=path_cfg, detect=detect,
                    known_obj=known_path,
                )
            except Exception as e:
                print(f"*** subtraction failed for {os.path.basename(simg)}: {e}. Skipping. ***")
    _done("subtraction", t0, len(glob.glob(f"{dirs.subt}*.new.*")))
    return dirs.subt


def run_rbclass(date: str):
    """Stage 10 - Real/Bogus ML classification of the candidate cutouts.

    Optional: requires the ``rbclass_kmtnet/`` model package.  Skipped cleanly
    if there are no cutouts or the model is unavailable.
    """
    import subprocess

    _banner(10, "rbclass (real/bogus)", date)
    dirs = _Dirs(date)
    t0 = time.time()
    if not os.path.isdir(dirs.snap) or len(os.listdir(dirs.snap)) == 0:
        print(f"No cutouts in {dirs.snap}. Skipping rbclass.")
        _done("rbclass", t0)
        return dirs.snap
    model_dir = os.path.join(path_base, "rbclass_kmtnet")
    if not os.path.isdir(model_dir):
        print(f"Model package {model_dir} not found. Skipping rbclass.")
        _done("rbclass", t0)
        return dirs.snap
    cmd = [
        "python", "rbclass_kmtnet/inference.py",
        "--dir_fits", dirs.snap,
        "--dir_ckpt", "rbclass_kmtnet/ckpt",
        "--ckpt_name",
        "model:OTrain_imsize:51_channels:rns_normalize:minmax_name:ri+ngi+gd_seed:0.bin",
    ]
    with _workdir(path_base):
        res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout if res.returncode == 0 else res.stderr)
    _done("rbclass", t0)
    return dirs.snap


# Ordered registry so run_all / the CLI / --from can iterate stages by name.
STEPS = [
    ("ampcom", run_ampcom),
    ("astrom", run_astrom),
    ("astromqa", run_astromqa),
    ("zpscale", run_zpscale),
    ("bpmask", run_bpmask),
    ("stacking", run_stacking),
    ("stackqa", run_stackqa),
    ("catalog", run_catalog),
    ("subtraction", run_subtraction),
    ("rbclass", run_rbclass),
]
_STEP_NAMES = [name for name, _ in STEPS]


def run_all(date: str, start_from: str | None = None, known_obj: str | None = None):
    """Run stages 1-10 in order (optionally resuming from *start_from*)."""
    names = _STEP_NAMES
    if start_from:
        if start_from not in names:
            raise ValueError(f"Unknown stage '{start_from}'. Choose from {names}.")
        names = names[names.index(start_from):]
    overall = time.time()
    for name, fn in STEPS:
        if name not in names:
            continue
        if name == "subtraction":
            fn(date, known_obj=known_obj)
        else:
            fn(date)
    print("#" * 70)
    print(f"#  ALL DONE for '{date}' in {time.time() - overall:.1f}s")
    print("#" * 70)


# --------------------------------------------------------------------------- #
# Convenience / inspection helpers (handy in the notebook)
# --------------------------------------------------------------------------- #
def make_subset(dst_date="quicktest0292", src_date="example",
                serials=("062875", "062876")):
    """Build a tiny raw subset for a fast end-to-end test.

    Symlinks just the requested raw frames from ``data/raw/<src_date>/`` into a
    fresh ``data/raw/<dst_date>/``.  The default pair maps to field 0292, which
    has a reference template, so the whole chain (incl. subtraction) runs in a
    few minutes instead of ~50.  Returns the new run name.
    """
    src = os.path.join(path_raw, src_date)
    dst = os.path.join(path_raw, dst_date)
    os.makedirs(dst, exist_ok=True)
    linked = []
    for serial in serials:
        matches = glob.glob(os.path.join(src, f"kmt?.????????.{serial}.fits"))
        if not matches:
            print(f"*** no raw frame for serial {serial} in {src} ***")
            continue
        target = os.path.realpath(matches[0])
        link = os.path.join(dst, os.path.basename(matches[0]))
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)
        linked.append(os.path.basename(link))
    print(f"Subset '{dst_date}' ready with {len(linked)} frame(s): {linked}")
    return dst_date


def list_outputs(date: str):
    """Print a one-line summary of what each stage has produced so far."""
    dirs = _Dirs(date)
    def _count(pat):
        return len(glob.glob(pat))
    print(f"Outputs for run '{date}':")
    print(f"  chips  (data/raw/{date}/)   : {_count(f'{dirs.raw}??????.??.fits')}")
    print(f"  scaled (data/scaled/{date}/): {_count(f'{dirs.scaled}*.scaled.fits')}")
    print(f"  stacks (data/stack/{date}/) : {_count(f'{dirs.stack}*.stack.fits')}")
    print(f"  diff   (data/subt/{date}/)  : {_count(f'{dirs.subt}*.new.*')}")
    print(f"  cutouts(data/subt/{date}/snap/): "
          f"{len(os.listdir(dirs.snap)) if os.path.isdir(dirs.snap) else 0}")
    cats = glob.glob(f"{dirs.subt}*.transient.cat")
    if cats:
        print(f"  transient catalogue(s): {[os.path.basename(c) for c in cats]}")


def show_fits(path, scale="zscale", cmap="gray", figsize=(7, 7), title=None):
    """Quick-look display of a FITS image (for notebooks).

    *scale* is 'zscale' (default), 'minmax', or a (vmin, vmax) tuple.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits

    data = fits.getdata(path)
    data = np.asarray(data, dtype=float)
    if scale == "zscale":
        try:
            from astropy.visualization import ZScaleInterval
            vmin, vmax = ZScaleInterval().get_limits(data)
        except Exception:
            vmin, vmax = np.nanpercentile(data, [5, 99])
    elif scale == "minmax":
        vmin, vmax = np.nanmin(data), np.nanmax(data)
    else:
        vmin, vmax = scale
    plt.figure(figsize=figsize)
    plt.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title or os.path.basename(str(path)))
    plt.tight_layout()
    plt.show()


def list_steps():
    """Print the ordered list of pipeline stages."""
    print("KMTNet ToO pipeline stages (in order):")
    for i, (name, fn) in enumerate(STEPS, 1):
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"  {i:2d}. {name:<12s} {doc}")


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #
def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="KMTNet ToO pipeline -- step-by-step runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("date", help="raw-data directory under data/raw/ (e.g. example)")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--step", choices=_STEP_NAMES, help="run only this stage")
    g.add_argument("--from", dest="start_from", choices=_STEP_NAMES,
                   help="run from this stage to the end")
    parser.add_argument("--known-obj", dest="known_obj", default=None,
                        help="CSV of known targets (relative to catalog/)")
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    args = parser.parse_args(argv)

    if args.list:
        list_steps()
        return
    if args.step:
        fn = dict(STEPS)[args.step]
        if args.step == "subtraction":
            fn(args.date, known_obj=args.known_obj)
        else:
            fn(args.date)
    else:
        run_all(args.date, start_from=args.start_from, known_obj=args.known_obj)


if __name__ == "__main__":
    _main()
