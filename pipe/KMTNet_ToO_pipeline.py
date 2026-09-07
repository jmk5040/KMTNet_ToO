#%% path defines
import time, os, sys, glob, re, copy, shutil, subprocess
import multiprocessing
from functools import partial

# Import path configuration from centralized config file
# This approach provides several benefits:
# 1. All paths are automatically relative to the repository root
# 2. Works regardless of where the user runs the script from
# 3. Centralized configuration - change paths in one place
# 4. Automatic directory creation - no manual setup required
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.working_directory_structure import *

# Create all necessary directories for the pipeline
# This ensures the directory structure exists before processing begins
paths = create_directories()
path_cfg = paths['path_config']
path_cat = paths['path_catalog']

path_raw = paths['path_raw']
path_scale = paths['path_scaled']
path_stack = paths['path_stack']
path_subt = paths['path_subt']
path_tmpl = paths['path_tmpl']

path_plot   = paths['path_plot']
path_log    = paths['path_log']

# Core budget for the whole reduction.
# Every stage divides this between worker processes and the thread count it
# hands to its external tools, so the two can never multiply past it. Raise it
# with --ncores, ToO_pipeline(ncores=...), or NCORES in config/local_settings.py.
# The machine has more cores than this on purpose: it is shared.
NCORES = 16

# Per-stage cap on concurrent worker processes. Stages that stream whole chip
# or stack images through memory stop scaling well before the core budget is
# spent -- see the measurements in split_budget(). Anything not listed here is
# limited only by NCORES and the number of work items.
STAGE_MAX_PROCS = {
    'astromqa': 8,      # 16 chips, budget 16: 8x2 = 268 s vs 16x1 = 395 s
}

try:
    from config.working_directory_structure import _local as _local_settings
    if _local_settings is not None:
        NCORES = int(getattr(_local_settings, 'NCORES', NCORES))
except Exception:
    pass

#%% KMTNet ToO Pipeline
from astropy.io import fits
from datetime import datetime
from astropy.table import Table, vstack
import KMTNet_ToO_functions as pipe
import logging

def _run_qatest(img, path_cfg, path_cat, field_info, threads):
    """One chip's astrometry QA. Module level so multiprocessing can pickle it.

    Failures are caught here rather than in the parent -- one bad chip should
    not take down the pool -- but the reason is returned so it reaches the run
    log instead of only the terminal.

    Returns (img, ok, reason).
    """
    try:
        pipe.qatest(img, configdir=path_cfg, refcatdir=path_cat, refcatname='gaiaxp',
                    gridcat=field_info, crreject=True, bleedreject=True,
                    weightmap=True, imtype='chip', threads=threads)
        return img, True, ''
    except Exception as e:
        print(f'*** astromqa failed for {os.path.basename(img)}: {e}. Skipping this chip. ***')
        return img, False, f'{type(e).__name__}: {e}' 


class RunLog:
    """Per-stage record of what happened, not just how long it took.

    The old log had three columns -- process, frames, time -- so a stage that
    quietly did nothing was indistinguishable from one that worked. Run 1 of
    250831_SAAO recorded `bpmaskpro 27` where 156 chips were expected and
    `stackingpro 0`, and nothing said why. Anything that went wrong had to be
    dug out of the terminal scrollback, which for an unattended run is gone.

    Two files are written next to each other:

      ToOprocess_<date>_<ts>.log   one row per stage: status, item counts,
                                   elapsed and cumulative seconds, a note
      ToOissues_<date>_<ts>.log    one row per item that failed or was skipped,
                                   with the reason

    `elapsed` is the stage's own cost; the old `time` column was cumulative, so
    reading a single stage's cost meant differencing consecutive rows by hand.
    Both are kept.
    """

    def __init__(self, path_summary, path_issues, start):
        self.path_summary = path_summary
        self.path_issues = path_issues
        self.start = start
        self.rows = []
        self.issues = []
        self._mark = start

    def skip(self, process):
        """A stage that was switched off."""
        self.rows.append(dict(process=process, status='off', items=0, ok=0,
                              failed=0, skipped=0, elapsed=0.0,
                              cumulative=round(time.time() - self.start, 2), note=''))
        self._mark = time.time()
        self.write()

    def record(self, process, items, failures=(), skips=(), note=''):
        """Close out a stage.

        failures : iterable of (item, reason) that errored
        skips    : iterable of (item, reason) deliberately not processed
        """
        now = time.time()
        failures, skips = list(failures), list(skips)
        ok = max(0, items - len(failures) - len(skips))
        if items == 0:
            status = 'empty'
        elif failures:
            status = 'failed' if ok == 0 else 'partial'
        elif skips:
            status = 'partial' if ok else 'skipped'
        else:
            status = 'done'
        self.rows.append(dict(
            process=process, status=status, items=items, ok=ok,
            failed=len(failures), skipped=len(skips),
            elapsed=round(now - self._mark, 2),
            cumulative=round(now - self.start, 2), note=note))
        for item, reason in failures:
            self.issues.append(dict(process=process, kind='FAILED',
                                    item=os.path.basename(str(item)), reason=str(reason)))
        for item, reason in skips:
            self.issues.append(dict(process=process, kind='SKIPPED',
                                    item=os.path.basename(str(item)), reason=str(reason)))
        self._mark = now
        self.write()
        line = (f"{process}: {status} -- {ok}/{items} ok"
                + (f", {len(failures)} failed" if failures else '')
                + (f", {len(skips)} skipped" if skips else '')
                + f" ({self.rows[-1]['elapsed']:.1f}s)")
        print(line)

    def write(self):
        if self.rows:
            Table(rows=self.rows, names=list(self.rows[0])).write(
                self.path_summary, format='ascii.fixed_width_two_line', overwrite=True)
        if self.issues:
            Table(rows=self.issues, names=list(self.issues[0])).write(
                self.path_issues, format='ascii.fixed_width_two_line', overwrite=True)

    def summary(self):
        nf = sum(r['failed'] for r in self.rows)
        ns = sum(r['skipped'] for r in self.rows)
        return (f"{len(self.rows)} stage(s), {nf} failed item(s), {ns} skipped item(s). "
                f"Details: {self.path_issues if self.issues else '(none)'}")


def _run_zpscale(img, path_output2, path_cfg, path_cat, path_plot, field_info, start, threads):
    """Zero-point scale one chip, then move its mask alongside the output."""
    try:
        pipe.set_thread_limits(threads)
        outname = pipe.zpscale(img, path_output2, path_cfg, path_cat, path_plot,
                               zpscaled=30.0, figure=False, start=start, gridcat=field_info,
                               threads=threads)
        if outname is not None and os.path.exists(img.replace('.fits', '.mask.fits')):
            os.rename(img.replace('.fits', '.mask.fits'),
                      os.path.join(path_output2, outname.replace('.scaled.', '.mask.')))
        return img, True, ''
    except Exception as e:
        print(f'*** zpscale failed for {os.path.basename(img)}: {e}. Skipping this chip. ***')
        return img, False, f'{type(e).__name__}: {e}'


def _run_stackqa(simg, path_cfg, path_cat, field_info, threads):
    try:
        pipe.qatest(simg, configdir=path_cfg, refcatdir=path_cat, refcatname='gaiaxp',
                    gridcat=field_info, crreject=False, bleedreject=False,
                    weightmap=True, imtype='stack', threads=threads)
        return simg, True, ''
    except Exception as e:
        print(f'*** stack QA failed for {os.path.basename(simg)}: {e}. Skipping. ***')
        return simg, False, f'{type(e).__name__}: {e}'


def _run_catalog(cat, path_output3, path_cat, path_plot, start, threads):
    try:
        pipe.set_thread_limits(threads)
        pipe.catalogmaker(cat, path_output=path_output3, path_cat=path_cat,
                          figure=False, start=start, path_plot=path_plot)
        return cat, True, ''
    except Exception as e:
        print(f'*** catalogmaker failed for {os.path.basename(cat)}: {e}. Skipping. ***')
        return cat, False, f'{type(e).__name__}: {e}'


def _run_subtraction(simg, path_tmpl, path_output3, path_output4, path_cfg, known_obj_path, threads):
    """Difference one stack.

    A missing reference template is reported separately from a genuine error:
    it is a fact about the data, not a fault in the run, and half of a typical
    ToO night lands on fields KS4 never covered.
    """
    try:
        pipe.set_thread_limits(threads)
        pipe.subtraction(simg, path_ref=path_tmpl, path_cat=path_output3,
                         path_refcat=path_tmpl, path_output=path_output4,
                         path_config=path_cfg, detect=1.5, known_obj=known_obj_path,
                         threads=threads)
        return simg, 'ok', ''
    except FileNotFoundError as e:
        print(f'*** subtraction skipped for {os.path.basename(simg)}: {e} ***')
        return simg, 'skip', str(e)
    except Exception as e:
        print(f'*** subtraction failed for {os.path.basename(simg)}: {e}. Skipping. ***')
        return simg, 'fail', f'{type(e).__name__}: {e}'


def _run_ampcom(chunk, path_output1, path_cfg, threads):
    """Combine amps into chips for one slice of the night's raw frames."""
    try:
        pipe.ampcom(path_output1, path_cfg, frames=chunk,
                    write_header=False, cleanup=False, threads=threads)
        return chunk, True, ''
    except Exception as e:
        print(f'*** ampcom failed for {len(chunk)} frame(s) starting {os.path.basename(chunk[0])}: {e} ***')
        return chunk, False, f'{type(e).__name__}: {e}'


def _run_astrom(chunk, path_output1, path_cfg, path_cat, field_info, threads):
    """Solve the WCS for one slice of the night's frames.

    astrom() walks whole frames (four chips each), so the split is by frame.
    Only the parent writes ToOastrom.txt and runs the closing directory-wide
    cleanup -- a worker doing either would clobber its siblings.
    """
    try:
        pipe.astrom(path_output1, path_cfg, path_cat, radius=0.73, ithresh=10,
                    gridcat=field_info, frames=chunk,
                    write_info=False, cleanup=False, threads=threads)
        return chunk, True, ''
    except Exception as e:
        print(f'*** astrom failed for {len(chunk)} frame(s) starting {os.path.basename(chunk[0])}: {e} ***')
        return chunk, False, f'{type(e).__name__}: {e}'


def _preflight():
    """Fail loudly if a file the pipeline depends on is missing.

    Several stages locate their inputs by plain string concatenation
    (f'{path_cfg}kmtnet.swarp', f'{path_cfg}badpixelmap/...', and ~40 more).
    When such a path is wrong the run does NOT crash: SWarp and SExtractor
    print a one-line warning and silently fall back to internal defaults, and
    BPM_update returns early. An unattended run then takes hours and produces
    scientifically wrong output that looks superficially fine.

    This check runs once at start-up and turns that silent degradation into an
    immediate, named failure.
    """
    required = [
        os.path.join(path_cfg, 'kmtnet.sex'),
        os.path.join(path_cfg, 'kmtnet.param'),
        os.path.join(path_cfg, 'kmtnet.conv'),
        os.path.join(path_cfg, 'kmtnet.nnw'),
        os.path.join(path_cfg, 'kmtnet.scamp'),
        os.path.join(path_cfg, 'kmtnet.swarp'),
        os.path.join(path_cfg, 'mask.swarp'),
        os.path.join(path_cfg, 'kmtnet.psfex'),
        os.path.join(path_cfg, 'kmtnet_psf.param'),
        os.path.join(path_cfg, 'kmtnet_imask.param'),
        os.path.join(path_cfg, 'kmtnet_novignet.param'),
        os.path.join(path_cfg, 'kmtnet_grid.fits'),
        os.path.join(path_cfg, 'badpixelmap'),
        os.path.join(path_cfg, 'ahead'),
        path_tmpl,
        path_cat,
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            'Pipeline pre-flight failed; these inputs are missing:\n  '
            + '\n  '.join(missing)
            + '\n\nA wrong path here would not crash the run -- SWarp/SExtractor fall back '
              'to internal defaults and BPM_update returns early -- so the pipeline stops now '
              'instead of producing silently wrong output.'
        )

    # The concatenation style the stages actually use, exercised on one file.
    probe = f'{path_cfg}kmtnet.swarp'
    if not os.path.exists(probe):
        raise FileNotFoundError(
            f'Pre-flight: string concatenation produced a bad path: {probe}\n'
            'path_cfg must end with a separator (see config/working_directory_structure.py).'
        )
    print('Pre-flight checks passed.')


def ToO_pipeline(date, field_info='kmtnet_grid.fits', known_obj=None, ncores=None, **steps):
    
    # process managements
    # Every stage defaults to ON, but individual stages can be toggled by the user via the `steps` argument. 
    _defaults = dict(
        ampcompro=True, 
        astrompro=True, 
        astromqapro=True, 
        zpscalepro=True,
        bpmaskpro=True, 
        stackingpro=True, 
        qa4stackpro=True, 
        catalogpro=True,
        subtpro=True, 
        rbclasspro=True,
    )

    # start of the process
    ncores = NCORES if ncores is None else max(1, int(ncores))
    start = time.time()
    print(f'KMTNet ToO Pipeline Starts for {date}.')
    print(f'Field/Tiling coordinate information referring to {field_info}.')
    print(f'Core budget: {ncores}')

    _preflight()

    # Optional list of known targets (e.g. gravitational-wave host-galaxy
    # candidates or already-known transients) whose matching detections must
    # always be snapshotted regardless of the artifact flags. The path is taken
    # relative to the catalog/ directory unless an absolute path is supplied.
    known_obj_path = None
    if known_obj:
        known_obj_path = known_obj if os.path.isabs(known_obj) else os.path.join(path_cat, known_obj)
        if os.path.isfile(known_obj_path):
            print(f'Known-object list: {known_obj_path}')
        else:
            print(f'*** known-obj CSV not found: {known_obj_path}. Continuing without forced snapshots. ***')
            known_obj_path = None
    
    unknown = set(steps) - set(_defaults)
    if unknown:
        raise TypeError(f'ToO_pipeline got unexpected step flag(s): {sorted(unknown)}')
    _defaults.update(steps)
    ampcompro   = _defaults['ampcompro']
    astrompro   = _defaults['astrompro']
    astromqapro = _defaults['astromqapro']
    zpscalepro  = _defaults['zpscalepro']
    bpmaskpro   = _defaults['bpmaskpro']
    stackingpro = _defaults['stackingpro']
    qa4stackpro = _defaults['qa4stackpro']
    catalogpro  = _defaults['catalogpro']
    subtpro     = _defaults['subtpro']
    rbclasspro  = _defaults['rbclasspro']

    process_status = {
        'ampcompro': ampcompro, 'astrompro': astrompro, 'astromqapro': astromqapro, 
        'zpscalepro': zpscalepro, 'bpmaskpro': bpmaskpro, 'stackingpro': stackingpro, 
        'qa4stackpro': qa4stackpro, 'catalogpro': catalogpro, 'subtpro': subtpro, 'rbclasspro': rbclasspro
    }

    # Print process status in a compact form
    for process, status in process_status.items():
        print(f"{process}:\t {'ON' if status else 'OFF'}")

    time.sleep(1)

    # make output directories
    path_output1= os.path.join(path_raw, f'{date}/') # where a*fits chip images will be located
    os.makedirs(path_output1, exist_ok=True)
    os.chmod(path_output1, 0o777)
    
    path_output2 = os.path.join(path_scale, f'{date}/') # where ToO*scaled.fits will be located
    os.makedirs(path_output2, exist_ok=True)
    os.chmod(path_output2, 0o777)

    path_output3 = os.path.join(path_stack, f'{date}/') # where ToO*stack.fits will be located
    os.makedirs(path_output3, exist_ok=True)
    os.chmod(path_output3, 0o777)

    path_output4 = os.path.join(path_subt, f'{date}/') # where conv.res.ToO*NxN.fits will be located
    os.makedirs(path_output4, exist_ok=True)
    os.chmod(path_output4, 0o777)
    
    path_output5 = os.path.join(path_output4, 'snap/') # where hd*.new|ref|sub.fits will be located
    os.makedirs(path_output5, exist_ok=True)
    os.chmod(path_output5, 0o777)

    # log file setting
    _ts         = datetime.fromtimestamp(start).strftime('%Y-%m-%d_%H:%M:%S')
    LOGname     = f"{path_log}ToOprocess_{date}_{_ts}.log"
    ISSUEname   = f"{path_log}ToOissues_{date}_{_ts}.log"
    runlog      = RunLog(LOGname, ISSUEname, start)
    # Every stage fills these before its runlog.record(); they are reset here so
    # a stage that reports nothing cannot inherit the previous stage's list.
    _failed, _skipped = [], []
    
    # pipelines
    if ampcompro:
        _failed, _skipped = [], []

        _frames = sorted(os.path.basename(f) for f in glob.glob(f'{path_output1}kmt*.fits'))
        nproc, nthread = pipe.split_budget(len(_frames), ncores,
                                           max_procs=STAGE_MAX_PROCS.get('ampcompro'))
        print(f'#\tampcom: {len(_frames)} frame(s), {nproc} process(es) x {nthread} thread(s)')
        pipe._write_ampcom_header(f'{path_output1}ToOampcom.cat')
        chunks = [_frames[i::nproc] for i in range(nproc)]
        chunks = [c for c in chunks if c]
        _ac = partial(_run_ampcom, path_output1=path_output1, path_cfg=path_cfg, threads=nthread)
        if len(chunks) <= 1:
            results = [_ac(c) for c in chunks]
        else:
            with multiprocessing.Pool(processes=len(chunks)) as pool:
                results = pool.map(_ac, chunks)
        _failed = [(c[0], why) for c, ok, why in results if not ok]
        os.system(f'chmod 777 {path_output1}*')
        for _d in ('badccderror', 'badseeing', 'badtracking'):
            os.system(f'chmod 777 {path_output1}{_d}/* 2>/dev/null')
        os.system(f'rm -f {path_output1}??????.??.ampcom.cat')
        runlog.record('ampcompro', len(glob.glob(f'{path_output1}kmt*.fits')),
                      failures=_failed, skips=_skipped)
    
    endampcom = time.time()
    time.sleep(0.1)
    print(f'Amp to chip combine process done.\t {endampcom-start:.2f}sec')

    if astrompro:
        _failed, _skipped = [], []

        _frames = sorted(os.path.basename(f) for f in glob.glob(f'{path_output1}kmt*.fits'))
        nproc, nthread = pipe.split_budget(len(_frames), ncores,
                                           max_procs=STAGE_MAX_PROCS.get('astrompro'))
        print(f'#\tastrom: {len(_frames)} frame(s), {nproc} process(es) x {nthread} thread(s)')
        if _frames:
            # Build the night's table once, here, before any worker starts.
            pipe.astrom(path_output1, path_cfg, path_cat, radius=0.73, ithresh=10,
                        gridcat=field_info, write_info=True, cleanup=False,
                        collect_only=True)
        # Round-robin rather than contiguous blocks: frames of one field sit
        # together in the listing and share a Gaia-XP catalogue, so dealing them
        # out spreads the reference-catalogue reads across workers.
        chunks = [_frames[i::nproc] for i in range(nproc)]
        chunks = [c for c in chunks if c]
        _ast = partial(_run_astrom, path_output1=path_output1, path_cfg=path_cfg,
                       path_cat=path_cat, field_info=field_info, threads=nthread)
        if len(chunks) <= 1:
            results = [_ast(c) for c in chunks]
        else:
            with multiprocessing.Pool(processes=len(chunks)) as pool:
                results = pool.map(_ast, chunks)
        _failed = [(c[0], why) for c, ok, why in results if not ok]
        os.system(f'chmod 777 {path_output1}*')
        os.system(f'rm -f {path_output1}??????.??.astrom.cat')
        runlog.record('astrompro', len(Table.read(f'{path_output1}ToOastrom.txt', format ='ascii')),
                      failures=_failed, skips=_skipped)

    endastrom = time.time()
    time.sleep(0.1)
    print(f'Astrometry process done.\t {endastrom-start:.2f}sec')

    if astromqapro:
        _failed, _skipped = [], []
        
        regex = re.compile(r"(?P<serial>\d{6})\.(?P<chip>kk|mm|tt|nn)\.fits")
        all_files   = sorted(glob.glob(f'{path_output1}*.fits'))
        imgs   = [file for file in all_files if regex.match(os.path.basename(file))]
        # Chips are independent, so this fans out over them. astroscrappy's
        # thread scaling falls off hard (95.1 s -> 25.4 s going from 1 to 8
        # threads on a real chip), which is why the budget goes to processes
        # first; qatest() pins its own threading layers from the `threads` it
        # is given. The chmod moved out of the loop -- it used to fire one
        # shell per image, each globbing every mask in the directory.
        nproc, nthread = pipe.split_budget(len(imgs), ncores,
                                           max_procs=STAGE_MAX_PROCS.get('astromqa'))
        print(f'#\tastromqa: {len(imgs)} chip(s), {nproc} process(es) x {nthread} thread(s)')
        _qa = partial(_run_qatest, path_cfg=path_cfg, path_cat=path_cat,
                      field_info=field_info, threads=nthread)
        if nproc == 1:
            results = [_qa(img) for img in imgs]
        else:
            with multiprocessing.Pool(processes=nproc) as pool:
                results = pool.map(_qa, imgs)
        _failed = [(im, why) for im, ok, why in results if not ok]
        if imgs:
            os.system(f'chmod 777 {path_output1}*mask.fits')
        
        runlog.record('astromqapro', len(imgs),
                      failures=_failed, skips=_skipped)

    endastrom2 = time.time()
    time.sleep(0.1)
    print(f'1st Astrometry QA process done.\t {endastrom2-start:.2f}sec')

    if zpscalepro:
        _failed, _skipped = [], []

        regex = re.compile(r"(?P<serial>\d{6})\.(?P<chip>kk|mm|tt|nn)\.fits")
        all_files   = sorted(glob.glob(f'{path_output1}*.fits')) # ToOampcom.cat should be located
        imgs   = [file for file in all_files if regex.match(os.path.basename(file))]
        
        nproc, nthread = pipe.split_budget(len(imgs), ncores,
                                           max_procs=STAGE_MAX_PROCS.get('zpscalepro'))
        print(f'#\tzpscale: {len(imgs)} chip(s), {nproc} process(es) x {nthread} thread(s)')
        _zp = partial(_run_zpscale, path_output2=path_output2, path_cfg=path_cfg,
                      path_cat=path_cat, path_plot=path_plot, field_info=field_info,
                      start=start, threads=nthread)
        if nproc == 1:
            results = [_zp(img) for img in imgs]
        else:
            with multiprocessing.Pool(processes=nproc) as pool:
                results = pool.map(_zp, imgs)
        _failed = [(im, why) for im, ok, why in results if not ok]
        runlog.record('zpscalepro', len(imgs),
                      failures=_failed, skips=_skipped)

    endzpscale     = time.time()
    time.sleep(0.1)
    print(f'Photometric ZP scaling process done.\t {endzpscale-start:.2f}sec')

    if bpmaskpro:
        _failed, _skipped = [], []

        regex = re.compile(r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\.(?P<type>scaled)\.fits")
        all_files = sorted(glob.glob(f'{path_output2}*.fits'))
        imgs   = [file for file in all_files if regex.match(os.path.basename(file))]

        for img in imgs:
            try:
                pipe.BPM_update(img, path_cfg)
            except Exception as e:
                print(f'*** BPM update failed for {os.path.basename(img)}: {e}. Skipping this chip. ***')
        
        runlog.record('bpmaskpro', len(imgs),
                      failures=_failed, skips=_skipped)

    endbpmask    = time.time()
    time.sleep(0.1)
    print(f'Badpixel masking process done.\t {endbpmask-start:.2f}sec')       

    if stackingpro:
        _failed, _skipped = [], []

        pattern = r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\.(?P<type>scaled|mask)\.fits"
        # stacking() walks (observatory, field, band) internally, so the budget
        # goes to SWarp's own threads here rather than to worker processes.
        #
        # MEM_MAX stays at the shipped 256 MB. Raising it to 4096 looked like an
        # obvious win -- a 22000x22000 coadd on 256 MB should be swapping -- but
        # measured on one real stack it bought 2% (386.3 s -> 378.8 s) for 13x
        # the resident memory (0.3 GB -> 4.0 GB peak), and the output was
        # pixel-identical. On a machine shared with other users that trade is
        # not worth taking.
        _sw_threads = max(1, ncores)
        _memmax     = 256
        print(f'#\tstacking: SWarp with {_sw_threads} thread(s), MEM_MAX {_memmax} MB')
        total   = pipe.stacking(pattern, path_output2, path_output3, path_cfg, path_tmpl,
                                combinetype='MEDIAN', start=start, gridcat=field_info,
                                threads=_sw_threads, memmax=_memmax)
        runlog.record('stackingpro', total,
                      failures=_failed, skips=_skipped)

    endstack    = time.time()
    time.sleep(0.1)
    print(f'Image stacking process done.\t {endstack-start:.2f}sec')       

    if qa4stackpro:
        _failed, _skipped = [], []
        
        regex = re.compile(r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.(?P<type>stack|mstack)\.fits")
        all_files   = sorted(glob.glob(f'{path_output3}*.fits'))
        stackimgs = [file for file in all_files if regex.match(os.path.basename(file)) and regex.match(os.path.basename(file)).group('type') == 'stack']
        nproc, nthread = pipe.split_budget(len(stackimgs), ncores,
                                           max_procs=STAGE_MAX_PROCS.get('qa4stackpro'))
        print(f'#\tqa4stack: {len(stackimgs)} stack(s), {nproc} process(es) x {nthread} thread(s)')
        _sq = partial(_run_stackqa, path_cfg=path_cfg, path_cat=path_cat,
                      field_info=field_info, threads=nthread)
        if nproc == 1:
            results = [_sq(si) for si in stackimgs]
        else:
            with multiprocessing.Pool(processes=nproc) as pool:
                results = pool.map(_sq, stackimgs)
        _failed = [(im, why) for im, ok, why in results if not ok]

        runlog.record('qa4stackpro', len(stackimgs),
                      failures=_failed, skips=_skipped)
            
    endqa = time.time()
    time.sleep(0.1)
    print(f'2nd Astrometry QA process done.\t {endqa-start:.2f}sec')

    if catalogpro:
        _failed, _skipped = [], []

        regex = re.compile(r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.(?P<type>stack)\.fits\.cat")
        all_cats= sorted(glob.glob(f'{path_output3}*.cat'))
        cats    = [file for file in all_cats if regex.match(os.path.basename(file))]
        nproc, nthread = pipe.split_budget(len(cats), ncores,
                                           max_procs=STAGE_MAX_PROCS.get('catalogpro'))
        print(f'#\tcatalog: {len(cats)} catalogue(s), {nproc} process(es) x {nthread} thread(s)')
        _cm = partial(_run_catalog, path_output3=path_output3, path_cat=path_cat,
                      path_plot=path_plot, start=start, threads=nthread)
        if nproc == 1:
            results = [_cm(c) for c in cats]
        else:
            with multiprocessing.Pool(processes=nproc) as pool:
                results = pool.map(_cm, cats)
        _failed = [(c, why) for c, ok, why in results if not ok]

        runlog.record('catalogpro', len(cats),
                      failures=_failed, skips=_skipped)
            
    endcatalog = time.time()
    time.sleep(0.1)
    print(f"Catalog making process done.\t {endcatalog-start:.2f}sec")

    if subtpro:
        _failed, _skipped = [], []

        # subtraction with hotpants
        regex = re.compile(r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.stack\.fits")
        all_files   = sorted(glob.glob(f'{path_output3}*.fits'))
        stackimgs   = [file for file in all_files if regex.match(os.path.basename(file))]
        nproc, nthread = pipe.split_budget(len(stackimgs), ncores,
                                           max_procs=STAGE_MAX_PROCS.get('subtpro'))
        print(f'#\tsubtraction: {len(stackimgs)} stack(s), {nproc} process(es) x {nthread} thread(s)')
        _sb = partial(_run_subtraction, path_tmpl=path_tmpl, path_output3=path_output3,
                      path_output4=path_output4, path_cfg=path_cfg,
                      known_obj_path=known_obj_path, threads=nthread)
        if nproc == 1:
            results = [_sb(si) for si in stackimgs]
        else:
            with multiprocessing.Pool(processes=nproc) as pool:
                results = pool.map(_sb, stackimgs)
        _failed  = [(im, why) for im, st, why in results if st == 'fail']
        _skipped = [(im, why) for im, st, why in results if st == 'skip']
        
        runlog.record('subtpro', len(stackimgs),
                      failures=_failed, skips=_skipped)
        
    endsubt     = time.time()
    time.sleep(0.1)
    print(f'Image subtraction process done.\t {endsubt-start:.2f}sec')
    
    if rbclasspro:
        _failed, _skipped = [], []
        
        # This stage now both scores the candidates and writes the surviving
        # snapshots. inference_cutout.py cuts each candidate's 51x51 window out
        # of the three full-frame images in memory, so subtraction() no longer
        # writes tens of thousands of stamps that only a handful are ever
        # looked at. Its input is the transient catalogue, not a stamp
        # directory.
        _pipedir = os.path.dirname(os.path.abspath(__file__))
        trcats = sorted(glob.glob(f'{path_output4}*.transient.cat'))
        if trcats:
            for trcat in trcats:
                command = [
                    sys.executable, os.path.join(_pipedir, 'rbclass_kmtnet', 'inference_cutout.py'),
                    '--transient_cat', trcat,
                    '--outdir', path_output5,
                    '--dir_ckpt', os.path.join(_pipedir, 'rbclass_kmtnet', 'ckpt'),
                    '--ckpt_name', 'model:OTrain_imsize:51_channels:rns_normalize:minmax_name:ri+ngi+gd_seed:0.bin',
                    '--snap_thresh', '0.5',
                    '--num_workers', '8',
                ]
                result = subprocess.run(command, capture_output=True, text=True, cwd=_pipedir)
                if result.returncode == 0:
                    print(f"inference_cutout.py executed successfully for {os.path.basename(trcat)}.")
                    print(result.stdout)
                else:
                    print(f"inference_cutout.py execution failed for {os.path.basename(trcat)}.")
                    print(result.stderr)
        else:
            print(f"No transient catalogue found in {path_output4}. Skipping rbclasspro subprocess.")

        # Number of candidates actually scored -- the stage's real workload.
        # Counting snapshot files would now under-report it by ~180x, since only
        # the survivors get written.
        _rbscore = os.path.join(path_output5, 'rbscore.csv')
        try:
            with open(_rbscore) as _f:
                _n_scored = max(sum(1 for _ in _f) - 1, 0)
        except OSError:
            _n_scored = 0
        runlog.record('rbclasspro', _n_scored,
                      failures=_failed, skips=_skipped,
                      note='candidates scored')

    endrb   = time.time()
    time.sleep(0.1)
    print(f'Real/Bogus classification process done.\t {endrb-start:.2f}sec')

    runlog.write()
    print(runlog.summary())

    # end of process (LOG saving)
    try:
        LOG.write(LOGname, format='ascii', overwrite=True)
        print(f"KMTNet Reduction Pipeline for {date} Has Done. \nLog file location: {LOGname}")
    except NameError:
        pass
    
    return

#%% KMTNet ToO Pipeline
import shutil
import argparse
import multiprocessing
from collections import defaultdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TooWatcher(FileSystemEventHandler):

    # basics
    def __init__(self, watch_directory, ncores, known_obj=None):
        self.watch_directory    = watch_directory
        self.ncores             = ncores
        self.known_obj          = known_obj
        self.pool               = multiprocessing.Pool(processes=ncores)

    # creation checking sequences
    def on_created(self, event, max_wait_time=600, wait_interval=5):
        
        # KMTNet image check (unimpaired)
        pattern = r'kmt[asc]\.\d{8}\.\d{6}\.fits'
        locations = {
            "kmts"  : "SAAO",
            "kmtc"  : "CTIO",
            "kmta"  : "SSO"}
        file_sizes = {
            "SAAO"  : 1361664000,
            "CTIO"  : 1361583360,
            "SSO"   : 1361583360}

        # file path check
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        if not os.path.isfile(file_path): return # this event is not a file upload
        
        # Log the detection of a new file
        logging.info(f"Detected new file: {file_name} at {file_path}")
        
        # observatory check (SAAO, SSO, CTIO)
        try:
            prefix = re.search(r"(kmts|kmta|kmtc)", file_name).group(0)
        except AttributeError:
            prefix = 'kmta'
        location = locations.get(prefix)

        # process time check
        start_time = time.time()
        last_move_time = time.time()
        
        # file completeness & move file
        while True:
            
            try:
                file_size = os.path.getsize(file_path)
            except FileNotFoundError: # temporary file name get changed after completion.
                match   = re.search(pattern, file_path) # assuming file name is involved in the temp name (e.g. .kmts.20230308.061242.fits.7uUd6N.)
                if match:
                    file_name   = match.group()
                    file_path   = os.path.join(self.watch_directory, file_name)
                    file_size   = os.path.getsize(file_path)
                else:
                    print(f"Lost File {file_path}")
                    return

            # Log file size check
            # logging.info(f"Checking file size for {file_name}: {file_size} bytes")

            if file_size >= file_sizes[location]:
                logging.info(f"File upload complete: {file_name} ({file_size} bytes)")
                time.sleep(1)
                # Check if the file name matches the expected pattern
                if not re.match(pattern, file_name):
                    print(f"Invalid file name {file_name}. Skipping.")
                    return
                print(f'New upload file detection: {os.path.basename(file_path)}')
                # self.move_to_directory(file_path, location, self.watch_directory, self.uploads_directory)
                last_move_time  = time.time()
                break
            elif time.time() - start_time > max_wait_time:
                print(f"Timeout reached for {file_name}. Skipping.")
                break

            time.sleep(wait_interval)

        # file set completeness & pipeline run
        while True:
            if time.time() - last_move_time > max_wait_time:
                print(f'{max_wait_time}sec have passed since the last image upload. \nKMTNet reduction pipeline will be running (n_core={self.ncores}).')
                
                pattern = r'kmt[asc]\.\d{8}\.\d{6}\.fits'
                alluploads = [os.path.basename(entry.path) for entry in os.scandir(watch_directory) if entry.is_file() and re.compile(pattern).match(os.path.basename(entry.path))]

                # classify the upload files
                for (date, site), files in self.group_files_by_date_and_site(alluploads).items():
                    process_directory   = f'{self.watch_directory}{date}_{locations.get(site)}'
                    # files to directory (yyyymmdd_SITE)
                    os.makedirs(process_directory, exist_ok=True)
                    for file in files:
                        src = os.path.join(self.watch_directory, file)
                        dst = os.path.join(process_directory, file)
                        if os.path.exists(dst):
                            os.remove(dst)  # Remove the existing file
                        shutil.move(src, dst)
                    # run the pipeline
                    if self.ncores == 1:
                        ToO_pipeline(os.path.basename(process_directory), known_obj=self.known_obj)
                    else:
                        self.pool.apply_async(ToO_pipeline, args=(os.path.basename(process_directory),), kwds={'known_obj': self.known_obj}) # working directories are defined inside the function
                break
            elif self.is_new_file_generated(self.watch_directory, last_move_time):
                break
            else:
                time.sleep(wait_interval)

    # new "additional" file detections
    def is_new_file_generated(self, watch_directory, last_check_time):
        for file_name in os.listdir(watch_directory):
            file_path = os.path.join(watch_directory, file_name)
            if os.path.isfile(file_path):
                pattern = r'kmt[asc]\.\d{8}\.\d{6}\.fits'
                if re.match(pattern, file_name):
                    file_creation_time = os.path.getctime(file_path)
                    if file_creation_time > last_check_time:
                        # print('New file detected.')
                        return True
        return False

    # grouping newly uploaded file sets by dates and sites
    def group_files_by_date_and_site(self, file_list):

        # Define a regex pattern to extract the site and date information
        file_pattern = re.compile(r'(?P<site>kmt[asc])\.(?P<date>\d{8})\.\d{6}\.fits')

        # Create a defaultdict to store the grouped files
        grouped_files = defaultdict(list)

        # Iterate through the file list and group them by date and site
        for file in file_list:
            match = file_pattern.match(file)
            if match:
                site = match.group('site')
                date = match.group('date')
                grouped_files[(date, site)].append(file)
        
        return grouped_files
    
    def wait_for_file_transfer(file_path, check_interval=5, stable_time=10):
        
        previous_size = -1
        stable_count = 0

        while True:
            try:
                current_size = os.path.getsize(file_path)
            except FileNotFoundError:
                time.sleep(check_interval)
                continue

            if current_size == previous_size:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count * check_interval >= stable_time:
                break

            previous_size = current_size
            time.sleep(check_interval)
            
    def close_pool(self):
        self.pool.close()
        self.pool.join()

#%% main program
if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="KMTNet_ToO_pipeline.py")
    parser.add_argument('date', nargs='?', default=None,
                        help="Data directory under raw/ to process (e.g. 250212_CTIO), "
                             "or 'AUTO' for upload monitoring. Omit for interactive selection.")
    parser.add_argument('--known-obj', dest='known_obj', default=None, metavar='CSV',
                        help="CSV of known targets (columns: Name, RA, Dec, and optional per-row "
                             "'radius' in arcsec), given relative to the catalog/ directory. Any "
                             "transient candidate matching a target (default radius 2\") always gets "
                             "a snapshot regardless of its flags, tagged with the target name in the header.")
    parser.add_argument('--ncores', type=int, default=None, metavar='N',
                        help=f"Total cores the reduction may use (default {NCORES}). Each stage "
                             "splits this between worker processes and the thread count handed to "
                             "SExtractor/SCAMP/SWarp/PSFEx/astroscrappy, so they never multiply out "
                             "past it.")
    args = parser.parse_args()

    watch_directory     = path_raw
    ncores              = 1

    # A directory passed on the command line runs non-interactively; otherwise the
    # user is prompted to choose one (the original interactive behaviour).
    if args.date is not None:
        user_input = args.date
    else:
        data_dirs   = sorted([d for d in os.listdir(watch_directory) if os.path.isdir(os.path.join(watch_directory, d))])
        print(f"List of Data Directories in {watch_directory}:")
        print("="*20)
        for directory in data_dirs:
            print(directory)
        print("="*20)
        user_input = input('Enter the directory name to process, or type ‘AUTO’ to start automatic monitoring of new uploads: ')

    if user_input == "AUTO":
        # Set up logging
        logging.basicConfig(
            filename=f'{path_log}file_uploads.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s'
        )

        print('KMTNet ToO Data WatchDog Activated: Looking for kmtx.00000000.000000.fits')
        observer = Observer()
        event_handler = TooWatcher(watch_directory, ncores, known_obj=args.known_obj)
        observer.schedule(event_handler, watch_directory, recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()

        observer.join()
        event_handler.close_pool()
    
    else:
        if os.path.isdir(os.path.join(watch_directory, user_input)):
            ToO_pipeline(user_input, known_obj=args.known_obj, ncores=args.ncores)
        else:
            print(f'Check if {os.path.join(watch_directory, user_input)} exists.')
