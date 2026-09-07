'''
Real/Bogus inference straight from the full-frame images -- no snapshot files.

Why this exists
---------------
The file-based inference.py expects one directory of pre-cut 51x51-ish FITS
stamps. Producing those stamps dominates the reduction: for a single KMTNet
field, subtraction() wrote 3 x 26,028 = 78,084 files (8.8 GB) and spent 3,402 s
doing it, so that a classifier could look at 26,028 of them and flag ~145 as
worth a human's time. On top of that the stamps are 150x150 while the model
center-crops to 51x51, so only 11.6% of the written pixels were ever read.

This script inverts the order: score every candidate by cutting its 51x51
window out of the three full-frame images in memory, then write stamps only for
the survivors.

Exactness
---------
The cutouts are produced by the SAME code path as generate_snapshot() --
Cutout2D(cutsize arcmin) followed by albumentations CenterCrop(imsize) -- so the
arrays handed to the model are bit-identical to the old ones. Cutting 51x51
directly does NOT reproduce them: Cutout2D rounds an even-sized (150) and an
odd-sized (51) cutout differently, which shifts the window by 0 or 1 pixel
depending on the sub-pixel position of the source. Verified bit-identical on
120/120 sampled cutouts.

The only thing hoisted out of the loop is the world->pixel transform, which is
done once for all candidates with skycoord_to_pixel() and then handed to
Cutout2D as a pixel position. That alone takes the cut from 131 ms to 4.1 ms per
candidate; passing pixel positions was verified to give identical arrays to
passing a SkyCoord.

Usage
-----
$ python inference_cutout.py --transient_cat <...transient.cat> \
      --dir_ckpt rbclass_kmtnet/ckpt --ckpt_name <...>.bin \
      --outdir <subt dir>/snap --snap_thresh 0.5
'''

import argparse
import os
import time

import albumentations as A
import numpy as np
import pandas as pd
import torch

from albumentations.pytorch import ToTensorV2
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
import astropy.units as u
from torch.utils.data import DataLoader, Dataset

from modules.networks import OTrain
from modules.transforms import Normalize

# generate_snapshot lives one directory up, in the pipeline's util module.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from KMTNet_util_functions import (generate_snapshot,        # noqa: E402
                                   generate_single_snapshots)


CHANNEL_COLUMN = {'new': 'inim', 'ref': 'hcim', 'sub': 'hdim'}
# NOTE: 'ref' is hcim = the CONVOLVED REFERENCE (hcREF...), not hcCalib.
# generate_snapshot() zips [inim, hcim, hdim] against ['new', 'ref', 'sub'].


class CutoutDataset(Dataset):
    """51x51x3 windows cut on demand from three memory-mapped full-frame images."""

    def __init__(self, table, images, channels, cutsize, imsize):
        super().__init__()
        self.table = table
        self.channels = channels
        self.cutsize = cutsize
        self.transform = A.Compose([A.CenterCrop(imsize, imsize),
                                    Normalize('min_max'), ToTensorV2()])

        # Open once; memmap keeps the 3 x 1.9 GB frames out of resident memory
        # and lets DataLoader workers share the page cache.
        self._paths = {c: images[c] for c in channels}
        self._hdu = {c: fits.open(images[c], memmap=True)[0] for c in channels}
        wcs = {c: WCS(self._hdu[c].header) for c in channels}

        # One vectorised world->pixel transform for every candidate.
        pos = SkyCoord(ra=np.asarray(table['ALPHA_J2000'], dtype=float),
                       dec=np.asarray(table['DELTA_J2000'], dtype=float),
                       unit='deg', frame='icrs')
        self.pix = {c: skycoord_to_pixel(pos, wcs[c]) for c in channels}

        # Cutout2D refuses an angular size unless it is also handed a WCS, and
        # re-deriving the pixel size per call is exactly the cost we are trying
        # to avoid. Resolve it once, per channel, by letting Cutout2D itself do
        # the angular->pixel conversion on the first candidate -- self-calibrating,
        # so it stays correct if the pixel scale or astropy's rounding changes.
        self.size_px = {}
        for c in channels:
            probe = Cutout2D(self._hdu[c].data, position=pos[0], size=cutsize,
                             wcs=wcs[c], mode='partial', fill_value=0)
            self.size_px[c] = probe.shape          # (ny, nx)

    def __len__(self):
        return len(self.table)

    def _reopen(self):
        # DataLoader workers fork after __init__; re-open so each worker owns
        # its own file handle rather than sharing a forked mmap.
        self._hdu = {c: fits.open(p, memmap=True)[0] for c, p in self._paths.items()}

    def __getitem__(self, idx):
        X = []
        for c in self.channels:
            cut = Cutout2D(self._hdu[c].data,
                           position=(self.pix[c][0][idx], self.pix[c][1][idx]),
                           size=self.size_px[c], mode='partial', fill_value=0)
            X.append(np.nan_to_num(cut.data, 0.0))
        X = np.stack(X).transpose((1, 2, 0))
        return self.transform(image=X)['image'], -1.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--transient_cat', required=True,
                   help='*.transient.cat written by subtraction() (full table, tab separated).')
    p.add_argument('--outdir', default=None,
                   help='Where surviving snapshots are written (default: <subt dir>/snap).')
    p.add_argument('--dir_ckpt', default='./ckpt')
    p.add_argument('--ckpt_name', default='otrain_51_l2_rns.bin')
    p.add_argument('-e', '--ensemble', action='store_true')
    p.add_argument('-s', '--imsize', type=int, default=51)
    p.add_argument('-c', '--channels', nargs='+', default=['ref', 'new', 'sub'])
    p.add_argument('--cutsize', type=float, default=1.0,
                   help='Cutout size in arcmin, must match subtraction() (default 1.0).')
    p.add_argument('--all_sources', action='store_true',
                   help="Score every source in the catalogue, ignoring the flag cuts. For "
                        "checking what the flags throw away: the columns are kept in the "
                        "output so a score can be attributed to the flags that were set.")
    p.add_argument('--no_snapshots', action='store_true',
                   help='Score only; write no stamps.')
    p.add_argument('--single_thresh', type=float, default=0.9,
                   help="Also cut the candidate out of each contributing single-chip "
                        "exposure above this score, into <outdir>/single/. Those are what "
                        "tell a cosmic ray or a subtraction residual from something that "
                        "was on the sky. Set above 1 to disable.")
    p.add_argument('--snap_thresh', type=float, default=0.5,
                   help='Write a snapshot for candidates scoring above this (default 0.5). '
                        'known-object matches are always written.')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--pixscale', type=float, default=0.4)
    return p.parse_args()


def _as_bool(col):
    return np.array([str(x).strip().lower() == 'true' for x in col])


def _normalise_types(tbl):
    """Restore Python types that ascii.read() flattens into strings.

    subtraction() hands generate_snapshot() a live table; here the same table
    comes back off disk as tab-separated text, so booleans arrive as the strings
    'True'/'False' and empty cells as masked values. That matters because
    generate_snapshot() does `bool(row['known_match'])` -- and bool('False') is
    True, which would stamp KNOWNOBJ=True on every snapshot it writes.
    """
    for col in ('known_match', 'flag_0', 'flag'):
        if col in tbl.colnames and tbl[col].dtype.kind in 'US':
            tbl[col] = _as_bool(tbl[col])
    if 'known_target' in tbl.colnames:
        tbl['known_target'] = np.array(
            ['' if (x is np.ma.masked or str(x) in ('--', 'nan')) else str(x)
             for x in tbl['known_target']], dtype=object)
    return tbl


@torch.no_grad()
def predict(model, loader, device):
    probs = []
    model.eval()
    for X, _ in loader:
        probs.append(model(X.to(device)).cpu().numpy())
    return np.concatenate(probs)


def main():
    args = parse_args()
    t_start = time.time()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full = ascii.read(args.transient_cat, format='tab')
    flag_pass = ~_as_bool(full['flag']) if 'flag' in full.colnames else np.ones(len(full), bool)
    known = _as_bool(full['known_match']) if 'known_match' in full.colnames else np.zeros(len(full), bool)
    if args.all_sources:
        sel = np.ones(len(full), bool)
        print('Scoring EVERY source (flag cuts ignored).')
    else:
        sel = flag_pass | known
    tbl = _normalise_types(full[sel])
    print(f'Candidates in catalogue : {len(full)}')
    print(f'Scored (flag-pass | known-object) : {len(tbl)}')

    images = {c: str(tbl[CHANNEL_COLUMN[c]][0]) for c in args.channels}
    for c in args.channels:
        print(f'  {c:<4} <- {os.path.basename(images[c])}')

    cutsize = u.Quantity((args.cutsize, args.cutsize), u.arcmin)
    dataset = CutoutDataset(tbl, images, args.channels, cutsize, args.imsize)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        worker_init_fn=(lambda _: dataset._reopen()) if args.num_workers else None)

    model = OTrain(imsize=args.imsize, in_channels=len(args.channels))
    state_dicts = []
    if not args.ensemble:
        state_dicts.append(torch.load(os.path.join(args.dir_ckpt, args.ckpt_name), map_location=device))
    else:
        from glob import glob as _glob
        for p in _glob(os.path.join(args.dir_ckpt, '*.bin')):
            state_dicts.append(torch.load(p, map_location=device))
    model.to(device)

    t_infer = time.time()
    meta = pd.DataFrame()
    stem = os.path.splitext(os.path.basename(str(tbl['hdim'][0])))[0]
    meta['id'] = [f'{stem}.{int(n):06d}' for n in tbl['NUMBER']]
    # Carry the flags through so a score can be attributed to what excluded it.
    for _c in [c for c in tbl.colnames if c == 'flag' or c.startswith('flag_')]:
        meta[_c] = _as_bool(tbl[_c])
    for i, sd in enumerate(state_dicts):
        model.load_state_dict(sd['model_state_dict'])
        meta[f'prob_model_{i}'] = predict(model, loader, device)
    meta['prob'] = meta[[f'prob_model_{i}' for i in range(len(state_dicts))]].mean(axis=1)
    if not args.ensemble:
        meta = meta.drop('prob_model_0', axis=1)
    print(f'Scoring done in {time.time() - t_infer:.1f}s')

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.transient_cat)), 'snap')
    os.makedirs(outdir, exist_ok=True)
    meta.to_csv(os.path.join(outdir, 'rbscore.csv'), index=False)

    # Snapshots only for what a human would actually open.
    if args.no_snapshots:
        print(f"Summary: The number of objects got rbscore>0.5: "
              f"{int((meta['prob'] > 0.5).sum())}")
        print(f'Total {time.time() - t_start:.1f}s')
        return
    survive = (meta['prob'].values > args.snap_thresh) | known[sel]
    n_forced = int(np.count_nonzero(known[sel] & ~(meta['prob'].values > args.snap_thresh)))
    print(f'Writing snapshots for {int(survive.sum())} candidate(s) '
          f'(rbscore > {args.snap_thresh}; {n_forced} forced by known-object match)')
    t_snap = time.time()
    keep = tbl[survive]
    for i in range(len(keep)):
        generate_snapshot(keep[i], cutsize=args.cutsize, pixscale=args.pixscale, outdir=outdir)
    print(f'Snapshots written in {time.time() - t_snap:.1f}s')

    # The few worth opening by eye also get one cutout per contributing
    # exposure, under <outdir>/single/. Known-object matches are included
    # whatever they scored -- that is the point of forcing them through.
    single = (meta['prob'].values > args.single_thresh) | known[sel]
    if single.any():
        t_sing = time.time()
        sdir = os.path.join(outdir, 'single')
        n_files = 0
        best = tbl[single]
        for i in range(len(best)):
            n_files += generate_single_snapshots(best[i], sdir, cutsize=args.cutsize,
                                                 pixscale=args.pixscale)
        print(f'Single-exposure cutouts: {n_files} file(s) for '
              f'{int(single.sum())} candidate(s) (rbscore > {args.single_thresh}) '
              f'in {time.time() - t_sing:.1f}s')
    print(f'Summary: The number of objects got rbscore>0.5: '
          f"{int((meta['prob'] > 0.5).sum())}")
    print(f'Total {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
