'''
Predict the label of each fits file in a given directory.

## Installation (successfully tested on Mac OS)
$ pip install -r requirements.txt

- If you get "undefined symbol: cublasLtGetStatusString, version libcublasLt.so.11" error,
- please try `$ pip uninstall nvidia_cublas_cu11`.

## Usage
$ python inference.py --dir_fits <target directory>
$ python inference.py --dir_fits ./data/kmtnet/fits
$ python inference.py --dir_fits ./data/kmtnet/fits --imsize 51 --channels ref new sub --normalize min_max
$ (or equivalently) python inference.py --dir_fits ./data/kmtnet/fits -s 51 -c ref new sub -n min_max
$ python inference.py --dir_fits /data/SNU/final_snapshots --dir_ckpt /home/postech/projects/kmtnet/ckpt --ckpt_name model:OTrain_imsize:51_channels:rns_normalize:minmax_name:ri_seed:0.bin

- Arguments
See `parse_arge()` function

- Output
A CSV file, named result_table.csv`, containing two columns, labeled `id` and `prob`.
The `id` column contains file names, while the `prob` column contains the corresponding model predictions.
'''


import argparse
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch

from albumentations.pytorch import ToTensorV2
from astropy.io import fits
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from modules.networks import OTrain
from modules.transforms import Normalize


class TestDataset(Dataset):
    def __init__(self, root, meta, channels, transform):
        super(TestDataset, self).__init__()
        self.root = root
        self.meta = meta
        self.channels = channels
        self.n_channels = len(channels)
        self.transform = transform

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        X = []
        y = self.meta.loc[idx, 'label']

        # Stack data from specified channels into one array.
        for c, channel in enumerate(self.channels):
            fpath = os.path.join(self.root, self.meta.loc[idx, f'{channel}im'])
            img = fits.getdata(fpath)
            # Converts NaN values to 0 as per preprocessing strategy. See 'Note' in docstring for details.
            img = np.nan_to_num(img, 0.0)
            X.append(img)
        X = np.stack(X).transpose((1, 2, 0))

        if self.transform is not None:
            X = self.transform(image=X)['image']

        return X, y

    def __len__(self):
        return len(self.meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_fits', type=str, default='/data/Postech/data/kmtnet/fits',
                        help='The directory where *.fits files are stored.')
    parser.add_argument('--dir_ckpt', type=str, default='./ckpt',
                        help='The directory where the model checkpoints are locaated.')
    parser.add_argument('--ckpt_name', type=str, default='otrain_51_l2_rns.bin',
                        help='The name of the checkpoint')
    parser.add_argument('-e', '--ensemble', action='store_true',
                        help='Flag whether using ensemble or not. If the argument is passed' +
                        '--ckpt_name argument is ignored and use all ckpt files in the --dir_fits.')

    # Data processing options
    parser.add_argument('-s', '--imsize', type=int, default=51,
                        help='The size of an image. The original image should be larger than this argument')
    parser.add_argument('-n', '--normalize', type=str, default='min_max',
                        help='The normalization method. It should be one of [`l2`, `min_max`, `zscale`]')
    parser.add_argument('-c', '--channels', nargs='+', default=['ref', 'new', 'sub'],
                        help='The channels to be used. You should pass the multiple channels by giving a whitespace' +
                        "e.g., > python inferece.py --channels ref new sub")
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The number of samples that the model processes for each iteration.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID.')
    args = parser.parse_args()

    return args


def create_meta_tables(targetdir, channels):
    fname_list = glob(os.path.join(targetdir, '*.???.fits')) # *.new.fits, *.ref.fits, *.sub.fits
    fname_list = sorted(fname_list)
    fname_list = list(map(os.path.basename, fname_list))

    ids = ['.'.join(fname.split('.')[:-2]) for fname in fname_list]
    ids = np.sort(np.unique(ids))

    meta = pd.DataFrame()
    meta['id'] = ids
    meta['label'] = -1.0
    for c in channels:
        meta[f'{c}im'] = meta['id'].apply(lambda x: x + f'.{c}.fits')

    return meta


@torch.no_grad()
def predict(model, loader, device):
    probs = []
    model.eval()
    for (X, _) in tqdm(loader, total=len(loader), desc='Inference'):
        X = X.to(device)
        prob = model(X)
        probs.append(prob.cpu().numpy())
    probs = np.concatenate(probs)

    return probs


def main():
    args = parse_args()
    print(f"Preparing data for {args.dir_fits}.")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    meta = create_meta_tables(args.dir_fits, args.channels)
    transform = A.Compose([A.CenterCrop(args.imsize, args.imsize), Normalize(args.normalize), ToTensorV2()])
    dataset = TestDataset(root=args.dir_fits,
                          meta=meta,
                          channels=args.channels,
                          transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loading images (n={len(meta)}).")
    # Load model
    model = OTrain(imsize=args.imsize, in_channels=len(args.channels))
    state_dict_list = []
    if not args.ensemble:
        state_dict_list.append(torch.load(os.path.join(args.dir_ckpt, args.ckpt_name), map_location=device))
    else:
        for ckpt_path in glob(os.path.join(args.dir_ckpt, '*.bin')):
            state_dict_list.append(torch.load(ckpt_path, map_location=device))
    model.to(device)
    for i, state_dict in enumerate(state_dict_list):
        model.load_state_dict(state_dict['model_state_dict'])
        prob = predict(model, loader, device)
        meta[f'prob_model_{i}'] = prob

    print(f"Saving the result to {os.path.join(args.dir_fits, 'rbscore.csv')}")
    meta['prob'] = meta[[f'prob_model_{i}' for i in range(len(state_dict_list))]].mean(axis=1)
    if not args.ensemble:
        meta = meta.drop('prob_model_0', axis=1)
    drop_columns = ['label'] + [f'{c}im' for c in args.channels]
    meta = meta.drop(drop_columns, axis=1)
    meta.to_csv('./result_table.csv', index=False)
    
    # Save only objects with RB score > 0.5
    meta_over50 = meta[meta['prob'] > 0.5].copy()
    meta_over50.to_csv('./result_table_over50.csv', index=False)
    
    meta.to_csv(os.path.join(args.dir_fits, 'rbscore.csv'), index=False)
    print(f"Summary: The number of objects got rbscore>0.5: {len(meta[meta['prob'] > 0.5])}")

if __name__ == '__main__':
    main()
