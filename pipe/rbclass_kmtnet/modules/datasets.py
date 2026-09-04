import os
from   typing import List, Tuple

import numpy as np
import pandas as pd

from   astropy.io import fits
from   torch.utils.data import Dataset


class KMTNetDataset(Dataset):
    def __init__(
        self,
        root: str,
        meta: pd.DataFrame,
        channels: List[str] = ['ref', 'new', 'sub'],
        transform=None,
    ):
        """
        A dataset class for loading and preprocessing astronomical image data from KMTNet.
        This class supports loading images from different channels (e.g., 'ref', 'new', 'sub'),
        applying normalization, and optional transformations.

        Attributes:
            root (str): Root directory path where images are stored.
            meta (pd.DataFrame): Metadata DataFrame containing image labels and file paths.
            channels (list of str): List of channels to be loaded for each sample.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(KMTNetDataset, self).__init__()
        self.root = root
        self.meta = meta
        self.channels = channels
        self.n_channels = len(channels)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:

        if idx >= len(self):
            raise IndexError
        X = []
        y = self.meta.loc[idx, 'label']

        # Stack data from specified channels into one array.
        for c, channel in enumerate(self.channels):
            fpath = os.path.join(self.root, self.meta.loc[idx, 'class'], self.meta.loc[idx, f'{channel}im'])
            img = fits.getdata(fpath)
            X.append(img)
        X = np.stack(X).transpose((1, 2, 0))

        if self.transform is not None:
            X = self.transform(image=X)['image']

        return X, y

    def __len__(self):
        return len(self.meta)


class TestDataset(Dataset):
    def __init__(
        self,
        root: str,
        meta: pd.DataFrame,
        channels: List[str] = ['ref', 'new', 'sub'],
        transform=None,
    ):
        """
        A dataset class designed for loading and preprocessing test data to inference step.
        Similar to KMTNetDataset, it supports loading images from specified channels,
        normalization, and optional transformations. This class is specifically tailored for
        test datasets, where images might be stored differently or additional preprocessing steps
        might be required.

        Attributes:
            root (str): Root directory path where test images are stored.
            meta (pd.DataFrame): Metadata DataFrame containing image labels and file paths.
            channels (list of str): List of channels to be loaded for each sample.
            transform (callable, optional): Optional transform to be applied on a sample.

        Note:
            Converting NaN values to 0 is a deliberate preprocessing step based on prior research by Gregory Paek (SNU).
            This approach is adopted from https://github.com/SilverRon/gppy/blob/main/phot/gregoryfind_bulk_mp_2021.py#L100,
            where handling NaN values in this manner was found to be effective for the dataset and analysis involved.
        """
        super(TestDataset, self).__init__()
        self.root = root
        self.meta = meta
        self.channels = channels
        self.n_channels = len(channels)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        if idx >= len(self):
            raise IndexError
        X = []
        y = self.meta.loc[idx, 'label']

        # Stack data from specified channels into one array.
        for c, channel in enumerate(self.channels):
            fpath = os.path.join(self.root, 'fits', self.meta.loc[idx, f'{channel}im'])
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
