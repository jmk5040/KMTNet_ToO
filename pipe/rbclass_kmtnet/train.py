import os

import hydra
import numpy as np
import pandas as pd

from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset

import modules
from modules.datasets import KMTNetDataset, TestDataset
from modules.trainer import Trainer
from modules.utils import seed_all, ConfigParser


def prepare_dataset(meta, data_root, cfg, train_transform, valid_transform):
    train_indices, valid_indices = train_test_split(np.arange(len(meta)), test_size=0.2, random_state=cfg.seed)

    train_dataset = KMTNetDataset(
        root=data_root,
        meta=meta.loc[train_indices].reset_index(drop=True),
        channels=cfg.dataset.channels,
        transform=train_transform
    )
    valid_dataset = KMTNetDataset(
        root=data_root,
        meta=meta.loc[valid_indices].reset_index(drop=True),
        channels=cfg.dataset.channels,
        transform=valid_transform
    )

    return train_dataset, valid_dataset


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    parser = ConfigParser(cfg)
    parser.save_config()
    seed_all(cfg.seed)

    # Loading meta tables
    # Meta table for RI dataset
    meta_random = pd.read_csv(cfg.dataset.meta_path[0])
    # When training a model using NGI dataset, taking only bogus samples from RI dataset
    if cfg.dataset.train_dataset_type == 'ngi':
        meta_random = meta_random[meta_random['label'] == 0].reset_index(drop=True)

    # Meta table for NGI dataset
    meta_near_galaxy = pd.read_csv(cfg.dataset.meta_path[1])
    # When training a model using RI dataset, taking only bogus samples from NGI dataset
    if cfg.dataset.train_dataset_type == 'ri':
        meta_near_galaxy = meta_near_galaxy[meta_near_galaxy['label'] == 0].reset_index(drop=True)

    if cfg.dataset.train_dataset_type == 'concat':
        ri_bogus_indices = meta_random[meta_random['label'] == 0].index.tolist()
        ri_real_indices = meta_random[meta_random['label'] == 1].sample(frac=0.5, random_state=cfg.seed).index.tolist()
        ri_indices = ri_bogus_indices + ri_real_indices
        meta_random = meta_random.loc[ri_indices].reset_index(drop=True)

        ngi_bogus_indices = meta_near_galaxy[meta_near_galaxy['label'] == 0].index.tolist()
        ngi_real_indices = meta_near_galaxy[meta_near_galaxy['label'] == 1].sample(frac=0.5, random_state=cfg.seed).index.tolist()
        ngi_indices = ngi_bogus_indices + ngi_real_indices
        meta_near_galaxy = meta_near_galaxy.loc[ngi_indices].reset_index(drop=True)

    if cfg.dataset.train_dataset_type == 'concat-with-gd':
        ri_bogus_indices = meta_random[meta_random['label'] == 0].index.tolist()
        ri_real_indices = meta_random[meta_random['label'] == 1].sample(frac=0.5, random_state=cfg.seed).index.tolist()
        ri_indices = ri_bogus_indices + ri_real_indices
        meta_random = meta_random.loc[ri_indices].reset_index(drop=True)

        ngi_bogus_indices = meta_near_galaxy[meta_near_galaxy['label'] == 0].index.tolist()
        ngi_real_indices = meta_near_galaxy[meta_near_galaxy['label'] == 1].index.tolist()
        ngi_indices = ngi_bogus_indices + ngi_real_indices
        meta_near_galaxy = meta_near_galaxy.loc[ngi_indices].reset_index(drop=True)

    # Meta table for test dataset
    test_meta = pd.read_csv(cfg.dataset.test_meta_path)

    train_transform = parser.train_transform
    valid_transform = parser.valid_transform

    train_dataset_random, valid_dataset_random = prepare_dataset(meta_random,
                                                                 cfg.dataset.data_root[0],
                                                                 cfg, train_transform, valid_transform)
    train_dataset_galaxy, valid_dataset_galaxy = prepare_dataset(meta_near_galaxy,
                                                                 cfg.dataset.data_root[1],
                                                                 cfg, train_transform, valid_transform)
    train_dataset = ConcatDataset([train_dataset_random, train_dataset_galaxy])
    valid_dataset = ConcatDataset([valid_dataset_random, valid_dataset_galaxy])

    test_dataset = TestDataset(
        root=cfg.dataset.test_data_root,
        meta=test_meta,
        channels=cfg.dataset.channels,
        transform=valid_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = instantiate(cfg.model)
    trainer = Trainer(model, cfg)
    trainer.fit(train_loader, valid_loader, test_loader)


if __name__ == '__main__':
    main()
