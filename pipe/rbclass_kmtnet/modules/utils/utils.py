import os
import json
import random

from datetime import datetime

import albumentations as A
import numpy as np
import torch

from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf, open_dict

from ..transforms import Normalize

OmegaConf.register_new_resolver("len", lambda x: len(x))


def load_config(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def accuracy(output, target):
    with torch.no_grad():
        n = len(target)
        pred = (output > 0.5).long()
        # pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / n, n


def recall(output, target):
    '''
    Recall = TP/(TP+FN)
    TP는 target vector와 prediction vector의 내적으로 쉽게 구할 수 있습니다.
    한편, TP+FN의 값은 target vector의 각 성분을 모두 합하는 것으로 쉽게 구할 수 있습니다.
    '''
    if torch.sum(target).item() == 0:
        return (0.0, 0)

    with torch.no_grad():
        n = torch.sum(target).item()
        # pred = torch.argmax(output, dim=1)
        pred = (output > 0.5).long()
        assert pred.shape[0] == len(target)
        TP = 0
        TP += torch.inner(target, pred).item()
    return TP / n, n


def precision(output, target):
    '''
    Precision = TP/(TP+FP)
    TP는 target vector와 prediction vector의 내적으로 쉽게 구할 수 있습니다.
    한편, TP+FP의 값은 pred vector의 각 성분을 모두 합하는 것으로 쉽게 구할 수 있습니다.
    '''
    with torch.no_grad():
        # pred = torch.argmax(output, dim=1)
        pred = (output > 0.5).long()
        n = torch.sum(pred).item()
        if n == 0:
            return (0.0, 0)

        assert pred.shape[0] == len(target)
        TP = 0
        TP += torch.inner(target, pred).item()
    return TP / n, n


def F1_score(output, target):
    '''
    F1 score는 recall과 precision의 조화평균으로 주어집니다.
    '''
    n = len(target)
    recl, n1 = recall(output, target)
    prec, n2 = precision(output, target)
    if n1 == 0 or n2 == 0:
        return (0.0, 0)
    elif (recl + prec == 0):
        return (0.0, n)
    return 2 * recl * prec / (recl + prec), n


def parse_id(name):
    name = name.split('.')
    obs = name[1].split('_')[1]
    field = name[2]
    date = datetime.strptime(name[3], '%Y%m%d-%H%M%S')
    band = name[4]
    exposure_time = int(name[5])
    sec = name[7].split('_')[1]
    sec_x, sec_y = map(int, sec.split('x'))

    return dict(date=date, obs=obs, field=field, band=band, sec_x=sec_x, sec_y=sec_y, exposure_time=exposure_time)


class ConfigParser:
    def __init__(self, config, training=True):
        self.config = config

        # Update the result directory
        if self.config.run_id is None:
            run_id = self._create_run_id()
            self.update_config(self.config, 'run_id', run_id)
        self.update_config(self.config, 'resultdir', os.path.join(self.config.resultdir, self.config.run_id))

        if training:
            os.makedirs(self.config.resultdir, exist_ok=True)
            self.update_config(self.config.wandb, 'group', '_'.join(run_id.split('_')[:-1]))
            self.update_config(self.config.wandb, 'name', run_id)

    @classmethod
    def from_dict(cls, obj, training=False):
        return cls(OmegaConf.create(obj), training)

    def _create_run_id(self):
        placeholder = dict(
            model=self.config.model._target_.split('.')[-1],
            imsize=self.config.dataset.imsize,
            channels=''.join(list(map(lambda x: x[0], self.config.dataset.channels))),
            normalize=self.config.dataset.normalize,
            batch_size=self.config.batch_size,
            name=self.config.exp_name,
            seed=self.config.seed
        )
        run_id = '_'.join([f'{k}:{v}' for k, v in placeholder.items()])

        return run_id

    @property
    def train_transform(self):
        transform = A.Compose([
            A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
            A.CenterCrop(self.config.dataset.imsize, self.config.dataset.imsize),
            Normalize(self.config.dataset.normalize),
            A.HorizontalFlip(),
            ToTensorV2()
        ])
        return transform

    @property
    def valid_transform(self):
        transform = A.Compose([
            A.CenterCrop(self.config.dataset.imsize, self.config.dataset.imsize),
            Normalize(self.config.dataset.normalize),
            ToTensorV2()
        ])
        return transform

    def update_config(self, config, key, value):
        with open_dict(config):
            setattr(config, key, value)

    def save_config(self, fpath=None):
        if fpath is None:
            fpath = os.path.join(self.config.resultdir, 'config.json')
        with open(fpath, 'w') as f:
            json.dump(OmegaConf.to_container(self.config, resolve=True), f, indent=4)
