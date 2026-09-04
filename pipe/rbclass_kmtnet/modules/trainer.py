"""Training utilities for efficient deep learning model experimentation and evaluation.

This script integrates essential functionalities for training deep learning models,
including dynamic learning rate adjustments, metric tracking, and experiment logging.

It supports training, validation, and optional testing phases, with automatic metrics
calculation and logging via the Weights & Biases platform.

"""
import os
import math
import time

import pandas as pd
import torch
import torch.nn as nn
import wandb

from   glob import glob
from   hydra.utils import instantiate
from   omegaconf import OmegaConf
from   torch.optim.lr_scheduler import _LRScheduler

from   .utils import F1_score, accuracy, precision, recall


class Trainer:
    """
    Facilitates the training, validation, and optional testing of a model.

    This class encapsulates the entire cycle of model training including but
    not limited to handling data loaders for training and validation phases,
    optimizing the model parameters, adjusting the learning rate, logging
    metrics, and checkpointing the model states.
    It supports metrics tracking through predefined functions and experiment
    logging using Weights & Biases.

    Parameters:
    - model: The PyTorch model to be trained.
    - cfg: A configuration object containing training, model architecture, optimizer, scheduler,
      and logging parameters.

    Attributes:
    - epochs (int): Number of training epochs as specified in the configuration.
    - resultdir (str): Directory path for saving training artifacts (e.g., model checkpoints).
    - criterion (torch.nn.modules.loss): Loss function instantiated based on the model's output type.
    - optimizer (torch.optim.Optimizer): Optimizer for the model parameters.
    - scheduler: Learning rate scheduler.
    - device (torch.device): Computation device, CUDA or CPU.
    - best_metric (float): Best metric value for model checkpointing.
    - metric_ftns (tuple): Tuple containing metric functions to be used for evaluation.
    """
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.epochs = self.cfg.epochs
        self.resultdir = self.cfg.resultdir

        self.criterion = nn.BCELoss() if cfg.model.num_classes == 1 else nn.CrossEntropyLoss()
        self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())
        self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, **self.cfg.scheduler)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.best_metric = 0.0
        self.metric_ftns = (accuracy, recall, precision, F1_score)

    def fit(self, train_loader, valid_loader, test_loader=None):
        """
        Conducts the training, validation, and optionally testing phases of the model.

        Iterates over the specified number of epochs, performing training and validation
        for each epoch. Optionally evaluates the model on a test dataset if provided.
        Logs metrics and training progress to Weights & Biases and saves model checkpoints
        based on validation recall or another specified metric.

        Parameters:
        - train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        - valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        - test_loader (torch.utils.data.DataLoader, optional): DataLoader for test data.
        """
        os.makedirs(self.resultdir, exist_ok=True)
        wandb.init(**self.cfg.wandb, config=OmegaConf.to_container(self.cfg, resolve=True))
        for e in range(self.epochs):
            # Training phase
            train_metrics = self.train(train_loader)
            # Validation phase
            valid_metrics = self.evaluate(valid_loader)
            # Test pahse (Optional)
            test_metrics = dict()
            if test_loader is not None:
                test_metrics = self.evaluate(test_loader)

            # Log metircs and learning rate to wandb
            log = {}
            log['Epoch'] = e + 1
            for k, v in train_metrics.items():
                log['train_' + k] = v
            for k, v in valid_metrics.items():
                log['valid_' + k] = v
            for k, v in test_metrics.items():
                log['test_' + k] = v

            if self.scheduler is not None:
                log['LR'] = self.scheduler.get_lr()[0]
                self.scheduler.step()
            wandb.log(log)

            # Checkpointing
            self.save(f'{self.resultdir}/last_ckpt.bin')
            if valid_metrics['recall'] > self.best_metric:
                self.best_metric = valid_metrics['recall']
                self.save(f'{self.resultdir}/best_ckpt_{str(e+1).zfill(4)}.bin')
                # Keep top 3 models
                for path in sorted(glob(f'{self.resultdir}/best_ckpt_*.bin'))[:-3]:
                    os.remove(path)
        wandb.finish()

    def train(self, loader):
        """
        Performs a single epoch of training on the provided dataset loader.

        Iterates over the training dataset, computes the loss for each batch, performs backpropagation,
        and updates the model parameters. Additionally, tracks and logs training metrics for each batch.

        Parameters:
        - loader (torch.utils.data.DataLoader): DataLoader providing the training dataset.

        Returns:
        A dictionary containing averaged metric values (e.g., loss, accuracy) for the training epoch.
        """
        self.model.train()
        n_iters = len(loader)
        start = time.time()
        metric_tracker = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        for i, (X, y) in enumerate(loader):
            print(f"\r[Training steps: {i + 1} / {n_iters}] ", end='')
            # Forward pass
            X, y = X.to(self.device), y.to(self.device)
            probs = self.model(X)
            loss = self.criterion(probs, y.float())

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metric tracker for loss
            n = len(X)
            metric_tracker.update(key='loss', value=loss.item(), n=n)

        print(f"Elapsed time: {time.time() - start:.2f}")
        return metric_tracker.result()  # Return averaged metrics for this training epoch

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Evaluates the model on the provided dataset loader without gradient computation.

        Iterates over the dataset, computes the loss, and aggregates predictions for metric
        calculation. Useful for both validation and testing phases. Metrics are calculated
        using the specified functions in self.metric_ftns.

        Parameters:
        - loader (torch.utils.data.DataLoader): DataLoader providing the dataset for evaluation.

        Returns:
        A dictionary containing averaged metric values (e.g., loss, accuracy, precision) for the evaluated dataset.
        """
        self.model.eval()
        y_true = []
        y_pred = []
        n_iters = len(loader)
        start = time.time()
        metric_tracker = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        for i, (X, y) in enumerate(loader):
            print(f"\r[Validation steps: {i + 1} / {n_iters}] ", end='')
            X, y = X.to(self.device), y.to(self.device)
            # Forward pass
            probs = self.model(X)
            loss = self.criterion(probs, y.float())

            # Update metric tracker for loss
            n = len(X)
            metric_tracker.update(key='loss', value=loss.item(), n=n)

            y_true.append(y)
            y_pred.append(probs)

        # Calculate and update additional metrics
        y_true = torch.concat(y_true)
        y_pred = torch.concat(y_pred)
        for met in self.metric_ftns:
            score, n = met(y_pred.cpu(), y_true.cpu())
            metric_tracker.update(met.__name__, score, n)
        print(f"Elapsed time: {time.time() - start:.2f}")

        return metric_tracker.result()  # Return averaged metrics for this evaluation phase

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
    https://gaussian37.github.io/dl-pytorch-lr_scheduler/
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) *
                    (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class MetricTracker:
    """
    Tracks and calculates average values for metrics over an epoch.

    This class is designed to facilitate the tracking of various metrics
    during the training and evaluation phases of a model.
    It can handle multiple metrics simultaneously, keeping track of the sum
    and count of each metric to compute their average values.
    This is particularly useful for monitoring performance metrics like loss,
    accuracy, precision, and recall during model training and evaluation.

    Parameters:
    - *keys (str): Variable length argument list specifying the names of the metrics to track.

    Attributes:
    - _data (pandas.DataFrame): A DataFrame to store metric totals, counts, and averages.
    """
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()  # Initialize or reset the tracking data

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """
        Updates the tracked data for a specified metric.

        Adds a new value to the total for the specified metric and increments the count of values
        added. This method also recalculates the average for the metric based on the new total and count.

        Parameters:
        - key (str): The name of the metric to update.
        - value (float): The new value to add to the metric's total.
        - n (int, optional): The number of occurrences of the metric value. Defaults to 1.
        """
        if not n == 0:
            self._data.total[key] += value * n
            self._data.counts[key] += n
            self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
