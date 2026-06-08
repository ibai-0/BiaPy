"""
Learning rate scheduler with delayed cosine annealing for BiaPy.

This module provides the DelayedCosineAnnealingScheduler class, which keeps
the learning rate constant for a specified fraction of epochs, then applies
a single half-period cosine decay over the remaining epochs.
"""

from torch.optim.optimizer import Optimizer
import math


class DelayedCosineAnnealingScheduler:
    """Learning rate scheduler with constant phase and delayed cosine decay.

    The learning rate remains at its initial (maximum) value for the first
    ``(1 - decay_fraction) * epochs`` epochs, then cosine-decays to ``min_lr``
    over the final ``decay_fraction * epochs`` epochs.

    Parameters
    ----------
    lr : float
        Initial (maximum) learning rate.
    min_lr : float
        Minimum learning rate after decay.
    epochs : int
        Total number of training epochs.
    decay_fraction : float, optional
        Fraction of total epochs allocated to cosine decay (default 0.2,
        i.e. decay over the final 20 % of training).
    """

    def __init__(
        self,
        lr: float,
        min_lr: float,
        epochs: int,
        decay_fraction: float = 0.2,
    ):
        self.lr = lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.decay_fraction = decay_fraction
        self.decay_start = int(epochs * (1 - decay_fraction))
        self.decay_epochs = max(1, epochs - self.decay_start)
        self.current_epoch = 0

    def adjust_learning_rate(
        self,
        optimizer: Optimizer,
        epoch: float | int,
    ) -> float:
        """Adjust the learning rate based on the current epoch.

        Parameters
        ----------
        optimizer : Optimizer
            PyTorch optimizer whose learning rate will be adjusted.
        epoch : float or int
            Current epoch (can be fractional for finer granularity).

        Returns
        -------
        lr : float
            The adjusted learning rate.
        """
        if epoch < self.decay_start:
            lr = self.lr
        else:
            effective = epoch - self.decay_start
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * effective / self.decay_epochs)
            )
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr