from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    Noam Learning Rate Scheduler

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        model_size (int): model size (d_model)
        warmup_steps (int): number of warmup steps
        factor (float): scaling factor for learning rate
        min_lr (float): minimum learning rate after decay
        last_epoch (int): index of the last epoch (used for resuming training)
    """

    def __init__(
        self,
        optimizer,
        model_size,
        warmup_steps=4000,
        factor=1.0,
        min_lr=1e-9,
        last_epoch=-1,
    ):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.min_lr = min_lr

        self.k2 = factor * (model_size**-0.5)
        self.k1 = self.k2 * (warmup_steps**-1.5)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)

        if step <= self.warmup_steps:
            scale = self.k1 * step
        else:
            scale = self.k2 / (step**0.5)

        lr = max(scale, self.min_lr)
        return [lr for _ in self.base_lrs]
