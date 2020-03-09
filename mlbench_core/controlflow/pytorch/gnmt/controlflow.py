import logging
import math

import torch
import torch.optim
import torch.utils.data
from mlbench_core.controlflow.pytorch.controlflow import \
    _record_train_batch_stats
from mlbench_core.utils import AverageMeter


class GNMTTrainer:

    def __init__(self, model, criterion, fp_optimizer, scheduler,
                 schedule_per, tracker, metrics, target, iter_size):
        self.model = model
        self.batch_first = model.batch_first
        self.criterion = criterion
        self.epoch = 0

        # Optimizers & Scheduler
        self.fp_optimizer = fp_optimizer

        self.schedule_per = schedule_per
        self.scheduler = scheduler
        self.device = next(model.parameters()).device

        self.metrics = metrics
        self.target = target
        self.iter_size = iter_size

        self.tracker = tracker

    def compute_model_output(self, src, tgt):
        src, src_length = src
        tgt, tgt_length = tgt
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_length = src_length.to(self.device)

        if self.batch_first:
            output = self.model(src, src_length, tgt[:, :-1])
        else:
            output = self.model(src, src_length, tgt[:-1])

        return output

    def compute_loss(self, src, tgt, output):
        src, src_length = src
        tgt, tgt_length = tgt
        tgt = tgt.to(self.device)
        src_length = src_length.to(self.device)

        num_toks = {'tgt': int(sum(tgt_length - 1)),
                    'src': int(sum(src_length))}

        if self.batch_first:
            tgt_labels = tgt[:, 1:]
            T, B = output.size(1), output.size(0)

        else:
            tgt_labels = tgt[1:]
            T, B = output.size(0), output.size(1)

        loss = self.criterion(output.view(T * B, -1),
                              tgt_labels.contiguous().view(-1))

        loss_per_batch = loss.item()
        loss /= (B * self.iter_size)

        loss_per_token = loss_per_batch / num_toks['tgt']
        loss_per_sentence = loss_per_batch / B

        return loss, loss_per_token, loss_per_sentence, num_toks

    def feed_data(self, data_loader, num_batches_per_device_train,
                  training=True):
        """
        Runs training or validation on batches from data_loader.

        :param data_loader: data loader
        :param training: if True runs training else runs validation
        """

        losses_per_token = AverageMeter()

        for batch_idx, data in enumerate(data_loader):

            if self.tracker:
                self.tracker.batch_start()

            if self.tracker and self.schedule_per == "batch":
                self.scheduler.step()

            # Clear gradients in the optimizer.
            self.fp_optimizer.zero_grad()
            if self.tracker:
                self.tracker.record_batch_step("init")

            # Compute the output
            src, tgt = data.src, data.trg
            output = self.compute_model_output(src, tgt)
            if self.tracker:
                self.tracker.record_batch_step("fwd_pass")

            # Compute the loss
            stats = self.compute_loss(src, tgt, output)
            loss, loss_per_token, loss_per_sentence, num_toks = stats
            losses_per_token.update(loss_per_token, num_toks['tgt'])

            if self.tracker:
                self.tracker.record_batch_step("comp_loss")
            print(batch_idx, losses_per_token.avg)

            # Backprop
            self.fp_optimizer.backward_loss(loss)
            if self.tracker:
                self.tracker.record_batch_step("backprop")

            if training:
                self.fp_optimizer.step()
                if self.tracker:
                    self.tracker.record_batch_step("opt_step")

            if self.tracker:
                self.tracker.batch_end()

                _record_train_batch_stats(
                    batch_idx,
                    loss.item(),
                    output,
                    self.target,
                    self.metrics,
                    self.tracker,
                    num_batches_per_device_train,
                )

    def optimize(self, data_loader, num_batches_per_device_train):
        """
        Sets model in training mode, preallocates memory and runs training on
        data provided by data_loader.
        Args:
            data_loader: Data loader

        Returns:

        """
        torch.set_grad_enabled(True)
        self.model.train()

        self.feed_data(data_loader, num_batches_per_device_train,
                       training=True)

    def evaluate(self, data_loader, num_batches_per_device_train):
        """
        Sets model in eval mode, disables gradients, preallocates memory and
        runs validation on data provided by data_loader.

        :param data_loader: data loader
        """
        torch.set_grad_enabled(False)
        self.model.eval()
        self.feed_data(data_loader, num_batches_per_device_train,
                       training=False)


def perhaps_convert_float(param, total):
    if isinstance(param, float):
        param = int(param * total)
    return param


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with exponential warmup and step decay.
    """

    def __init__(self, optimizer, iterations, warmup_steps=0,
                 remain_steps=1.0, decay_interval=None, decay_steps=4,
                 decay_factor=0.5, last_epoch=-1):
        """
        Constructor of WarmupMultiStepLR.

        Parameters: warmup_steps, remain_steps and decay_interval accept both
        integers and floats as an input. Integer input is interpreted as
        absolute index of iteration, float input is interpreted as a fraction
        of total training iterations (epochs * steps_per_epoch).

        If decay_interval is None then the decay will happen at regulary spaced
        intervals ('decay_steps' decays between iteration indices
        'remain_steps' and 'iterations').

        :param optimizer: instance of optimizer
        :param iterations: total number of training iterations
        :param warmup_steps: number of warmup iterations
        :param remain_steps: start decay at 'remain_steps' iteration
        :param decay_interval: interval between LR decay steps
        :param decay_steps: max number of decay steps
        :param decay_factor: decay factor
        :param last_epoch: the index of last iteration
        """

        # iterations before learning rate reaches base LR
        self.warmup_steps = perhaps_convert_float(warmup_steps, iterations)
        logging.info(f'Scheduler warmup steps: {self.warmup_steps}')

        # iteration at which decay starts
        self.remain_steps = perhaps_convert_float(remain_steps, iterations)
        logging.info(f'Scheduler remain steps: {self.remain_steps}')

        # number of steps between each decay
        if decay_interval is None:
            # decay at regulary spaced intervals
            decay_iterations = iterations - self.remain_steps
            self.decay_interval = decay_iterations // (decay_steps)
            self.decay_interval = max(self.decay_interval, 1)
        else:
            self.decay_interval = perhaps_convert_float(decay_interval,
                                                        iterations)
        logging.info(f'Scheduler decay interval: {self.decay_interval}')

        # multiplicative decay factor
        self.decay_factor = decay_factor
        logging.info(f'Scheduler decay factor: {self.decay_factor}')

        # max number of decay steps
        self.decay_steps = decay_steps
        logging.info(f'Scheduler max decay steps: {self.decay_steps}')

        if self.warmup_steps > self.remain_steps:
            logging.warn(f'warmup_steps should not be larger than '
                         f'remain_steps, setting warmup_steps=remain_steps')
            self.warmup_steps = self.remain_steps

        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            # exponential lr warmup
            if self.warmup_steps != 0:
                warmup_factor = math.exp(math.log(0.01) / self.warmup_steps)
            else:
                warmup_factor = 1.0
            inv_decay = warmup_factor ** (self.warmup_steps - self.last_epoch)
            lr = [base_lr * inv_decay for base_lr in self.base_lrs]

        elif self.last_epoch >= self.remain_steps:
            # step decay
            decay_iter = self.last_epoch - self.remain_steps
            num_decay_steps = decay_iter // self.decay_interval + 1
            num_decay_steps = min(num_decay_steps, self.decay_steps)
            lr = [
                base_lr * (self.decay_factor ** num_decay_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # base lr
            lr = [base_lr for base_lr in self.base_lrs]
        return lr
