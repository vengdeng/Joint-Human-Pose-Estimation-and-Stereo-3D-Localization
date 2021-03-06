"""Train a pifpaf net."""

import copy
import logging
import time
import torch


class Trainer(object):
    def __init__(self, model, losses, optimizer, out, lambdas,
                 lr_scheduler=None,
                 log_interval=10,
                 device=None,
                 fix_batch_norm=False,
                 stride_apply=1,
                 ema_decay=None,
                 encoder_visualizer=None,
                 train_profile=None):
        self.model = model
        self.losses = losses
        self.optimizer = optimizer
        self.out = out
        self.lr_scheduler = lr_scheduler

        self.log_interval = log_interval
        self.device = device
        self.fix_batch_norm = fix_batch_norm
        self.lambdas = lambdas
        self.stride_apply = stride_apply

        self.ema_decay = ema_decay
        self.ema = None
        self.ema_restore_params = None

        self.encoder_visualizer = encoder_visualizer

        if train_profile:
            # monkey patch to profile self.train_batch()
            self.train_batch_without_profile = self.train_batch
            def train_batch_with_profile(*args, **kwargs):
                with torch.autograd.profiler.profile() as prof:
                    result = self.train_batch_without_profile(*args, **kwargs)
                print(prof.key_averages())
                print(prof.total_average())
                prof.export_chrome_trace(train_profile)
                return result
            self.train_batch = train_batch_with_profile

        self.log = logging.getLogger(self.__class__.__name__)

    def lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def step_ema(self):
        if self.ema is None:
            return

        for p, ema_p in zip(self.model.parameters(), self.ema):
            ema_p.mul_(1.0 - self.ema_decay).add_(self.ema_decay, p.data)

    def apply_ema(self):
        if self.ema is None:
            return

        self.log.info('applying ema')
        self.ema_restore_params = copy.deepcopy(
            [p.data for p in self.model.parameters()])
        for p, ema_p in zip(self.model.parameters(), self.ema):
            p.data.copy_(ema_p)

    def ema_restore(self):
        if self.ema_restore_params is None:
            return

        self.log.info('restoring params from before ema')
        for p, ema_p in zip(self.model.parameters(), self.ema_restore_params):
            p.data.copy_(ema_p)
        self.ema_restore_params = None

    def loop(self, train_scenes, val_scenes, epochs, start_epoch=0):
        for _ in range(start_epoch):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for epoch in range(start_epoch, epochs):
            self.train(train_scenes, epoch)

            self.write_model(epoch + 1, epoch == epochs - 1)
            self.val(val_scenes, epoch + 1)

    def loss(self, outputs, targets,targets3):
        loss = None
        assert len(self.losses) == len(outputs)
        head_losses = []
        for i in range(len(outputs)):
            if i != 2:
                for c in self.losses[i](outputs[i],targets[i]):
                    head_losses.append(c)
            if i ==2:
                for c in self.losses[i](outputs[2],targets3[0]):
                    head_losses.append(c)

        # for c in self.losses[0](outputs[0], targets3[0],targets[0]):
        #     head_losses.append(c)
        #
        # for c in self.losses[1](outputs[1], targets[1]):
        #     head_losses.append(c)
        # print(len(self.losses))
        # head_losses = [c
        #                for l, o, t in zip(self.losses, outputs, targets)
        #                for c in l(o, t)]
        assert len(self.lambdas) == len(head_losses)
        loss_values = [lam * l
                       for lam, l in zip(self.lambdas, head_losses)
                       if l is not None]
        if loss_values:
            loss = sum(loss_values)

        return loss, head_losses

    def train_batch(self, data, targets,targets3, meta, apply_gradients=True):
        if self.encoder_visualizer:
            self.encoder_visualizer(data, targets, meta)

        if self.device:
            data = data.to(self.device, non_blocking=True)
            targets = [[t.to(self.device, non_blocking=True) for t in head] for head in targets]
            targets3 = [[t.to(self.device, non_blocking=True) for t in head] for head in targets3]
        # train encoder
        outputs = self.model(data)
        loss, head_losses = self.loss(outputs, targets,targets3)
        if loss is not None:
            loss.backward()
        if apply_gradients:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step_ema()

        return (
            float(loss.item()) if loss is not None else None,
            [float(l.item()) if l is not None else None
             for l in head_losses],
        )

    def val_batch(self, data, targets,targets3):
        if self.device:
            data = data.to(self.device, non_blocking=True)
            targets = [[t.to(self.device, non_blocking=True) for t in head] for head in targets]
            targets3 = [[t.to(self.device, non_blocking=True) for t in head] for head in targets3]
        with torch.no_grad():
            outputs = self.model(data)
            loss, head_losses = self.loss(outputs, targets,targets3)

        return (
            float(loss.item()) if loss is not None else None,
            [float(l.item()) if l is not None else None
             for l in head_losses],
        )

    def train(self, scenes, epoch):
        start_time = time.time()
        self.model.train()
        if self.fix_batch_norm:
            for m in self.model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    # print('fixing parameters for {}. Min var = {}'.format(
                    #     m, torch.min(m.running_var)))
                    m.eval()
                    # m.weight.requires_grad = False
                    # m.bias.requires_grad = False

                    # avoid numerical instabilities
                    # (only seen sometimes when training with GPU)
                    # Variances in pretrained models can be as low as 1e-17.
                    # m.running_var.clamp_(min=1e-8)
                    m.eps = 1e-4
        self.ema_restore()
        self.ema = None

        print('epoch', epoch)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        epoch_loss = 0.0
        head_epoch_losses = [0.0 for _ in self.lambdas]
        last_batch_end = time.time()
        self.optimizer.zero_grad()
        for batch_idx, datas in enumerate(scenes):

            (data1, data2, target1, target2,target3, meta) = datas
            #print(target3[0][0][0] == target3[0][0][1])
            data = torch.cat((data1,data2),0)
            ## cat augmented data together for training
            target1[0][0] = torch.cat((target1[0][0], target2[0][0]), 0)
            target1[0][1] = torch.cat((target1[0][1], target2[0][1]), 0)
            target1[0][2] = torch.cat((target1[0][2], target2[0][2]), 0)
            target1[1][0] = torch.cat((target1[1][0], target2[1][0]), 0)
            target1[1][1] = torch.cat((target1[1][1], target2[1][1]), 0)
            target1[1][2] = torch.cat((target1[1][2], target2[1][2]), 0)
            target1[1][3] = torch.cat((target1[1][3], target2[1][3]), 0)

            preprocess_time = time.time() - last_batch_end
            #print(data.shape[0])
            batch_start = time.time()
        # for batch_idx, (data, target, meta) in enumerate(scenes):
        #
        #     preprocess_time = time.time() - last_batch_end
        #
        #     batch_start = time.time()
            apply_gradients = batch_idx % self.stride_apply == 0
            loss, head_losses = self.train_batch(data, target1,target3, meta, apply_gradients)
            if loss is not None:
                epoch_loss += loss
            for i, head_loss in enumerate(head_losses):
                if head_loss is None:
                    continue
                head_epoch_losses[i] += head_loss
            batch_time = time.time() - batch_start

            # write training loss
            if batch_idx % self.log_interval == 0:
                self.log.info({
                    'type': 'train',
                    'epoch': epoch, 'batch': batch_idx, 'n_batches': len(scenes),
                    'time': round(batch_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.lr(),
                    'loss': round(loss, 3) if loss is not None else None,
                    'head_losses': [round(l, 3) if l is not None else None
                                    for l in head_losses],
                })

            # initialize ema
            if self.ema is None and self.ema_decay:
                self.ema = copy.deepcopy([p.data for p in self.model.parameters()])

            last_batch_end = time.time()

        self.apply_ema()
        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / len(scenes), 5),
            'head_losses': [round(l / len(scenes), 5) for l in head_epoch_losses],
            'time': round(time.time() - start_time, 1),
        })

    def val(self, scenes, epoch):
        start_time = time.time()
        self.model.eval()

        epoch_loss = 0.0
        head_epoch_losses = [0.0 for _ in self.lambdas]
        for data1, data2, target1, target2,target3, meta in scenes:
            data = torch.cat((data1, data2), 0)
            target1[0][0] = torch.cat((target1[0][0], target2[0][0]), 0)
            target1[0][1] = torch.cat((target1[0][1], target2[0][1]), 0)
            target1[0][2] = torch.cat((target1[0][2], target2[0][2]), 0)
            target1[1][0] = torch.cat((target1[1][0], target2[1][0]), 0)
            target1[1][1] = torch.cat((target1[1][1], target2[1][1]), 0)
            target1[1][2] = torch.cat((target1[1][2], target2[1][2]), 0)
            target1[1][3] = torch.cat((target1[1][3], target2[1][3]), 0)

            loss, head_losses = self.val_batch(data, target1,target3)
            if loss is not None:
                epoch_loss += loss
            for i, head_loss in enumerate(head_losses):
                if head_loss is None:
                    continue
                head_epoch_losses[i] += head_loss
        eval_time = time.time() - start_time

        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch,
            'loss': round(epoch_loss / len(scenes), 5),
            'head_losses': [round(l / len(scenes), 5) for l in head_epoch_losses],
            'time': round(eval_time, 1),
        })

    def write_model(self, epoch, final=True):
        self.model.cpu()

        if isinstance(self.model, torch.nn.DataParallel):
            self.log.debug('Writing a dataparallel model.')
            model = self.model.module
        else:
            self.log.debug('Writing a single-thread model.')
            model = self.model

        if final:
            filename = self.out
        else:
            filename = '{}.epoch{:03d}'.format(self.out, epoch)

        self.log.debug('about to write model')
        torch.save({
            'model': model,
            'epoch': epoch,
        }, filename)
        self.log.debug('model written')

        self.model.to(self.device)
