# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import os
import sys
import numpy as np
import pandas as pd
from typing import Iterable, Optional
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

from timm.data import Mixup
from timm.utils import accuracy

import misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast('cuda'):
            outputs = model(samples)
            
            loss = criterion(outputs, targets.float())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            # Use wandb.log directly
            log_writer.log({'train_loss': loss_value_reduce, 'lr': max_lr}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda'):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_bce(data_loader, model, device, save_npy=None):
    criterion = nn.BCEWithLogitsLoss()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # We'll gather predictions and targets for the entire dataset
    all_preds = []
    all_targets = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        targets = batch[-1]  # Assumes the last element is the label/target

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            # Convert targets to float for BCE
            loss = criterion(outputs, targets.float())

        # Save predictions and targets for computing ROC-AUC after the loop
        all_preds.append(outputs.detach().cpu())
        all_targets.append(targets.detach().cpu())

        # Log the loss
        metric_logger.update(loss=loss.item())

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    if save_npy is not None:
        np.save(os.path.join(save_npy, 'all_preds.npy'), all_preds)
        np.save(os.path.join(save_npy, 'all_targets.npy'), all_targets)

    # Compute ROC-AUC (sklearn expects arrays of shape [N] for binary case)
    roc_auc = roc_auc_score(all_targets, all_preds)
    # Update the metric logger for ROC-AUC
    metric_logger.meters['roc_auc'].update(roc_auc, n=len(all_preds))
    # Gather the stats from all processes (if using distributed training)
    metric_logger.synchronize_between_processes()

    # Print final stats
    print('* ROC-AUC {roc_auc.global_avg:.3f}  '
          'loss {losses.global_avg:.3f}'
          .format(
              roc_auc=metric_logger.roc_auc,
              losses=metric_logger.loss
          ))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def extract_features(data_loader, model, device, df=None):
    """
    Extract features from a data loader using the given model and add them to a DataFrame.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader providing batches of data.
        model (torch.nn.Module): The model for feature extraction.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to use for computation.
        df (pd.DataFrame, optional): Existing DataFrame to which features will be added. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with extracted features added.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Feature extraction:'

    feature_list = []

    # Iterate over the data loader with logging
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]  # Assuming the batch contains images as the first element
        images = images.to(device, non_blocking=True)

        # Compute features using mixed precision if available
        # with torch.amp.autocast(device_type='cuda'):
        feature, _ = model(images, return_feature=True)
        feature_list.append(feature.detach().cpu().numpy())

    # Combine all extracted features into a single array
    features = np.concatenate(feature_list, axis=0)

    # Reshape features into a flat list (if necessary, based on feature dimensions)
    features = features.reshape(-1, features.shape[-1])

    # Create a DataFrame for features
    feature_df = pd.DataFrame(features, columns=[f"pred_{idx}" for idx in range(features.shape[1])])

    # If an existing DataFrame is provided, concatenate features; otherwise, return feature_df
    if df is not None:
        df = pd.concat([df, feature_df], axis=1)
    else:
        df = feature_df

    return df