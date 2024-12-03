import argparse
import math
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import models.longmae as models_mae
from dataset import LMDB_Dataset
from transforms import RandomResizedCrop3D, ZScoreNormalizationPerSample


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--csv_path', default='csvs/ct_rate_train.csv', type=str,
                        help='csv dataset path')
    parser.add_argument('--lmdb_path', default='/mnt/nas/Datasets/than/CT/LMDB/ct_rate_train_512.lmdb', type=str,
                        help='lmdb dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Additional parameters
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--print_freq', default=20, type=int, help='Print frequency during training')

    return parser

# Utility functions (adjusted from the scripts you provided)
def adjust_learning_rate(optimizer, epoch, args, batch_idx, num_training_steps_per_epoch):
    """Decay the learning rate with half-cycle cosine after warmup"""
    progress = epoch + batch_idx / num_training_steps_per_epoch
    if progress < args.warmup_epochs:
        lr = args.lr * progress / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (progress - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
    return lr

def add_weight_decay(model, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (not bias_wd and len(param.shape) == 1) or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

class MAELightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        self.args = args

        # Define the model
        self.model = models_mae.__dict__[self.hparams.model](norm_pix_loss=self.hparams.norm_pix_loss)

        # Initialize loss scaler for mixed precision
        if self.hparams.use_amp:
            self.loss_scaler = torch.cuda.amp.GradScaler()
        else:
            self.loss_scaler = None

        # For tracking learning rate
        self.current_lr = self.hparams.lr

    def forward(self, x):
        return self.model(x, mask_ratio=self.hparams.mask_ratio)

    def training_step(self, batch, batch_idx):
        samples, _ = batch  # Unpack the batch
        samples = samples.to(self.device, non_blocking=True)

        optimizer = self.optimizers()
        num_training_steps_per_epoch = len(self.trainer.datamodule.train_dataloader())

        # Adjust learning rate per iteration
        lr = adjust_learning_rate(optimizer, self.current_epoch, self.args, batch_idx, num_training_steps_per_epoch)
        self.current_lr = lr  # For logging

        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.hparams.use_amp):
            loss, _, _ = self(samples)

        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        # Prepare optimizer
        param_groups = add_weight_decay(self.model, self.hparams.weight_decay, bias_wd=False)
        optimizer = torch.optim.AdamW(param_groups, lr=self.hparams.lr, betas=(0.9, 0.95))

        return optimizer

    def on_train_epoch_start(self):
        # Reset learning rate at the start of each epoch
        optimizer = self.optimizers()
        adjust_learning_rate(optimizer, self.current_epoch, self.args, batch_idx=0, num_training_steps_per_epoch=1)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)  # Better performance

class MAEDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        # Define transforms
        self.transforms_train = transforms.Compose([
            RandomResizedCrop3D(size=(128, 224, 224)),
            ZScoreNormalizationPerSample()
        ])

    def setup(self, stage=None):
        # Create dataset
        self.dataset_train = LMDB_Dataset(csv_path=self.args.csv_path,
                                          lmdb_path=self.args.lmdb_path,
                                          transforms=self.transforms_train)

    def train_dataloader(self):
        # Use DistributedSampler if multiple GPUs are used
        if self.trainer.num_devices > 1:
            sampler = torch.utils.data.DistributedSampler(self.dataset_train)
        else:
            sampler = None

        return torch.utils.data.DataLoader(self.dataset_train,
                                           batch_size=self.batch_size,
                                           sampler=sampler,
                                           num_workers=self.num_workers,
                                           shuffle=(sampler is None),
                                           pin_memory=self.args.pin_mem,
                                           drop_last=True)

def main(args):
    # Set the seed for reproducibility
    seed = args.seed + int(os.environ.get('LOCAL_RANK', 0))
    pl.seed_everything(seed)

    # Determine devices
    if args.device == 'cuda':
        devices = torch.cuda.device_count()
        accelerator = 'gpu'
    else:
        devices = 1
        accelerator = 'cpu'

    # Compute effective batch size
    eff_batch_size = args.batch_size * args.accum_iter * devices

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base LR: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual LR: {args.lr:.2e}")
    print(f"Accumulate grad iterations: {args.accum_iter}")
    print(f"Effective batch size: {eff_batch_size}")

    # Create the DataModule
    data_module = MAEDataModule(args)

    # Create the LightningModule
    model = MAELightningModule(args)

    # Set up WandbLogger
    wandb_logger = WandbLogger(project='MAE', config=args, save_dir=args.log_dir)

    # Set up callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename='checkpoint-{epoch}',
        save_top_k=-1,  # Save all models
        every_n_epochs=20,  # Save every 20 epochs
    )

    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=devices,
        accelerator=accelerator,
        strategy='ddp' if devices > 1 else None,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision=16 if args.use_amp else 32,
        accumulate_grad_batches=args.accum_iter,
        log_every_n_steps=args.print_freq,
        num_sanity_val_steps=0,
        gradient_clip_val=0.0,  # Set to a value if gradient clipping is needed
        enable_checkpointing=True,
        check_val_every_n_epoch=None,  # No validation set
    )

    # Resume from checkpoint if provided
    if args.resume:
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model, datamodule=data_module)

    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MAE pre-training script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
