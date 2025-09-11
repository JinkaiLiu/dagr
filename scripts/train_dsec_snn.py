# avoid matlab error on server
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# avoid hdf5 bug
for k in ('HDF5_VOL_CONNECTOR', 'HDF5_PLUGIN_PATH'):
    os.environ.pop(k, None)

import hdf5plugin
import h5py

import torch
import tqdm
import wandb
from pathlib import Path
import argparse

from torch_geometric.data import DataLoader
from torch.utils.data import Subset

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams
from dagr.utils.buffers import DetectionBuffer
from dagr.utils.args import FLAGS
from dagr.utils.learning_rate_scheduler import LRSchedule

from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.dsec_data import DSEC

from dagr.model.networks.dagr_snn import DAGR
from dagr.model.networks.ema import ModelEMA


def gradients_broken(model):
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            valid_gradients = not (torch.isnan(param.grad).any())
            if not valid_gradients:
                break
    return not valid_gradients

def fix_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.nan_to_num(param.grad, nan=0.0)


def train(loader: DataLoader,
          model: torch.nn.Module,
          ema: ModelEMA,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          optimizer: torch.optim.Optimizer,
          args: argparse.ArgumentParser,
          run_name=""):

    model.train()

    total_loss_sum = 0.0
    num_steps = 0

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):
        data = data.cuda(non_blocking=True)
        data = format_data(data)

        optimizer.zero_grad(set_to_none=True)

        model_outputs = model(data)

        loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
        loss = loss_dict.pop("total_loss")

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)

        fix_gradients(model)

        optimizer.step()
        scheduler.step()

        ema.update(model)

        training_logs = {f"training/loss/{k}": v for k, v in loss_dict.items()}
        wandb.log({"training/loss": loss.item(), "training/lr": scheduler.get_last_lr()[-1], **training_logs})

        # accumulate for epoch mean
        total_loss_sum += float(loss.item())
        num_steps += 1

    # return mean loss for console print
    mean_loss = total_loss_sum / max(1, num_steps)
    return mean_loss

def run_test(loader: DataLoader,
         model: torch.nn.Module,
         dry_run_steps: int=-1,
         dataset="gen1"):

    model.eval()

    # Unwrap Subset to access base dataset attributes
    base_dataset = loader.dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset

    mapcalc = DetectionBuffer(height=base_dataset.height, width=base_dataset.width, classes=base_dataset.classes)

    for i, data in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        data = format_data(data)

        detections, targets = model(data)
        if i % 10 == 0:
            torch.cuda.empty_cache()

        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])

        if dry_run_steps > 0 and i == dry_run_steps:
            break

    torch.cuda.empty_cache()

    return mapcalc

if __name__ == '__main__':
    import torch_geometric
    import random
    import numpy as np

    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()

    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)
    log_hparams(args)

    augmentations = Augmentations(args)

    print("init datasets")
    dataset_path = args.dataset_directory / args.dataset

    train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
                         min_bbox_diag=15, min_bbox_height=10)
    test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
                        min_bbox_diag=15, min_bbox_height=10)

    # Experiment trend modes: fast / mid / full
    # fast: train+val use subsets; evaluate every epoch on fast subset, no full eval
    # mid: train full; eval every epoch on fast subset AND every 3 epochs on full
    # full: train full; eval only every 3 epochs on full (no fast subset)

    EXP_TREND = getattr(args, 'exp_trend', 'fast')
    TRAIN_SUB_N = 5000
    VAL_SUB_N = 1000

    use_train_subset = (EXP_TREND == 'fast')
    use_val_fast_subset = (EXP_TREND in ('fast', 'mid'))
    full_eval_every_n_epochs = 3 if EXP_TREND in ('mid', 'full') else None

    if use_train_subset and len(train_dataset) > TRAIN_SUB_N:
        train_indices = list(range(TRAIN_SUB_N))
        train_dataset = Subset(train_dataset, train_indices)
        print(f"[FastTrend] Using train subset: {len(train_indices)} / original", flush=True)

    # Always have full validation dataset
    test_dataset_full = test_dataset
    test_dataset_fast = None
    if use_val_fast_subset:
        if len(test_dataset) > VAL_SUB_N:
            val_indices = list(range(VAL_SUB_N))
            test_dataset_fast = Subset(test_dataset, val_indices)
            print(f"[FastTrend] Using val subset: {len(val_indices)} / original for frequent eval", flush=True)
        else:
            test_dataset_fast = test_dataset

    train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    num_iters_per_epoch = len(train_loader)

    # Build validation loaders per configured datasets
    test_loader_fast = None
    if test_dataset_fast is not None:
        sampler = np.random.permutation(np.arange(len(test_dataset_fast)))
        test_loader_fast = DataLoader(test_dataset_fast, sampler=sampler, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    test_loader_full = DataLoader(test_dataset_full, sampler=np.random.permutation(np.arange(len(test_dataset_full))), follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Training with {num_params} number of parameters.")

    model = model.cuda()
    ema = ModelEMA(model)

    nominal_batch_size = 64
    lr = args.l_r * np.sqrt(args.batch_size) / np.sqrt(nominal_batch_size)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=args.weight_decay)

    lr_func = LRSchedule(warmup_epochs=.3,
                         num_iters_per_epoch=num_iters_per_epoch,
                         tot_num_epochs=args.tot_num_epochs)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    checkpointer = Checkpointer(output_directory=output_directory,
                                model=model, optimizer=optimizer,
                                scheduler=lr_scheduler, ema=ema,
                                args=args)

    start_epoch = checkpointer.restore_if_existing(output_directory, resume_from_best=False)

    start_epoch = 0
    if "resume_checkpoint" in args:
        start_epoch = checkpointer.restore_checkpoint(args.resume_checkpoint, best=False)
        print(f"Resume from checkpoint at epoch {start_epoch}")

    # Warmup evaluation based on exp_trend
    with torch.no_grad():
        if EXP_TREND == 'full':
            warmup_loader = test_loader_full
        else:
            warmup_loader = test_loader_fast if test_loader_fast is not None else test_loader_full
        mapcalc = run_test(warmup_loader, ema.ema, dry_run_steps=1, dataset=args.dataset)
        metrics_pre = mapcalc.compute()
        # print concise warmup evaluation summary to console
        summary_keys = ['mAP', 'mAP_50', 'mAP_75', 'mAP_S', 'mAP_M', 'mAP_L']
        summary_parts = []
        for k in summary_keys:
            if k in metrics_pre:
                try:
                    summary_parts.append(f"{k}={metrics_pre[k]:.4f}")
                except Exception:
                    summary_parts.append(f"{k}={metrics_pre[k]}")
        if summary_parts:
            print("[Eval][Warmup] " + ", ".join(summary_parts), flush=True)

    print("starting to train")
    for epoch in range(start_epoch, args.tot_num_epochs):
        mean_loss = train(train_loader, model, ema, lr_scheduler, optimizer, args, run_name=args.exp_name)
        try:
            current_lr = lr_scheduler.get_last_lr()[-1]
        except Exception:
            current_lr = optimizer.param_groups[0]['lr']
        print(f"[Train][Epoch {epoch}] loss={mean_loss:.6f}, lr={current_lr:.6g}", flush=True)
        checkpointer.checkpoint(epoch, name=f"last_model")

        # Evaluation scheduling per exp_trend
        with torch.no_grad():
            if EXP_TREND == 'fast':
                if test_loader_fast is not None:
                    mapcalc = run_test(test_loader_fast, ema.ema, dataset=args.dataset)
                    metrics = mapcalc.compute()
                    checkpointer.process(metrics, epoch)
            elif EXP_TREND == 'mid':
                if (epoch % 3 == 0):
                    mapcalc = run_test(test_loader_full, ema.ema, dataset=args.dataset)
                else:
                    mapcalc = run_test(test_loader_fast, ema.ema, dataset=args.dataset)
                metrics = mapcalc.compute()
                checkpointer.process(metrics, epoch)
            elif EXP_TREND == 'full':
                if (epoch % 3 == 0):
                    mapcalc = run_test(test_loader_full, ema.ema, dataset=args.dataset)
                    metrics = mapcalc.compute()
                    checkpointer.process(metrics, epoch)

