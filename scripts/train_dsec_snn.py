import tqdm
import wandb
from pathlib import Path
import argparse
import traceback
import torch
from torch_geometric.data import DataLoader

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams
from dagr.utils.buffers import DetectionBuffer
from dagr.utils.args import FLAGS
from dagr.utils.learning_rate_scheduler import LRSchedule

from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.dsec_data import DSEC

# 从DAGR_SNN替换DAGR
# from dagr.model.networks.dagr import DAGR
from dagr.model.networks.dagr_snn import DAGR_SNN as DAGR
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
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)


def debug_snn(model, data):
    """调试SNN事件分支"""
    print("\n===== DEBUGGING SNN EVENT BRANCH =====")
    model.eval()
    
    # 保存原始推理模式状态
    original_inference_mode = False
    if hasattr(model, 'event_processor'):
        if hasattr(model.event_processor, 'inference_mode'):
            original_inference_mode = model.event_processor.inference_mode
    
    with torch.no_grad():
        # 处理单个样本
        if isinstance(data, list):
            sample = data[0].cuda()
        else:
            sample = data.cuda()
            
        sample = format_data(sample)
        
        # 检查输入数据
        print(f"Input data type: {type(sample)}")
        print(f"Input data attributes: {sample.__dict__.keys()}")
        
        try:
            # 设置为训练模式进行测试
            if hasattr(model, 'event_processor') and hasattr(model.event_processor, 'set_inference_mode'):
                model.event_processor.set_inference_mode(False)
                print("\n----- Testing in TRAINING mode -----")
            
            # 运行SNN处理
            if hasattr(model, 'event_processor'):
                event_features = model.event_processor(sample)
                print(f"SNN output features: {len(event_features)}")
                for i, feat in enumerate(event_features):
                    print(f"Feature {i} shape: {feat.x.shape if hasattr(feat, 'x') else 'No x attribute'}")
            else:
                print("Model has no event_processor attribute. Using regular forward.")
            
            # 运行完整推理(训练模式)
            detections, targets = model(sample)
            print(f"Training mode - Detections length: {len(detections)}")
            
            # 设置为推理模式进行测试
            if hasattr(model, 'event_processor') and hasattr(model.event_processor, 'set_inference_mode'):
                model.event_processor.set_inference_mode(True)
                print("\n----- Testing in INFERENCE mode -----")
                
                # 再次运行SNN处理(推理模式)
                event_features = model.event_processor(sample)
                print(f"SNN output features (inference mode): {len(event_features)}")
                for i, feat in enumerate(event_features):
                    print(f"Feature {i} shape: {feat.x.shape if hasattr(feat, 'x') else 'No x attribute'}")
                
                # 运行完整推理(推理模式)
                detections, targets = model(sample)
                print(f"Inference mode - Detections length: {len(detections)}")
            
            # 恢复原始状态
            if hasattr(model, 'event_processor') and hasattr(model.event_processor, 'set_inference_mode'):
                model.event_processor.set_inference_mode(original_inference_mode)
                
            print("SNN event branch test passed!")
            return True
        except Exception as e:
            # 确保恢复原始状态
            if hasattr(model, 'event_processor') and hasattr(model.event_processor, 'set_inference_mode'):
                model.event_processor.set_inference_mode(original_inference_mode)
                
            print(f"SNN event branch test failed: {e}")
            traceback.print_exc()
            return False
    
    print("========================================\n")


def train(loader: DataLoader,
          model: torch.nn.Module,
          ema: ModelEMA,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          optimizer: torch.optim.Optimizer,
          args: argparse.ArgumentParser,
          run_name=""):

    model.train()
    # 设置为训练模式
    if hasattr(model, 'event_processor'):
        model.event_processor.set_inference_mode(False)

    total_loss = 0.0
    num_batches = 0

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):
        data = data.cuda(non_blocking=True)
        data = format_data(data)

        optimizer.zero_grad(set_to_none=True)

        try:
            model_outputs = model(data)

            loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
            loss = loss_dict.pop("total_loss")

            total_loss += loss.item()
            num_batches += 1

            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)

            fix_gradients(model)

            # 确保所有梯度和参数在同一设备上
            device = next(model.parameters()).device
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad = p.grad.to(device)

            # 确保优化器状态在正确设备上
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            optimizer.step()
            scheduler.step()

            ema.update(model)

            training_logs = {f"training/loss/{k}": v for k, v in loss_dict.items()}
            wandb.log({"training/loss": loss.item(), "training/lr": scheduler.get_last_lr()[-1], **training_logs})
        
        except Exception as e:
            print(f"Error in training batch {i}: {e}")
            traceback.print_exc()
            # 清理内存并继续
            torch.cuda.empty_cache()
            continue
        
        # 每隔几个批次清理一次缓存
        if i % 5 == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def run_test(loader: DataLoader,
         model: torch.nn.Module,
         dry_run_steps: int=-1,
         dataset="gen1"):

    model.eval()
    # 设置为推理模式
    if hasattr(model, 'event_processor'):
        model.event_processor.set_inference_mode(True)

    mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)

    for i, data in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        data = format_data(data)

        try:
            detections, targets = model(data)
            if i % 10 == 0:
                torch.cuda.empty_cache()

            mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])
        except Exception as e:
            print(f"Error in test batch {i}: {e}")
            traceback.print_exc()
            continue

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

    # 硬编码SNN参数
    args.use_snn = True
    args.snn_scale = 's'  # 使用's'规模匹配dagr-s配置
    args.snn_timesteps = 1  # 默认时间步长


    # # 添加SNN相关参数
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--use_snn', action='store_true', default=True, help='使用SNN事件分支')
    # parser.add_argument('--snn_scale', type=str, default='n', help='SNN骨干网络的规模 (n/s/m/l/x)')
    # parser.add_argument('--snn_timesteps', type=int, default=4, help='SNN时间步长')
    
    # # 解析新参数并添加到args
    # snn_args, _ = parser.parse_known_ars()
    # for k, v in vars(snn_args).items():
    #     setattr(args, k, v)

    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)
    log_hparams(args)

    augmentations = Augmentations(args)

    print("init datasets")
    #dataset_path = args.dataset_directory / args.dataset
    dataset_path = args.dataset_directory

    train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
                         min_bbox_diag=15, min_bbox_height=10)
    test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
                        min_bbox_diag=15, min_bbox_height=10)

    train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    num_iters_per_epoch = len(train_loader)

    #sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(test_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

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

    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    lr_func = LRSchedule(warmup_epochs=.3,
                         num_iters_per_epoch=num_iters_per_epoch,
                         tot_num_epochs=args.tot_num_epochs)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    checkpointer = Checkpointer(output_directory=output_directory,
                                model=model, optimizer=optimizer,
                                scheduler=lr_scheduler, ema=ema,
                                args=args)

    # 测试SNN事件分支
    print("Testing SNN event branch...")
    try:
        debug_sample = next(iter(train_loader))
        debug_result = debug_snn(model, debug_sample)

        if not debug_result:
            print("SNN event branch test failed. Proceeding with caution...")
    except Exception as e:
        print(f"SNN debug test failed with exception: {e}")
        traceback.print_exc()
        print("Proceeding with training anyway...")
    
    start_epoch = checkpointer.restore_if_existing(output_directory, resume_from_best=True)
    if start_epoch is None:
        start_epoch = 0

    # 如果从断点恢复，同步学习率调度器
    if start_epoch > 0:
        print(f"[INFO] Resuming from epoch {start_epoch}, synchronizing lr_scheduler")
        # 快进学习率调度器到正确的步数
        total_steps = start_epoch * num_iters_per_epoch
        for _ in range(total_steps):
            lr_scheduler.step()
        print(f"[INFO] LR scheduler advanced {total_steps} steps")

    if "resume_checkpoint" in args:
        start_epoch = checkpointer.restore_checkpoint(args.resume_checkpoint, best=False)
        print(f"Resume from checkpoint at epoch {start_epoch}")

    with torch.no_grad():
        mapcalc = run_test(test_loader, ema.ema, dry_run_steps=2, dataset=args.dataset)
        mapcalc.compute()

    print("starting to train")
    print(f"[FINAL] Starting training from epoch: {start_epoch}")
    for epoch in range(start_epoch, args.tot_num_epochs):
        avg_loss = train(train_loader, model, ema, lr_scheduler, optimizer, args, run_name=wandb.run.name)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        checkpointer.checkpoint(epoch, name=f"last_model")

        if epoch % 3 > 0:
            continue

        with torch.no_grad():
            print("test_loader length =", len(test_loader))
            mapcalc = run_test(test_loader, ema.ema, dataset=args.dataset)
            metrics = mapcalc.compute()
            print(f"mAP: {metrics.get('mAP', 'N/A')}")
            checkpointer.process(metrics, epoch)