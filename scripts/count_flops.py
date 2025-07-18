import os
import tqdm
import torch
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from torch_geometric.data import DataLoader

from dagr.utils.args import FLOPS_FLAGS
from dagr.utils.buffers import DictBuffer, format_data

from dagr.data.augment import Augmentations
from dagr.data.dsec_data import DSEC

from dagr.model.networks.dagr import DAGR

from dagr.asynchronous.evaluate_flops import evaluate_flops


if __name__ == '__main__':
    import torch_geometric
    seed = 42
    torch_geometric.seed.seed_everything(seed)
    args = FLOPS_FLAGS()
    assert "checkpoint" in args

    project = f"flops-{args.dataset}-{args.task}"
    pbar = tqdm.tqdm(total=4)

    pbar.set_description("Loading dataset")
    dataset_path = args.dataset_directory / args.dataset
    #print(f"[DEBUG] Dataset path: {dataset_path}")

    #print(f"[DEBUG] Loading with split: test")
    print(f"[DEBUG] Loading with split: train")
    print("init datasets")
    #dataset = DSEC(args.dataset_directory, "test", Augmentations.transform_testing, debug=True, min_bbox_diag=15, min_bbox_height=10)
    dataset = DSEC(args.dataset_directory, "train", Augmentations.transform_testing, debug=True, min_bbox_diag=15, min_bbox_height=10)
    print(f"[DEBUG] Dataset loaded. Total samples: {len(dataset)}")
    loader = DataLoader(dataset, follow_batch=['bbox', "bbox0"], batch_size=args.batch_size, shuffle=False, num_workers=16)
    pbar.update(1)

    pbar.set_description("Initializing net")
    model = DAGR(args, height=dataset.height, width=dataset.width)
    model = model.cuda()
    model.eval()
    pbar.update(1)
    print("[DEBUG] Network initialized.")

    assert "checkpoint" in args
    print("[DEBUG] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint)
    print(f"[DEBUG] Checkpoint keys: {checkpoint.keys()}")
    model.load_state_dict(checkpoint['ema'])
    print("[DEBUG] Checkpoint loaded.")
    pbar.update(1)

    model.cache_luts(radius=args.radius, height=dataset.height, width=dataset.width)

    #pbar.set_description("Computing FLOPS")
    #buffer = DictBuffer()
    #args.output_directory.mkdir(parents=True, exist_ok=True)
    #pbar_flops = tqdm.tqdm(total=len(loader.dataset), desc="Computing FLOPS")
    #for i, data in enumerate(loader):
    #    data = data.cuda(non_blocking=True)
    #    data = format_data(data)

    #    flops_evaluation = evaluate_flops(model, data,
    #                                      check_consistency=args.check_consistency,
    #                                      return_all_samples=True, dense=args.dense)
    #    if flops_evaluation is None:
    #        continue

    #    buffer.update(flops_evaluation['flops_per_layer'])
    #    buffer.save(args.output_directory / "flops_per_layer.pth")
    #    tot_flops = sum(buffer.compute().values())

    #    pbar_flops.set_description(f"Total FLOPS {tot_flops}")
    #    pbar_flops.update(1)

    #print(sum(buffer.compute().values()))
    #pbar.update(1)
    print("Computing FLOPS")
    buffer = DictBuffer()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    pbar_flops = tqdm.tqdm(total=len(loader.dataset), desc="Computing FLOPS")

    for i, data in enumerate(loader):
        print(f"[DEBUG] Loading sample {i}: {data}")
        data = data.cuda(non_blocking=True)
        data = format_data(data)
        print(f"[DEBUG] Sample {i} formatted: {data}")

        print("[DEBUG] Evaluating FLOPS...")
        flops_evaluation = evaluate_flops(model, data,
                                          check_consistency=args.check_consistency,
                                          return_all_samples=True, dense=args.dense)
        print(f"[DEBUG] FLOPS Evaluation for sample {i}: {flops_evaluation}")
        if flops_evaluation is None:
            print(f"[WARNING] FLOPs evaluation for batch {i} returned None")
            continue

        buffer.update(flops_evaluation.get('flops_per_layer', {}))
        buffer.save(args.output_directory / "flops_per_layer.pth")
        tot_flops = sum(buffer.compute().values()) if buffer.compute() is not None else 0
        print(f"[DEBUG] Total FLOPS for batch {i}: {tot_flops}")
        pbar_flops.set_description(f"Total FLOPS {tot_flops}")
        pbar_flops.update(1)

    print("[DEBUG] Finalizing FLOPS calculation...")
    flops_data = buffer.compute()
    if flops_data is not None:
        print("Total FLOPS:", sum(flops_data.values()))
    else:
        print("No valid FLOPS data calculated.")
    pbar.update(1)



