# Copyright 2022 - Valeo Comfort and Driving Assistance
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from VICReg: https://github.com/facebookresearch/vicreg


from pathlib import Path
import argparse
import shutil
import time

import torch
from torch import nn
import torchvision.datasets

from src.swav.logger import create_logger, create_stats
from src.vicreg.optim import adjust_learning_rate, LARS
from src.vicreg.utils import AverageMeter
from src.vicreg.distributed import init_distributed_mode
from src.sfrik.dataset import ImageNetSubset
import src.sfrik.augmentations as aug
import src.sfrik.ssl as ssl


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Self-supervised learning pretraining script for: SFRIK, SimCLR, AUH and VICReg",
    )

    # Data
    parser.add_argument("--dataset", type=str, choices=["ImageNet", "STL10"])
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet_or_stl10", required=True,
                        help='Path to the ImageNet or STL10 dataset')
    parser.add_argument("--subset", default=-1, type=int, help="Take a fix number of images per class (example 260)")
    parser.add_argument("--augmentations", choices=["SimCLR", "VICReg", "AUH", "OBoW", "SwAV"],
                        help="choose image augmentations implemented in previous SSL methods")

    # Two-views image augmentation parameter (when setting '--augmentations SimCLR' or 'AUH' or 'AUH')
    parser.add_argument("--size", type=int, help="Size of resized cropped image")

    # OBoW multicrop image augmentation parameters (when setting '--augmentations OBoW')
    parser.add_argument("--num_img_crops", type=int, help="Number of image crops (OBoW aug, e.g. 2)")
    parser.add_argument("--image_crop_size", type=int, help="Size of these crops (OBoW aug, e.g. 160)")
    parser.add_argument("--num_img_patches", type=int, help="Number of image patches (OBoW aug, e.g. 5)")
    parser.add_argument("--img_patch_size", type=int, help="Size of image patch (OBoW aug, e.g. 96)")

    # SwAV multicrop image augmentation parameters (when setting '--augmentations SwAV')
    parser.add_argument("--nmb_crops", type=int, nargs="+", help="list of number of crops (SwAV aug, e.g. [2, 6])")
    parser.add_argument("--size_crops", type=int, nargs="+", help="crops resolutions (SwAV aug, e.g. [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, nargs="+", help="min range (SwAV aug, e.g. [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, nargs="+", help="max range (SwAV aug, e.g. [1., 0.14])")
    parser.add_argument("--general_indices", type=int, nargs="+", help="idx of global views (SwAV aug, e.g. [0, 1])")

    # Additional multicrop image augmentation parameter (when setting '--augmentations OBoW' or 'SwAV')
    parser.add_argument("--reg_small", type=str, choices=["True", "False"],
                        help="apply MMD loss on: large views + small views if True; large views only if False.")

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all checkpoints will be stored')
    parser.add_argument("--stats-dir", type=Path, default="./exp",
                        help='Path to the results folder, where all the logs and stats will be stored.')
    parser.add_argument("--checkpoint-freq", type=int, default=100,
                        help='Save the model every [checkpoint-freq] epochs, in exp-dir/checkpoints directory.')

    # Model
    parser.add_argument("--arch", type=str, help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, help='Number of epochs')
    parser.add_argument("--batch-size", type=int,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # SSL method
    parser.add_argument("--ssl-method", type=str, choices=["SFRIK", "SimCLR", "AUH", "VICReg"])

    # Loss for VICReg
    parser.add_argument("--vicreg-sim-coeff", type=float,
                        help='Invariance regularization loss coefficient (e.g. 25.0)')
    parser.add_argument("--vicreg-std-coeff", type=float,
                        help='Variance regularization loss coefficient (e.g. 25.0)')
    parser.add_argument("--vicreg-cov-coeff", type=float,
                        help='Covariance regularization loss coefficient (e.g. 1.0)')

    # Loss for AUH
    parser.add_argument("--auh-sim-coeff", type=float,
                        help='Invariance regularization loss coefficient (e.g. 3000.0)')
    parser.add_argument("--auh-unif-coeff", type=float,
                        help='Uniformity regularization loss coefficient (e.g. 1.0)')
    parser.add_argument("--auh-scale", type=float,
                        help='Scale of the RBF kernel for computing pairwise energy (e.g. 2.5)')

    # Loss for SimCLR
    parser.add_argument("--simclr-temp", type=float,
                        help='Temperature parameter for NCE loss (e.g. 0.15)')

    # Loss for SFRIK
    parser.add_argument("--sfrik-sim-coeff", type=float,
                        help='Invariance regularization loss coefficient (e.g. 4000.0)')
    parser.add_argument("--sfrik-mmd-coeff", type=float,
                        help='MMD loss regularization loss coefficient (e.g. 1.0)')
    parser.add_argument("--sfrik-weights", type=str,
                        help="Weights for kernel in the form {w_1}-{w_2}-{w_3} (e.g. 1.0-40.0-40.0)")

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Distributed
    parser.add_argument('--jean-zay', action="store_true",
                        help="set True if running on Jean Zay to use idr_torch package for distributed training")
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


def exclude_bias_and_norm(p):
    return p.ndim == 1


if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    gpu = torch.device(args.device)

    # Save dir and logger
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        args.stats_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = args.exp_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(args.stats_dir / "train.log", rank=args.rank)
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    logger.info("The experiment checkpoints will be stored in %s\n" % args.exp_dir)
    logger.info("The experiment results (stats + logs) will be stored in %s\n" % args.stats_dir)
    train_stats = create_stats(args.stats_dir / "pretraining_stats", args, ["epoch", "loss"])

    # Augmentations
    # Two views image augmùentations
    if args.augmentations in ["VICReg", "SimCLR", "AUH"]:
        transforms = eval(f"aug.{args.dataset}{args.augmentations}TrainTransform")(args.size)
        args.multicrop = None
    # Multiview augmentation like in OBoW
    elif args.augmentations == "OBoW":
        transforms = eval(f"aug.{args.dataset}OBoWTrainTransform")(
            args.num_img_crops, args.image_crop_size, args.num_img_patches, args.img_patch_size
        )
        args.multicrop = "OBoW"
    # Multiview augmentation like in SwAV
    elif args.augmentations == "SwAV":
        transforms = eval(f"aug.{args.dataset}SwAVTrainTransform")(
            args.size_crops, args.nmb_crops, args.min_scale_crops, args.max_scale_crops
        )
        args.multicrop = "SwAV"
    else:
        raise NotImplementedError

    # Dataset
    if args.dataset == "ImageNet":
        assert args.augmentations in ["VICReg", "SimCLR", "OBoW", "SwAV"]
        dataset = ImageNetSubset(args.data_dir / "train", transforms, subset=args.subset)
    elif args.dataset == "STL10":
        assert args.augmentations in ["AUH", "OBoW", "SwAV"]
        dataset = torchvision.datasets.STL10(args.data_dir, split="unlabeled", transform=transforms, download=True)
    else:
        raise NotImplementedError
    logger.info(f"Size dataset: {len(dataset)}")

    # Dataloader
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    # Model
    model = eval(f"ssl.{args.ssl_method}")(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    # Checkpoint
    if (args.exp_dir / "last_ckp.pth").is_file():
        logger.info(f"Resuming from checkpoint found at: {args.exp_dir / 'last_ckp.pth'}")
        ckpt = torch.load(args.exp_dir / "last_ckp.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        logger.info("Found no checkpoint: start from scratch")
        start_epoch = 0

    # Self-supervised training
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        logger.info("============ Starting epoch %i ... ============" % epoch)
        sampler.set_epoch(epoch)
        batch_time = AverageMeter("batch_time")
        data_time = AverageMeter("data_time")
        loss_meter = AverageMeter("loss")
        end = time.time()
        for step, mini_batch in enumerate(loader, start=epoch * len(loader)):
            torch.cuda.reset_peak_memory_stats()
            data_time.update(time.time() - end)
            lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()

            if args.multicrop is None:
                (x, y) = mini_batch[0]
                x = x.cuda(gpu, non_blocking=True)
                y = y.cuda(gpu, non_blocking=True)
                bs = x.shape[0]
                with torch.cuda.amp.autocast():
                    loss = model.forward(x, y)
            elif args.multicrop == "OBoW":
                (img_orig, img_crops) = mini_batch[0]
                bs = img_orig.shape[0]
                img_orig = img_orig.cuda(gpu, non_blocking=True)
                img_crops = [torch.cat([img[:, i, :, :, :] for i in range(img.shape[1])], dim=0) for img in img_crops]
                img_crops = [img.cuda(gpu, non_blocking=True) for img in img_crops]
                with torch.cuda.amp.autocast():
                    loss = model.forward(img_orig, img_crops)
            elif args.multicrop == "SwAV":
                inputs = mini_batch[0]
                bs = inputs[0].shape[0]
                idx_crops = torch.cumsum(torch.unique_consecutive(
                    torch.tensor([inp.shape[-1] for inp in inputs]),
                    return_counts=True,
                )[1], 0)
                start_idx = 0
                img_views = []
                for end_idx in idx_crops:
                    img_views.append(torch.cat(inputs[start_idx: end_idx]).cuda(gpu, non_blocking=True))
                    start_idx = end_idx
                with torch.cuda.amp.autocast():
                    loss = model.forward(img_views, bs)
            else:
                raise NotImplementedError

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), bs)
            batch_time.update(time.time() - end)
            if args.rank == 0 and step % 30 == 0:
                logger.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Lr: {lr:.4f} \t"
                    "Max memory: {memory} MB".format(
                        epoch,
                        step,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=loss_meter,
                        lr=lr,
                        memory=torch.cuda.max_memory_allocated() // 1e6
                    )
                )
            end = time.time()

        train_stats.update((epoch, loss_meter.avg))
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "last_ckp.pth")
            if epoch % args.checkpoint_freq == 0 or epoch == args.epoch - 1:
                shutil.copyfile(
                    args.exp_dir / "last_ckp.pth",
                    checkpoints_dir / f"ckp-{epoch}.pth",
                )
    if args.rank == 0:
        torch.save(model.module.backbone.state_dict(), args.exp_dir / f"pretrained_backbone.pth")
