# Copyright 2022 - Valeo Comfort and Driving Assistance
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from VICReg: https://github.com/facebookresearch/vicreg


from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time

from torch import nn, optim
from torchvision import datasets, transforms
import torch

from src.vicreg.utils import AverageMeter, handle_sigusr1, handle_sigterm, accuracy
import src.vicreg.resnet
from src.sfrik.dataset import ImageNetSubset


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet by linear probing or semi-supervised learning."
    )

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    parser.add_argument("--subset", type=int, default=-1, help="subset of ImageNet")
    parser.add_argument("--train-percent", default=100, type=int, choices=(100, 10, 1),
                        help="size of training set in percent")
    parser.add_argument("--val-dataset", choices=["train", "val"],
                        help="Choice of the test dataset."
                             "Choose 'val' for choosing the usual ImageNet validation set."
                             "Choose 'train' for choosing a subset of the ImageNet train set.")
    parser.add_argument("--val-subset", type=int, help="Size of validation set when setting '--val-dataset train'."
                                                       "Take a fix number of images per class.")
    parser.add_argument("--only-inference", action='store_true', help="Perform only one inference on test dataset for "
                                                                      "evaluation.")

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument("--ckp-path", type=Path, help="path to linear head checkpoint")
    parser.add_argument("--exp-dir", type=Path, default="./exp", metavar="DIR",
                        help="path to checkpoint directory")
    parser.add_argument("--print-freq", default=100, type=int, metavar="N", help="print frequency")

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")

    # Optim
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--batch-size", default=256, type=int, metavar="N", help="mini-batch size")
    parser.add_argument("--lr-backbone", default=0.0, type=float, metavar="LR", help="backbone base learning rate")
    parser.add_argument("--lr-head", default=0.3, type=float, metavar="LR", help="classifier base learning rate")
    parser.add_argument("--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay")
    parser.add_argument("--weights", default="freeze", type=str, choices=("finetune", "freeze"),
                        help="finetune or freeze resnet weights")

    # Running
    parser.add_argument("--workers", default=8, type=int, metavar="N", help="number of data loader workers")

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
    # Single-node distributed training
    args.rank = 0
    args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    # Save logs
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))

    # Set up distributed training
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    # Backbone
    backbone, embedding = src.vicreg.resnet.__dict__[args.arch](zero_init_residual=True)
    state_dict = torch.load(args.pretrained, map_location="cpu")
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    assert missing_keys == [] and unexpected_keys == []

    # Linear layer
    head = nn.Linear(embedding, 1000)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()

    # Model
    model = nn.Sequential(backbone, head)
    model.cuda(gpu)
    if args.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # Optimizer
    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Automatically resume from checkpoint if it exists
    if args.ckp_path is not None or (args.exp_dir / "checkpoint.pth").is_file():
        # Load checkpoint from args.ckp_path in priority.
        # If this file does not exist, load from args.exp_dir / "checkpoint.pth".
        ckp_path = args.ckp_path if args.ckp_path is not None else args.exp_dir / "checkpoint.pth"
        ckpt = torch.load(ckp_path, map_location="cpu")
        print("Load ckp from: ", args.ckp_path)
        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        assert not args.only_inference
        print("Checkpoint not found")
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    # Loading train data set
    traindir = args.data_dir / "train"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = ImageNetSubset(
        traindir,
        transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), normalize]),
        subset=args.subset
    )

    # Loading test dataset
    val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                         normalize])
    if args.val_dataset == "val":
        valdir = args.data_dir / "val"
        val_dataset = datasets.ImageFolder(valdir, val_transforms)
    elif args.val_dataset == "train":
        valdir = args.data_dir / "train"
        val_dataset = ImageNetSubset(valdir, transform=val_transforms,start=args.subset+1, subset=args.val_subset)
    else:
        raise NotImplementedError
    print("Size of test dataset:", len(val_dataset))

    # Select 1% or 10% of labeled images for semi-supervised learning
    if args.train_percent in {1, 10}:
        train_dataset.samples = []
        with open(Path("imagenet_subsets") / f"{args.train_percent}percent.txt", "rb") as f:
            args.train_files = f.readlines()
            for fname in args.train_files:
                fname = fname.decode().strip()
                cls = fname.split("_")[0]
                train_dataset.samples.append((traindir / cls / fname, train_dataset.class_to_idx[cls]))
    print("Size of train dataset:", len(train_dataset))

    # Dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    # Only inference on test dataset with the current checkpoint
    start_time = time.time()
    if args.only_inference:
        model.eval()
        if args.rank == 0:
            evaluate(start_epoch, model, val_loader, gpu, best_acc, stats_file)

    # Train + evaluate over several epochs
    else:
        for epoch in range(start_epoch, args.epochs):
            # Train
            if args.weights == "finetune":
                model.train()
            elif args.weights == "freeze":
                model.eval()
            else:
                assert False
            train_sampler.set_epoch(epoch)
            for step, (images, target) in enumerate(
                train_loader, start=epoch * len(train_loader)
            ):
                output = model(images.cuda(gpu, non_blocking=True))
                loss = criterion(output, target.cuda(gpu, non_blocking=True))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % args.print_freq == 0:
                    torch.distributed.reduce(loss.div_(args.world_size), 0)
                    if args.rank == 0:
                        pg = optimizer.param_groups
                        lr_head = pg[0]["lr"]
                        lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                        stats = dict(
                            epoch=epoch,
                            step=step,
                            lr_backbone=lr_backbone,
                            lr_head=lr_head,
                            loss=loss.item(),
                            time=int(time.time() - start_time),
                        )
                        print(json.dumps(stats))
                        print(json.dumps(stats), file=stats_file)

            # Evaluate on test set
            model.eval()
            if args.rank == 0:
                evaluate(epoch, model, val_loader, gpu, best_acc, stats_file)

            # Scheduler step + save checkpoint
            scheduler.step()
            if args.rank == 0:
                state = dict(
                    epoch=epoch + 1,
                    best_acc=best_acc,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                )
                torch.save(state, args.exp_dir / "checkpoint.pth")


def evaluate(epoch, model, val_loader, gpu, best_acc, stats_file):
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    with torch.no_grad():
        for images, target in val_loader:
            output = model(images.cuda(gpu, non_blocking=True))
            acc1, acc5 = accuracy(
                output, target.cuda(gpu, non_blocking=True), topk=(1, 5)
            )
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
    best_acc.top1 = max(best_acc.top1, top1.avg)
    best_acc.top5 = max(best_acc.top5, top5.avg)
    stats = dict(
        epoch=epoch,
        acc1=top1.avg,
        acc5=top5.avg,
        best_acc1=best_acc.top1,
        best_acc5=best_acc.top5,
    )
    print(json.dumps(stats))
    print(json.dumps(stats), file=stats_file)


if __name__ == "__main__":
    main()
