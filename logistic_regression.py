# Copyright 2022 - Valeo Comfort and Driving Assistance
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import os
import random
import signal
import time

from torch import nn, optim
import torch
import torch.distributed
import torch.utils.data

from src.swav.logger import create_logger, create_stats
from src.vicreg.utils import AverageMeter, handle_sigusr1, handle_sigterm, accuracy
from src.sfrik.dataset import FeaturesLabelsDataset


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Logistic regression on previously extracted features (i.e., linear probing with frozen features)."
    )

    # Data
    parser.add_argument("--dataset", type=str, choices=["ImageNet", "STL10"])
    parser.add_argument("--extraction-dir", type=Path, help="if not None, use the given dir path where features have "
                                                            "been previously extracted")

    # Checkpoint + results directory
    parser.add_argument("--exp-dir", type=Path, default="./exp", metavar="DIR", help="path to checkpoint directory")
    parser.add_argument("--stats-dir", type=Path, default="./exp",
                        help='Path to the results folder, where all the logs and stats will be stored.')

    # Optim
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--batch-size", default=256, type=int, metavar="N", help="mini-batch size")
    parser.add_argument("--lr-head", default=0.3, type=float, metavar="LR", help="classifier base learning rate")
    parser.add_argument("--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay")

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
    # single-node distributed training
    args.rank = 0
    args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    # Set up distributed mode
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    # Create directory for saving checkpoints and experiment results + logger
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        args.stats_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(args.stats_dir / "train.log", rank=args.rank)
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    logger.info("The experiment checkpoints will be stored in %s\n" % args.exp_dir)
    logger.info("The experiment results (stats + logs) will be stored in %s\n" % args.stats_dir)
    train_stats = create_stats(args.stats_dir / "eval_training_stats", args,
                               ["epoch", "loss", "acc1", "acc5", "best_acc1", "best_acc5"])

    # ======================== Part 1: load extracted features ========================= #
    logger.info(f"Load previously extracted features (test + val) at: {args.extraction_dir}")
    train_features = torch.load(args.extraction_dir / "train_features.pth")
    val_features = torch.load(args.extraction_dir / "val_features.pth")
    train_labels = torch.load(args.extraction_dir / "train_labels.pth")
    val_labels = torch.load(args.extraction_dir / "val_labels.pth")

    # =================== Part 2: we train on the extracted features =================== #
    # Logistic regression model
    embedding = train_features.shape[-1]
    if args.dataset == "ImageNet":
        head_model = nn.Linear(embedding, 1000)
    elif args.dataset == "STL10":
        head_model = nn.Linear(embedding, 10)
    else:
        raise NotImplementedError
    head_model.weight.data.normal_(mean=0.0, std=0.01)
    head_model.bias.data.zero_()
    head_model = head_model.cuda(gpu)
    head_model = torch.nn.parallel.DistributedDataParallel(head_model, device_ids=[gpu])
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # Optimization
    param_groups = [dict(params=head_model.parameters(), lr=args.lr_head)]
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Automatically resume from checkpoint if it exists
    if (args.exp_dir / "checkpoint.pth").is_file():
        if args.rank == 0:
            logger.info(f"Resuming from checkpoint found at: {args.exp_dir / 'checkpoint.pth'}")
        ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        head_model.load_state_dict(ckpt["head_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        if args.rank == 0:
            logger.info("Found no checkpoint: start from scratch")
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code - Only the extracted features + labels
    train_dataset = FeaturesLabelsDataset(train_features, train_labels)
    val_dataset = FeaturesLabelsDataset(val_features, val_labels)

    # Data loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    # Training
    for epoch in range(start_epoch, args.epochs):
        logger.info("============ Starting epoch %i ... ============" % epoch)
        head_model.train()
        train_sampler.set_epoch(epoch)
        batch_time = AverageMeter("batch_time")
        data_time = AverageMeter("data_time")
        loss_meter = AverageMeter("loss")
        end = time.time()
        for step, (features, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            data_time.update(time.time() - end)

            output = head_model(features.cuda(gpu, non_blocking=True))
            loss = criterion(output, target.cuda(gpu, non_blocking=True))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), features.shape[0])
            batch_time.update(time.time() - end)

            if args.rank == 0 and step % 30 == 0:
                torch.distributed.reduce(loss.div_(args.world_size), 0)
                pg = optimizer.param_groups
                lr_head = pg[0]["lr"]
                logger.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Lr: {lr_head:.4f}".format(
                        epoch,
                        step,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=loss_meter,
                        lr_head=lr_head,
                    )
                )
            end = time.time()

        # evaluate
        head_model.eval()
        if args.rank == 0:
            top1 = AverageMeter("Acc@1")
            top5 = AverageMeter("Acc@5")
            with torch.no_grad():
                for features, target in val_loader:
                    output = head_model(features.cuda(gpu, non_blocking=True))
                    acc1, acc5 = accuracy(
                        output, target.cuda(gpu, non_blocking=True), topk=(1, 5)
                    )
                    top1.update(acc1[0].item(), features.size(0))
                    top5.update(acc5[0].item(), features.size(0))
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)

            logger.info(
                "Test:\t"
                "Acc@1 {top1.avg:.3f}\t"
                "Acc@5 {top5.avg:.3f}\t"
                "Best Acc@1 so far {best_acc.top1:.1f}\t"
                "Best Acc@5 so far {best_acc.top5:.1f}".format(
                    batch_time=batch_time, top1=top1, top5=top5, best_acc=best_acc))

            train_stats.update([epoch, loss_meter.avg, top1.avg, top5.avg, best_acc.top1, best_acc.top5])

        scheduler.step()
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                best_acc=best_acc,
                head_model=head_model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
            )
            torch.save(state, args.exp_dir / "checkpoint.pth")


if __name__ == "__main__":
    main()
