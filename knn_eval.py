# Copyright 2022 - Valeo Comfort and Driving Assistance
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from DINO: https://github.com/facebookresearch/dino


from pathlib import Path
import argparse
import os
import random
import signal

from torch import nn
import torch
import torch.distributed
import torch.utils.data

from src.swav.logger import create_logger, create_stats
from src.vicreg.utils import handle_sigusr1, handle_sigterm


def get_arguments():
    parser = argparse.ArgumentParser(description="Perform KNN classification on previously extracted features.")
    parser.add_argument("--dataset", type=str, choices=["ImageNet", "STL10"])
    parser.add_argument("--extraction-dir", type=Path, help="path to directory with saved extracted features")
    parser.add_argument("--stats-dir", type=Path, default="./exp",
                        help='Path to the results folder, where all the logs and stats will be stored.')
    parser.add_argument("--use_cuda", action="store_true", help="Do not use it if it does not fit in GPU memory.")
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature used in the voting coefficient')
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

    # Create directory for saving experiment results + logger
    if args.rank == 0:
        args.stats_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(args.stats_dir / "train.log", rank=args.rank)
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    logger.info("The experiment results (stats + logs) will be stored in %s\n" % args.stats_dir)
    train_stats = create_stats(args.stats_dir / "eval_training_stats", args,
                               ["k", "prec1_val", "prec5_val"])

    # ======================== Part 1: load extracted features ========================= #
    logger.info(f"Load previously extracted features (test + val) at: {args.extraction_dir}")
    train_features = torch.load(args.extraction_dir / "train_features.pth")
    val_features = torch.load(args.extraction_dir / "val_features.pth")
    train_labels = torch.load(args.extraction_dir / "train_labels.pth")
    val_labels = torch.load(args.extraction_dir / "val_labels.pth")

    # =================== Part 2: we train on the extracted features =================== #
    if args.dataset == "ImageNet":
        num_classes = 1000
    elif args.dataset == "STL10":
        num_classes = 10
    else:
        raise NotImplementedError

    # Normalize features for KNN classification
    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    val_features = nn.functional.normalize(val_features, dim=1, p=2)

    # Perform KNN classification on test set
    if args.rank == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            val_features = val_features.cuda()
            train_labels = train_labels.cuda()
            val_labels = val_labels.cuda()
    for k in args.nb_knn:
        top1, top5 = knn_classifier(train_features, train_labels, val_features, val_labels, k, args.temperature,
                                    args.use_cuda, num_classes=num_classes)
        logger.info(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
        train_stats.update([k, top1, top5])


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, use_cuda, num_classes=1000):
    """
    Code from DINO: https://github.com/facebookresearch/dino
    """
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes)
    if use_cuda:
        retrieval_one_hot = retrieval_one_hot.cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx: min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


if __name__ == "__main__":
    main()
