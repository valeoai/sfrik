# Copyright 2022 - Valeo Comfort and Driving Assistance
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from DINO: https://github.com/facebookresearch/dino


from pathlib import Path
import argparse

from torchvision import transforms
import torch
import torch.distributed
import torch.utils.data
from torchvision.transforms import InterpolationMode

from src.swav.logger import create_logger
import src.vicreg.resnet
from src.vicreg.utils import MetricLogger
import src.vicreg.distributed as dist


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Given a fixed backbone, extract features from ImageNet or STL10."
    )

    # Data
    parser.add_argument("--dataset", type=str, choices=["ImageNet", "STL10"])
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    parser.add_argument("--subset", type=int, default=-1, help="Take a fix number of images per class (example 260)"
                                                               "to construct the training set.")
    parser.add_argument("--val-dataset", choices=["train", "val"],
                        help="Choice of the test dataset."
                             "Choose 'val' for extracting features of the usual ImageNet validation set."
                             "Choose 'train' for extracting features from a subset of the ImageNet train set.")
    parser.add_argument("--val-subset", type=int, help="Size of validation set when setting '--val-dataset train'."
                                                       "Take a fix number of images per class.")

    # Model
    parser.add_argument("--arch", type=str)
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")

    # Feature extractor
    parser.add_argument("--batch-size", default=256, type=int, metavar="N", help="mini-batch size")
    parser.add_argument("--exp-dir", type=Path, default="./exp", metavar="DIR",
                        help="features are saved in this directory")

    # Running
    parser.add_argument("--workers", default=8, type=int, metavar="N", help="number of data loader workers")
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


@torch.no_grad()
def extract_features(model, data_loader):
    """
    Code from DINO: https://github.com/facebookresearch/vicreg
    """
    metric_logger = MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()

    # Set up distributed mode
    torch.backends.cudnn.benchmark = True
    dist.init_distributed_mode(args)
    gpu = torch.device(args.device)

    # Save dir and logger
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(args.exp_dir / "train.log", rank=args.rank)
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    logger.info("The experiment directory is %s\n" % args.exp_dir)

    # Backbone
    backbone, _ = src.vicreg.resnet.__dict__[args.arch](zero_init_residual=True)
    logger.info(f"Load pretrained weights at: {args.pretrained}")
    state_dict = torch.load(args.pretrained, map_location="cpu")
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    assert missing_keys == [] and unexpected_keys == []
    logger.info(f"Extracting features from scratch")
    backbone = backbone.cuda(gpu)
    backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[gpu])
    backbone.eval()

    # Dataset
    if args.dataset == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        traindir = args.data_dir / "train"
        train_dataset = src.sfrik.dataset.ReturnIndexDatasetSubset(traindir, subset=args.subset, transform=transform)
        train_labels = torch.tensor([s[-1] for s in train_dataset.samples]).long()

        if args.val_dataset == "val":
            valdir = args.data_dir / "val"
            val_dataset = src.sfrik.dataset.ReturnIndexDataset(valdir, transform=transform)
        elif args.val_dataset == "train":
            valdir = args.data_dir / "train"
            val_dataset = src.sfrik.dataset.ReturnIndexDatasetSubset(valdir, start=args.subset + 1, subset=args.val_subset,
                                                                     transform=transform)
        else:
            raise NotImplementedError
        val_labels = torch.tensor([s[-1] for s in val_dataset.samples]).long()
    elif args.dataset == "STL10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.441, 0.428, 0.387], std=[0.268, 0.261, 0.269]
            ),
        ])
        train_dataset = src.sfrik.dataset.ReturnIndexStlDataset(args.data_dir, split="train", transform=transform,
                                                                download=True)
        val_dataset = src.sfrik.dataset.ReturnIndexStlDataset(args.data_dir, split="test", transform=transform,
                                                              download=True)
        train_labels = torch.tensor(train_dataset.labels).long()
        val_labels = torch.tensor(val_dataset.labels).long()
    else:
        raise NotImplementedError
    print("Size of train dataset:", len(train_dataset))
    print("Size of test dataset:", len(val_dataset))

    # Data loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    # Feature extraction
    logger.info("Extracting features for train set...")
    train_features = extract_features(backbone, train_loader)
    logger.info("Extracting features for val set...")
    val_features = extract_features(backbone, val_loader)

    # Save extracted features
    if args.rank == 0:
        torch.save(train_features, args.exp_dir / "train_features.pth")
        torch.save(val_features, args.exp_dir / "val_features.pth")
        torch.save(train_labels, args.exp_dir / "train_labels.pth")
        torch.save(val_labels, args.exp_dir / "val_labels.pth")
