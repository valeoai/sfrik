# Copyright 2022 - Valeo Comfort and Driving Assistance
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from OBoW: https://github.com/valeoai/obow


import torchvision.datasets as datasets
import torch.utils.data


def build_label_index(labels):
    """
    Code from OBoW: https://github.com/valeoai/obow
    """
    label2inds = {}
    for idx, label in enumerate(labels):
        if label2inds.get(label) is None:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds


class ImageNetSubset(datasets.ImageFolder):
    """
    Code from OBoW: https://github.com/valeoai/obow
    """
    def __init__(self, *args, start=0, subset=-1, **kwargs):

        super(ImageNetSubset, self).__init__(*args, **kwargs)

        if subset > 0:
            all_indices = []
            label2inds = build_label_index(self.targets)
            for img_indices in label2inds.values():
                assert len(img_indices) >= subset
                all_indices += img_indices[start:start+subset]
            self.imgs = [self.imgs[idx] for idx in all_indices]
            self.samples = [self.samples[idx] for idx in all_indices]
            self.targets = [self.targets[idx] for idx in all_indices]
            assert len(self) == (subset * 1000)


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


class ReturnIndexDatasetSubset(ReturnIndexDataset, ImageNetSubset):
    def __init__(self, *args, **kwargs):
        ImageNetSubset.__init__(self, *args, **kwargs)


class ReturnIndexStlDataset(datasets.STL10):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexStlDataset, self).__getitem__(idx)
        return img, idx


class FeaturesLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        super(FeaturesLabelsDataset).__init__()
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __getitem__(self, item):
        return self.features[item], self.labels[item]

    def __len__(self):
        return len(self.features)
