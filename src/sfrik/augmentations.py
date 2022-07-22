# Copyright 2022 - Valeo Comfort and Driving Assistance
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from VICReg: https://github.com/facebookresearch/vicreg,
# OBoW: https://github.com/valeoai/obow
# SwAV: https://github.com/facebookresearch/swav


from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch
import random


class GaussianBlur(object):
    """
    Code from VICReg: https://github.com/facebookresearch/vicreg
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    """
    Code from VICReg: https://github.com/facebookresearch/vicreg
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    def __init__(self, size, horizontal_flip, color_jitter, gray_scale, blur, blur_prime,
                 solarization, solarization_prime, mean, std):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=horizontal_flip),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=color_jitter["brightness"],
                            contrast=color_jitter["contrast"],
                            saturation=color_jitter["saturation"],
                            hue=color_jitter["hue"]
                        )
                    ],
                    p=color_jitter["p"],
                ),
                transforms.RandomGrayscale(p=gray_scale),
                GaussianBlur(p=blur),
                Solarization(p=solarization),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=horizontal_flip),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=color_jitter["brightness"],
                            contrast=color_jitter["contrast"],
                            saturation=color_jitter["saturation"],
                            hue=color_jitter["hue"]
                        )
                    ],
                    p=color_jitter["p"],
                ),
                transforms.RandomGrayscale(p=gray_scale),
                GaussianBlur(p=blur_prime),
                Solarization(p=solarization_prime),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2


class ImageNetVICRegTrainTransform(TrainTransform):
    def __init__(self, size):
        color_jitter = {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.2,
            "hue": 0.1,
            "p": 0.8
        }
        super(ImageNetVICRegTrainTransform, self).__init__(
            size=size,
            horizontal_flip=0.5,
            color_jitter=color_jitter,
            gray_scale=0.2,
            blur=1.0,
            blur_prime=0.1,
            solarization=0.0,
            solarization_prime=0.2,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )


class ImageNetSimCLRTrainTransform(TrainTransform):
    def __init__(self, size):
        color_jitter = {
            "brightness": 0.8,
            "contrast": 0.8,
            "saturation": 0.8,
            "hue": 0.2,
            "p": 0.8
        }
        super(ImageNetSimCLRTrainTransform, self).__init__(
            size=size,
            horizontal_flip=0.5,
            color_jitter=color_jitter,
            gray_scale=0.2,
            blur=0.5,
            blur_prime=0.5,
            solarization=0.0,
            solarization_prime=0.0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )


class STL10AUHTrainTransform(TrainTransform):
    def __init__(self, size):
        color_jitter = {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.4,
            "p": 1.0
        }
        super(STL10AUHTrainTransform, self).__init__(
            size=size,
            horizontal_flip=0.5,
            color_jitter=color_jitter,
            gray_scale=0.2,
            blur=0.0,
            blur_prime=0.0,
            solarization=0.0,
            solarization_prime=0.0,
            mean=(0.441, 0.428, 0.387),
            std=(0.268, 0.261, 0.269)
        )


class StackMultipleViews:
    def __init__(self, transform, num_views):
        assert num_views >= 1
        self.transform = transform
        self.num_views = num_views

    def __call__(self, x):
        if self.num_views == 1:
            return self.transform(x).unsqueeze(dim=0)
        else:
            x_views = [self.transform(x) for _ in range(self.num_views)]
            return torch.stack(x_views, dim=0)


class CropImagePatches:
    """
    Crops from an image 3 x 3 overlapping patches.
    Code from OBoW: https://github.com/valeoai/obow
    """
    def __init__(self, patch_size, patch_jitter, num_patches, split_per_side=3):
        self.split_per_side = split_per_side
        self.patch_size = patch_size
        assert patch_jitter >= 0
        self.patch_jitter = patch_jitter
        if num_patches is None:
            num_patches = split_per_side**2
        assert 0 < num_patches <= (split_per_side**2)
        self.num_patches = num_patches

    def __call__(self, img):
        _, height, width = img.size()
        offset_y = ((height - self.patch_size - self.patch_jitter)
                    // (self.split_per_side - 1))
        offset_x = ((width - self.patch_size - self.patch_jitter)
                    // (self.split_per_side - 1))

        patches = []
        for i in range(self.split_per_side):
            for j in range(self.split_per_side):
                y_top = i * offset_y + random.randint(0, self.patch_jitter)
                x_left = j * offset_x + random.randint(0, self.patch_jitter)
                y_bottom = y_top + self.patch_size
                x_right = x_left + self.patch_size
                patches.append(img[:, y_top:y_bottom, x_left:x_right])

        if self.num_patches < (self.split_per_side * self.split_per_side):
            indices = torch.randperm(len(patches))[:self.num_patches]
            patches = [patches[i] for i in indices.tolist()]

        return torch.stack(patches, dim=0)


class ParallelTransforms:
    """
    Code from OBoW: https://github.com/valeoai/obow
    """
    def __init__(self, transform_list):
        assert isinstance(transform_list, (list, tuple))
        self.transform_list = transform_list

    def __call__(self, x):
        return [transform(x) for transform in self.transform_list]


class OBowTrainTransform(object):
    def __init__(self, only_patches, num_img_crops, image_crop_size, image_crop_range,
                 num_img_patches, img_patch_size, img_patch_preresize, img_patch_preresize_range, img_patch_jitter,
                 original_resize, original_center_crop,
                 color_jitter, color_jitter_p, gray_p, gaussian_blur_p, mean, std):

        normalize = transforms.Normalize(mean=mean, std=std)

        image_crops_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_crop_size, scale=image_crop_range),
            transforms.RandomApply([transforms.ColorJitter(**color_jitter)], p=color_jitter_p),
            transforms.RandomGrayscale(p=gray_p),
            GaussianBlur(p=gaussian_blur_p),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        image_crops_transform = StackMultipleViews(image_crops_transform, num_views=num_img_crops)

        transform_original_train = transforms.Compose([
            transforms.Resize(original_resize),
            transforms.CenterCrop(original_center_crop),
            transforms.RandomHorizontalFlip(),  # So as, to see both image views.
            transforms.ToTensor(),
            normalize
        ])
        transform_train = [transform_original_train, image_crops_transform]

        if num_img_patches > 0:
            assert num_img_patches <= 9
            image_patch_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_patch_preresize, scale=img_patch_preresize_range),
                transforms.RandomApply([transforms.ColorJitter(**color_jitter)], p=color_jitter_p),
                transforms.RandomGrayscale(p=gray_p),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                CropImagePatches(
                    patch_size=img_patch_size, patch_jitter=img_patch_jitter,
                    num_patches=num_img_patches, split_per_side=3),
            ])
            if only_patches:
                transform_train[-1] = image_patch_transform
            else:
                transform_train.append(image_patch_transform)

        self.transform_train = ParallelTransforms(transform_train)

    def __call__(self, sample):
        img_list = self.transform_train(sample)
        img_orig = img_list[0]
        img_crops = img_list[1:]
        return img_orig, img_crops


class ImageNetOBoWTrainTransform(OBowTrainTransform):
    def __init__(self, num_img_crops, image_crop_size, num_img_patches, img_patch_size):
        color_jitter = {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.4,
        }
        super(ImageNetOBoWTrainTransform, self).__init__(
            only_patches=False,
            num_img_crops=num_img_crops,
            image_crop_size=image_crop_size,
            image_crop_range=(0.08, 0.6),
            num_img_patches=num_img_patches,
            img_patch_size=img_patch_size,
            img_patch_preresize=256,
            img_patch_preresize_range=(0.6, 1.0),
            img_patch_jitter=24,
            original_resize=256,
            original_center_crop=224,
            color_jitter=color_jitter,
            color_jitter_p=0.8,
            gray_p=0.2,
            gaussian_blur_p=0.5,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )


class STL10OBoWTrainTransform(OBowTrainTransform):
    def __init__(self, num_img_crops, image_crop_size, num_img_patches, img_patch_size):
        color_jitter = {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.4,
        }
        super(STL10OBoWTrainTransform, self).__init__(
            only_patches=False,
            num_img_crops=num_img_crops,
            image_crop_size=image_crop_size,
            image_crop_range=(0.08, 0.6),
            num_img_patches=num_img_patches,
            img_patch_size=img_patch_size,
            img_patch_preresize=96,
            img_patch_preresize_range=(0.6, 1.0),
            img_patch_jitter=24,
            original_resize=96,
            original_center_crop=96,
            color_jitter=color_jitter,
            color_jitter_p=1.0,
            gray_p=0.2,
            gaussian_blur_p=0.5,
            mean=(0.441, 0.428, 0.387),
            std=(0.268, 0.261, 0.269)
        )


class SwAVTrainTransform(object):
    def __init__(self, size_crops, nmb_crops, min_scale_crops, max_scale_crops,
                 color_jitter, color_jitter_p, gray_p, gaussian_blur_p, mean, std):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        color_jitter = transforms.ColorJitter(
            brightness=color_jitter["brightness"],
            contrast=color_jitter["contrast"],
            saturation=color_jitter["saturation"],
            hue=color_jitter["hue"]
        )
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=color_jitter_p)
        rnd_gray = transforms.RandomGrayscale(p=gray_p)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])

        color_transform = [color_distort, GaussianBlur(p=gaussian_blur_p)]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __call__(self, sample):
        multi_crops = list(map(lambda trans: trans(sample), self.trans))
        return multi_crops


class ImageNetSwAVTrainTransform(SwAVTrainTransform):
    def __init__(self, size_crops, nmb_crops, min_scale_crops, max_scale_crops):
        color_jitter = {
            "brightness": 0.8,
            "contrast": 0.8,
            "saturation": 0.8,
            "hue": 0.2,
        }
        super(ImageNetSwAVTrainTransform, self).__init__(
            size_crops=size_crops,
            nmb_crops=nmb_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            color_jitter=color_jitter,
            color_jitter_p=0.8,
            gray_p=0.2,
            gaussian_blur_p=0.5,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )


class STL10SwAVTrainTransform(SwAVTrainTransform):
    def __init__(self, size_crops, nmb_crops, min_scale_crops, max_scale_crops):
        color_jitter = {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.4,
        }
        super(STL10SwAVTrainTransform, self).__init__(
            size_crops=size_crops,
            nmb_crops=nmb_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            color_jitter=color_jitter,
            color_jitter_p=1.0,
            gray_p=0.2,
            gaussian_blur_p=0.5,
            mean=(0.441, 0.428, 0.387),
            std=(0.268, 0.261, 0.269)
        )