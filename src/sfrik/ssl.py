# Copyright 2022 - Valeo Comfort and Driving Assistance
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import itertools

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import src.vicreg.resnet


class SSLMethod(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = src.vicreg.resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(args, self.embedding)

    def forward(self, *arguments):
        if self.args.multicrop is None:
            return self.two_views_forward(*arguments)
        if self.args.multicrop == "OBoW":
            return self.obow_multi_crop_forward(*arguments)
        if self.args.multicrop == "SwAV":
            return self.swav_multi_crop_forward(*arguments)
        raise NotImplementedError


class VICReg(SSLMethod):
    """
    Code from VICReg: https://github.com/facebookresearch/vicreg
    """
    def __init__(self, args):
        super().__init__(args)
        assert self.args.vicreg_sim_coeff is not None
        assert self.args.vicreg_std_coeff is not None
        assert self.args.vicreg_cov_coeff is not None
        assert self.args.multicrop is None

    def two_views_forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.vicreg_sim_coeff * repr_loss
            + self.args.vicreg_std_coeff * std_loss
            + self.args.vicreg_cov_coeff * cov_loss
        )
        return loss


class AUH(SSLMethod):
    def __init__(self, args):
        super().__init__(args)
        assert self.args.auh_sim_coeff is not None
        assert self.args.auh_unif_coeff is not None
        assert self.args.auh_scale is not None
        assert self.args.multicrop is None
        self.psi = lambda t: torch.exp(-2 * self.args.auh_scale * (1 - t))

    def two_views_forward(self, x, y):
        x = self.projector(self.backbone(x))
        x = nn.functional.normalize(x, dim=1, p=2)
        y = self.projector(self.backbone(y))
        y = nn.functional.normalize(y, dim=1, p=2)

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)

        unif_loss = torch.tensor(0, dtype=torch.float).cuda()
        for emb in [x, y]:
            gram_matrix = torch.matmul(emb, emb.t())
            gram_matrix = off_diagonal(gram_matrix)
            gram_matrix = self.psi(gram_matrix)
            unif_loss += gram_matrix.mean().log()
        unif_loss = unif_loss / 2

        loss = (
            self.args.auh_sim_coeff * repr_loss
            + self.args.auh_unif_coeff * unif_loss
        )
        return loss


class SimCLR(SSLMethod):
    def __init__(self, args):
        super().__init__(args)
        assert self.args.simclr_temp is not None
        assert self.args.multicrop is None
        self.cross_entropy = nn.CrossEntropyLoss()

    def two_views_forward(self, x, y):
        x = self.projector(self.backbone(x))
        x = nn.functional.normalize(x, dim=1, p=2)
        y = self.projector(self.backbone(y))
        y = nn.functional.normalize(y, dim=1, p=2)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)

        all_emb = torch.cat([x, y], dim=0)
        gram_matrix = torch.matmul(all_emb, all_emb.t())
        loss = torch.tensor(0, dtype=torch.float).cuda()
        for crop_idx in [0, 1]:
            logits, labels = multiview_nce_loss(gram_matrix, crop_idx, 2, self.args.simclr_temp)
            loss += self.cross_entropy(logits, labels)
        return loss / 2


class SFRIK(SSLMethod):
    def __init__(self, args):
        super().__init__(args)
        assert self.args.sfrik_sim_coeff is not None
        assert self.args.sfrik_mmd_coeff is not None
        assert self.args.sfrik_weights is not None
        self.weights = tuple(float(w) for w in args.sfrik_weights.split("-"))
        self.psi = lambda t: self.weights[0] * legendre_polynomial(self.num_features, 1, t) + \
                             self.weights[1] * legendre_polynomial(self.num_features, 2, t) + \
                             self.weights[2] * legendre_polynomial(self.num_features, 3, t)

    def mmd_loss(self, emb):
        gram_matrix = torch.mm(emb, emb.t())
        loss = self.psi(gram_matrix).mean()
        return loss

    def two_views_forward(self, x, y):
        x = self.projector(self.backbone(x))
        x = nn.functional.normalize(x, dim=1, p=2)
        y = self.projector(self.backbone(y))
        y = nn.functional.normalize(y, dim=1, p=2)

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)

        mmd_loss = (self.mmd_loss(x) + self.mmd_loss(y)) / 2

        loss = (
            self.args.sfrik_sim_coeff * repr_loss
            + self.args.sfrik_mmd_coeff * mmd_loss
        )
        return loss

    def obow_multi_crop_forward(self, img_orig, img_crops):
        bs = img_orig.shape[0]
        img_orig = self.projector(self.backbone(img_orig))
        img_orig = nn.functional.normalize(img_orig, dim=1, p=2)

        img_crops = [self.projector(self.backbone(img)) for img in img_crops]
        img_crops = [nn.functional.normalize(img, dim=1, p=2) for img in img_crops]
        img_crops = list(itertools.chain.from_iterable([[img[bs * i: bs * (i + 1)] for i in range(img.shape[0] // bs)]
                                                        for img in img_crops]))

        repr_loss = sum([F.mse_loss(img_orig, img) for img in img_crops]) / len(img_crops)

        img_orig = torch.cat(FullGatherLayer.apply(img_orig), dim=0)
        if self.args.reg_small == "True":
            img_crops = [torch.cat(FullGatherLayer.apply(img), dim=0) for img in img_crops]
            mmd_loss = (sum([self.mmd_loss(emb) for emb in img_crops]) + self.mmd_loss(img_orig)) / (len(img_crops) + 1)
        elif self.args.reg_small == "False":
            mmd_loss = self.mmd_loss(img_orig)
        else:
            raise NotImplementedError

        loss = (
            self.args.sfrik_sim_coeff * repr_loss
            + self.args.sfrik_mmd_coeff * mmd_loss
        )
        return loss

    def swav_multi_crop_forward(self, img_views, bs):
        img_views = [self.projector(self.backbone(img)) for img in img_views]
        img_views = [nn.functional.normalize(img, dim=1, p=2) for img in img_views]
        img_views = list(itertools.chain.from_iterable([[img[bs * i: bs * (i + 1)] for i in range(img.shape[0] // bs)]
                                                        for img in img_views]))

        repr_loss = sum([
            sum([F.mse_loss(img_views[view_idx], img_views[i])
                 for i in range(sum(self.args.nmb_crops)) if i != view_idx]) / (sum(self.args.nmb_crops) - 1)
            for view_idx in self.args.general_indices
        ]) / len(self.args.general_indices)

        img_views = [torch.cat(FullGatherLayer.apply(img), dim=0) for img in img_views]
        if self.args.reg_small == "True":
            mmd_loss = sum([self.mmd_loss(emb) for emb in img_views]) / len(img_views)
        elif self.args.reg_small == "False":
            mmd_loss = sum([self.mmd_loss(img_views[i])
                            for i in self.args.general_indices]) / len(self.args.general_indices)
        else:
            raise NotImplementedError

        loss = (
            self.args.sfrik_sim_coeff * repr_loss
            + self.args.sfrik_mmd_coeff * mmd_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def multiview_nce_loss(similarity_matrix, crop_idx, nmb_crops, temperature):
    """
    Multi-view NCE over a batch. similarity_matrix is assumed to be: emb @ emb.t(), where
    emb = [emb_view1, emb_view2, ...] is the concatenation of embeddings of a batch at different views.
    crop_idx is the view of the positive key.
    nmb_crops is the total number of views.
    """
    assert similarity_matrix.shape[0] % nmb_crops == 0
    bs = similarity_matrix.shape[0] // nmb_crops

    # Positive mask
    pos_grid = torch.zeros(nmb_crops, nmb_crops)
    pos_grid[:, crop_idx] = torch.ones(nmb_crops)
    pos_grid[crop_idx, crop_idx] = 0
    pos_mask = torch.kron(pos_grid, torch.eye(bs)).to(torch.bool)

    # Negative mask
    comp_neg_mask = torch.kron(torch.ones(nmb_crops, nmb_crops), torch.eye(bs)).to(torch.bool)
    comp_neg_mask[crop_idx * bs: (crop_idx + 1) * bs, :] = torch.ones((bs, comp_neg_mask.shape[1]), dtype=torch.bool)
    neg_mask = ~comp_neg_mask

    # Logits
    pos_logits = similarity_matrix[pos_mask].view(bs * (nmb_crops - 1), -1)
    neg_logits = similarity_matrix[neg_mask].view(bs * (nmb_crops - 1), -1)
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    logits /= temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    return logits, labels


def legendre_polynomial(dim, degree, t):
    if degree == 1:
        return t
    if degree == 2:
        constant = 1 / (dim - 1)
        return (1 + constant) * t**2 - constant
    if degree == 3:
        constant = 3 / (dim - 1)
        return (1 + constant) * t**3 - constant * t
    if degree == 4:
        a_1 = 1 + 6 / (dim - 1) + 3 / ((dim + 1) * (dim - 1))
        a_2 = - 6 / (dim - 1) * (1 + 1 / (dim + 1))
        a_3 = 3 / ((dim + 1) * (dim - 1))
        return a_1 * t**4 + a_2 * t**2 + a_3
    raise NotImplementedError


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]