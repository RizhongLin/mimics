from abc import ABC
from typing import *

import pytorch_lightning as pl
import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from monai.inferers import sliding_window_inference
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
import numpy as np


def kl_divergence(alpha, num_classes, device=None):
    # ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    ones = torch.ones_like(alpha)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


class EvidentialUNet(pl.LightningModule, ABC):
    def __init__(self,
                 learning_rate: float,
                 optimizer_class,
                 net_config: str = 'default',
                 spatial_dims: int = 3,
                 in_channels: int = 4,
                 out_channels: int = 4,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 dropout: float = 0.0, ):
        super(EvidentialUNet, self).__init__()
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dropout = dropout
        self.net = self.get_net(net_config)
        self.dice_loss = monai.losses.DiceLoss()

    def get_net(self, net_config: str):
        if net_config == "default":
            net = monai.networks.nets.UNet(
                spatial_dims=self.spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                channels=(16, 32, 64, 128),
                strides=(2, 2, 2),
                num_res_units=2,
                dropout=self.dropout,
            )
        elif net_config == "shallow":
            net = monai.networks.nets.UNet(
                spatial_dims=self.spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                channels=(4, 8, 16),
                strides=(2, 2),
                num_res_units=2,
                dropout=self.dropout
            )
        elif net_config == "large":
            net = monai.networks.nets.UNet(
                spatial_dims=self.spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                channels=(16, 32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                dropout=self.dropout
            )
        else:
            raise NotImplementedError

        return net

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    @staticmethod
    def prepare_batch(batch):
        return torch.cat([batch[x][tio.DATA] for x in ['flair', 't1', 't1ce', 't2']], dim=1).float(), \
               batch['seg'][tio.DATA].float()

    def forward(self, evidence):
        alpha = evidence.nan_to_num() + 1

        S = torch.sum(alpha, dim=1, keepdim=True)  # S: [B,1,H,W]; Dirichlet strength
        u = torch.ones_like(S) * self.out_channels / S  # uncertainty: [B,1,H,W]
        p = alpha / S  # probability: [B,self.out_channels,H,W]

        return {"probability": p, "uncertainty": u, "alpha": alpha, "dirichlet strength": S}
        # return p, u, alpha, S

    def get_loss(self, result, y, lambda_p=0.2, lambda_s=1):
        p, alpha, S = result["probability"], result["alpha"], result["dirichlet strength"]

        # ice loss
        ice_loss = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        ice_loss = ice_loss[abs(ice_loss) < float('inf')].mean()  # np.nanmean()
        # ice_loss = ice_loss.mean()

        # KL loss
        num_classes = y.shape[1]
        kl_alpha = (alpha - 1) * (1 - y) + 1
        KL_loss = kl_divergence(alpha=kl_alpha, num_classes=num_classes, device=y.device)
        KL_loss = KL_loss[abs(KL_loss) < float('inf')].mean()  # np.nanmean()
        # KL_loss = KL_loss.mean()

        # dice loss
        dice_loss = self.dice_loss(p.nan_to_num(), y)  # torch.nan_to_num()
        # dice_loss = self.dice_loss(p, y)

        loss = ice_loss + lambda_p * KL_loss + lambda_s * dice_loss
        return loss

    def training_step(self, batch, batch_idx):
        image, ground_truth = self.prepare_batch(batch)
        evidence = F.softplus(self.net(image))  # get the evidence

        result = self.forward(evidence)
        loss = self.get_loss(result, ground_truth)

        self.log('train_loss', loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        image, ground_truth = self.prepare_batch(batch)
        evidence = F.softplus(sliding_window_inference(image,
                                                       roi_size=(128, 128, 128),
                                                       sw_batch_size=self.batch_size,
                                                       predictor=self.net,
                                                       overlap=0.25).nan_to_num())

        result = self.forward(evidence)
        loss = self.get_loss(result, ground_truth)

        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        image, ground_truth = self.prepare_batch(batch)
        evidence = F.softplus(sliding_window_inference(image,
                                                       roi_size=(128, 128, 128),
                                                       sw_batch_size=self.batch_size,
                                                       predictor=self.net,
                                                       overlap=0.25).nan_to_num())

        result = self.forward(evidence)
        ground_truth = ground_truth.argmax(dim=1)
        p = result["probability"].argmax(dim=1)
        loss = self.dice_loss(p, ground_truth)

        self.log('test_loss', loss, prog_bar=True, batch_size=self.batch_size)
        return loss


if __name__ == '__main__':
    model = EvidentialUNet(learning_rate=0.001, net_config="large", optimizer_class=torch.optim.Adam)
