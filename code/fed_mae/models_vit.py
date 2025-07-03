# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import PatchEmbed, Block

class MultiHeadVisionTransformer(VisionTransformer):
    """
    多头版本的 ViT，继承自 VisionTransformer。
    在保持原有 patch_embed、blocks、pos_embed 等命名不变的基础上，
    将最后的分类层 head 替换为多个头 (ModuleList)。
    """
    def __init__(self, nb_classes=48, **kwargs):
        """
        参数解释:
          nb_classes: 总类别数
          global_pool: 是否使用全局池化, 与原先VisionTransformer一致
          **kwargs: 其余将传递给父类 VisionTransformer (如 embed_dim, depth, etc.)
        """
        # 1) 调用父类构造
        super().__init__( **kwargs)

        # 2) 移除或替换原先单一 head
        #    父类里通常: self.head = nn.Linear(...)
        #    我们这里改为多头
        self.head = None  # 删除父类的 head

        # 3) 定义多头, 每个头输出1维(二分类或logit), 最后会合并为 (batch_size, nb_classes)
        embed_dim = self.embed_dim  # 父类已有 self.embed_dim
        self.multi_heads = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(nb_classes)
        ])

        # 也可以根据需求保留 self.norm、self.fc_norm 等父类结构不变

    def forward(self, x):
        """
        对外的 forward:
          1) 先用父类的 forward_features() 得到图像最终特征 (batch_size, embed_dim).
          2) 用多个头进行输出, 拼接成 (batch_size, nb_classes).
        """
        # 1) 提取特征 (同父类)
        feats = super().forward_features(x)
        # feats shape: (batch_size, embed_dim)

        # 2) 多头输出
        out_list = []
        for head_i in self.multi_heads:
            out_list.append(head_i(feats))  # 每个是 (batch_size, 1)

        # 拼成 (batch_size, nb_classes)
        out = torch.cat(out_list, dim=1)
        return out


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
def multi_head_vit_base_patch16(nb_classes=48, **kwargs):
    model = MultiHeadVisionTransformer(
        nb_classes=nb_classes,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model