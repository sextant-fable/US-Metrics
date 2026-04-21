"""Minimal TinyUSFM backbone for metric inference."""

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention as BaseAttn
from timm.models.vision_transformer import DropPath, Mlp


class Attention(BaseAttn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity = nn.Identity()

    def forward(self, x, return_latent=False):
        bsz, n_tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            bsz, n_tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.identity(attn)
        if return_latent:
            return None
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, n_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_latent=False):
        if return_latent:
            return self.attn(self.norm1(x), return_latent=return_latent)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        global_pool=False,
        distilled=False,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **kwargs,
        )
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.num_heads = num_heads
        self.depth = depth
        self.global_pool = global_pool
        self.distilled = distilled
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm
            if self.distilled:
                self.dist_norm = norm_layer(embed_dim)
                if self.num_classes > 0:
                    self.head_dist = nn.Linear(self.embed_dim, self.num_classes)
                else:
                    self.head_dist = nn.Identity()
        self.init_weights("")

    def forward_features(self, x):
        bsz = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            if self.distilled:
                x_dist, x = x[:, 0, :], x[:, 1:, :].mean(dim=1)
                return self.fc_norm(x), self.dist_norm(x_dist)
            x = x[:, 1:, :].mean(dim=1)
            return self.fc_norm(x)

        x = self.norm(x)
        return x[:, 0]


class TinyUSFM(nn.Module):
    def __init__(self, num_classes=0, **kwargs):
        super().__init__()
        self.model = VisionTransformer(
            patch_size=16,
            embed_dim=192,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes,
            **kwargs,
        )

    def forward(self, x):
        return self.model(x)

