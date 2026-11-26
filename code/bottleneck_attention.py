"""Experimental dimension-controlled attention; baseline remains unchanged."""
import math
import torch
from torch import nn
from our_models import CBAMBlock, MambaBlock

class BottleneckMambaAttention(nn.Module):
    """Retain two feature views and all spatial tokens; scan at context_dim.

    Width controls compute. The optional gate controls residual amplitude,
    not executed compute. Output channels match the existing classifier.
    """
    def __init__(self, in_channels, context_dim=256, reduction=16, state_size=64, gated=True,
                 residual=True, dt_init=False, a_max=None, pos_enc=False, grid=7,
                 scan_grid=None):
        super().__init__()
        if context_dim < 2 or context_dim % 2 or context_dim > 2 * in_channels:
            raise ValueError('context_dim must be even and in [2, 2*in_channels]')
        self.residual = residual
        self.context_dim = context_dim
        self.attention = CBAMBlock(in_channels, reduction)
        self.channel_proj = nn.Conv2d(in_channels, context_dim // 2, 1, bias=False)
        self.spatial_proj = nn.Conv2d(in_channels, context_dim // 2, 1, bias=False)
        self.mamba_block = MambaBlock(context_dim, state_size, dt_init=dt_init, a_max=a_max)
        self.restore = nn.Conv2d(context_dim, 2 * in_channels, 1, bias=False)
        self.gate = nn.Conv2d(context_dim, 1, 1) if gated else None
        # 位置编码：展平成序列后位置信息是隐含的，显式加 2D 可学习位置嵌入，
        # 让扫描能区分「哪个 token 在哪个位置」（前端补回空间排布）。
        self.pos_emb = nn.Parameter(torch.zeros(1, grid * grid, context_dim)) if pos_enc else None
        if self.pos_emb is not None:
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.grid = grid
        if scan_grid is not None and scan_grid < 1:
            raise ValueError('scan_grid must be positive')
        self.scan_grid = scan_grid
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, math.log(0.1 / 0.9))
        self.features = None
        self.gradients = None

    def forward(self, x):
        fc, fs = self.attention(x)
        base = torch.cat((fc, fs), dim=1)
        z = torch.cat((self.channel_proj(fc), self.spatial_proj(fs)), dim=1)
        if hasattr(self, 'spatial_encoder'):
            z = z + self.spatial_encoder(z)
        b, _, h, w = z.shape
        scan_z = (torch.nn.functional.interpolate(
            z, size=(min(self.scan_grid, h), min(self.scan_grid, w)),
            mode='bilinear', align_corners=False)
            if self.scan_grid is not None else z)
        scan_h, scan_w = scan_z.shape[-2:]
        tokens = scan_z.flatten(2).transpose(1, 2)
        if self.pos_emb is not None:
            if tokens.shape[1] == self.pos_emb.shape[1]:
                tokens = tokens + self.pos_emb
            else:
                # 输入尺寸与训练尺寸不同（例如测试用小图）时，把位置编码按空间插值到当前网格。
                grid = int(self.pos_emb.shape[1] ** 0.5)
                embedding = self.pos_emb.reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
                embedding = torch.nn.functional.interpolate(
                    embedding, size=(scan_h, scan_w), mode='bilinear', align_corners=False)
                tokens = tokens + embedding.permute(0, 2, 3, 1).reshape(1, scan_h * scan_w, -1)
        tokens = self.mamba_block(tokens)
        if hasattr(self, 'output_norm'):
            tokens = self.output_norm(tokens)
        correction = self.restore(tokens.transpose(1, 2).reshape(b, self.context_dim, scan_h, scan_w))
        if (scan_h, scan_w) != (h, w):
            correction = torch.nn.functional.interpolate(
                correction, size=(h, w), mode='bilinear', align_corners=False)
        if self.gate is not None:
            correction = torch.sigmoid(self.gate(z)) * correction
        self.features = base + correction if self.residual else correction
        return self.features

    def backward_hook(self, grad):
        self.gradients = grad

    def register_hooks(self):
        if self.features is None:
            raise ValueError('Run a forward pass before registering hooks')
        self.features.register_hook(self.backward_hook)
