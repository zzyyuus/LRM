import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2d(nn.Module):    #第二个消融
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, depthwise=True, **kwargs):
        super(BasicBlock, self).__init__()
        if depthwise:
            self.conv1 = DepthwiseSeparableConv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = DepthwiseSeparableConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            # 全卷积对照：与 ResNet-18 原始残差单元一致，用于和全卷积主干公平比较
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet_mamba_attention(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=10,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 context_dim=None,
                 context_residual=True,
                 context_gated=True,
                 depthwise=True,
                 refine_stage='layer4',
                 context_dt_init=False,
                 context_a_max=None,
                 pool='gap',
                 context_pos_enc=False,
                 context_scan_grid=None):
        super(ResNet_mamba_attention, self).__init__()
        if refine_stage not in ('layer3', 'layer3_then_layer4', 'layer4'):
            raise ValueError("refine_stage must be 'layer3', 'layer3_then_layer4', or 'layer4'")
        if pool not in ('gap', 'flat'):
            raise ValueError("pool must be 'gap' or 'flat'")
        if pool == 'flat' and refine_stage != 'layer4':
            raise ValueError('pool=flat 目前只支持 refine_stage=layer4（7x7 网格）')
        self.refine_stage = refine_stage
        self.pool = pool
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group
        self.depthwise = depthwise

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.include_top = include_top

        # layer3 输出 stride=16（224 输入下 14x14=196 token，感受野尚未覆盖全图），
        # layer4 输出 stride=32（7x7=49 token，感受野已全局）。精修放在哪一级由 refine_stage 决定。
        stage_channels = (256 if refine_stage in ('layer3', 'layer3_then_layer4') else 512) * block.expansion
        head_channels = (512 if refine_stage == 'layer3_then_layer4' else
                         (256 if refine_stage == 'layer3' else 512)) * block.expansion
        if refine_stage == 'layer3_then_layer4':
            # The attention module returns two stage-3 views. Fuse them back
            # to the stage-3 width so the normal stage-4 blocks can follow.
            self.pre_layer4_fuse = nn.Sequential(
                nn.Conv2d(2 * stage_channels, stage_channels, 1, bias=False),
                nn.BatchNorm2d(stage_channels), nn.ReLU(inplace=True))

        if self.include_top:
            if pool == 'flat':
                # 位置感知分类头：不做全局池化，直接展开 7x7x2C，保留空间排布。
                # 用自适应池化固定到 7x7，使任意输入尺寸都能展开到同一维度。
                self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
                self.fc = nn.Linear(2 * head_channels * 49, num_classes)
            else:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
                self.fc = nn.Linear(head_channels if refine_stage == 'layer3_then_layer4'
                                    else 2 * head_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.conv_layer = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                        self.layer1, self.layer2, self.layer3, self.layer4)

        if context_dim is None:
            self.mamba_attention = MambaAttention(stage_channels)
        else:
            self.mamba_attention = BottleneckMambaAttention(
                stage_channels, context_dim=context_dim,
                residual=context_residual, gated=context_gated,
                dt_init=context_dt_init, a_max=context_a_max,
                pos_enc=context_pos_enc,
                grid=(context_scan_grid if context_scan_grid is not None else
                      (14 if refine_stage in ('layer3', 'layer3_then_layer4') else 7)),
                scan_grid=context_scan_grid)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group,
                            depthwise=self.depthwise))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                depthwise=self.depthwise))

        return nn.Sequential(*layers)

    def last_layer(self, x):
        if self.pool == 'flat':
            x = self.avgpool(x)   # 固定到 7x7，保留空间排布
            return self.fc(torch.flatten(x, 1))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.refine_stage == 'layer4':
            x = self.layer4(x)
        x = self.mamba_attention(x)  #第一个消融，去掉mamba，
        if self.refine_stage == 'layer3_then_layer4':
            x = self.layer4(self.pre_layer4_fuse(x))
       # print(x.shape)
        if self.include_top:
            x = self.last_layer(x)
        return x

class MambaBlock(nn.Module):
    """Equations (8)-(11), with causal, length-preserving depthwise Conv1D.

    The paper does not specify padding. Left padding is chosen to preserve
    the stated forward-only information flow. No extra conv_linear is used.
    """
    def __init__(self, d_model, state_size=64, device=None, dt_init=False, a_max=None):
        super().__init__()
        inner = 2 * d_model
        self.norm = RMSNorm(d_model, device=device)
        self.inp_proj = nn.Linear(d_model, inner, bias=False, device=device)
        self.conv = nn.Conv1d(inner, inner, 3, groups=inner,
                              bias=False, device=device)
        self.S6 = S6(inner, state_size, device=device, dt_init=dt_init, a_max=a_max)
        self.D = nn.Linear(d_model, inner, bias=False, device=device)
        self.out_proj = nn.Linear(inner, d_model, bias=False, device=device)

    def forward(self, x):
        z = self.norm(x)
        u = self.inp_proj(z).transpose(1, 2)
        u = F.silu(self.conv(F.pad(u, (2, 0)))).transpose(1, 2)
        y = self.S6(u)
        return self.out_proj(F.silu(y) * F.silu(self.D(z)))

class S6(nn.Module):
    """Reference selective recurrence from equations (9)-(10).

    State resets per image/call, propagates across tokens, and never across
    batch entries. A Python loop favors correctness over fused-kernel speed.
    """
    def __init__(self, d_model, state_size=64, device=None, dt_init=False, a_max=None,
                 dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        # dt_init=True 时回到参考实现的参数化：Δ = softplus(线性 + 偏置)，偏置按
        # 对数均匀在 [dt_min, dt_max] 初始化，使状态分量覆盖从 ~1 到 ~0 的多种
        # 记忆尺度。否则保持原行为（softplus 零偏置 ≈ 0.693，衰减过快）。
        self.fc1 = nn.Linear(d_model, d_model, bias=dt_init, device=device)
        if dt_init:
            span = (torch.rand(d_model, device=device)
                    * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))
                    + torch.log(torch.tensor(dt_min)))
            dt = span.exp()
            with torch.no_grad():
                self.fc1.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.fc2 = nn.Linear(d_model, state_size, bias=False, device=device)
        self.fc3 = nn.Linear(d_model, state_size, bias=False, device=device)
        # Initialization is not specified in the paper: use negative 1..S。
        # a_max 可把量级压到 [1, a_max]，减小 Δ·|A| 使更多状态分量在训练后仍然存活。
        grid = torch.linspace(1.0, float(a_max) if a_max else float(state_size),
                              state_size, device=device, dtype=torch.float32)
        self.A_log = nn.Parameter(grid.log().repeat(d_model, 1))
        self.selective = True

    def to_nonselective(self):
        """转为 S4 式**非选择性** SSM：Δ、B、C 不再由输入决定。

        这是 Reviewer 2 要求的「同参对照」：外壳、通道数、状态维、扫描方式都不变，
        只把输入相关的选择性去掉，用来回答「收益究竟来自选择性状态空间，还是来自
        这条序列通路本身」。
        """
        if not self.selective:
            return self
        with torch.no_grad():
            self.delta_static = nn.Parameter(
                self.fc1.bias.detach().clone() if self.fc1.bias is not None
                else torch.zeros(self.d_model))
            self.B_static = nn.Parameter(self.fc2.weight.detach().mean(1).clone())
            self.C_static = nn.Parameter(self.fc3.weight.detach().mean(1).clone())
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        self.selective = False
        return self

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.d_model or x.shape[1] == 0:
            raise ValueError("Expected nonempty [batch, length, d_model] input")
        # Keep the recurrence in float32 for low precision inputs.
        work_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        if self.selective:
            delta = F.softplus(self.fc1(x)).to(work_dtype)
            B = self.fc2(x).to(work_dtype)
            C = self.fc3(x).to(work_dtype)
        else:
            batch, length = x.shape[0], x.shape[1]
            delta = F.softplus(self.delta_static).to(work_dtype).view(1, 1, -1).expand(batch, length, -1)
            B = self.B_static.to(work_dtype).view(1, 1, -1).expand(batch, length, -1)
            C = self.C_static.to(work_dtype).view(1, 1, -1).expand(batch, length, -1)
        u = x.to(work_dtype)
        A = -self.A_log.to(work_dtype).exp()
        decay = torch.exp(delta.unsqueeze(-1) * A)
        drive = delta.unsqueeze(-1) * u.unsqueeze(-1) * B.unsqueeze(-2)
        backend = getattr(self, 'scan_backend', 'sequential')
        if backend == 'parallel_chunked':
            h = self._chunked_scan(decay, drive)
        elif backend == 'parallel_checkpoint':
            from torch.utils.checkpoint import checkpoint
            h = checkpoint(self._parallel_scan, decay, drive, use_reentrant=False)
        elif backend == 'parallel':
            h = self._parallel_scan(decay, drive)
        else:
            state = u.new_zeros(x.shape[0], self.d_model, self.state_size)
            values = []
            for t in range(x.shape[1]):
                state = decay[:, t] * state + drive[:, t]
                values.append(state)
            h = torch.stack(values, dim=1)
        return (h * C.unsqueeze(-2)).sum(-1).to(x.dtype)

    @staticmethod
    def _parallel_prefix(decay, drive):
        """Associative scan of h[t] = decay[t]*h[t-1] + drive[t].

        Each doubling stage combines adjacent affine updates. It uses O(log L)
        sequential launches, at a higher peak-memory cost than the reference
        loop. Kept opt-in until full-size MPS memory/speed is measured.
        """
        length = drive.shape[1]
        step = 1
        a, b = decay, drive
        while step < length:
            a, b = (torch.cat((a[:, :step], a[:, step:] * a[:, :-step]), dim=1),
                    torch.cat((b[:, :step], b[:, step:] + a[:, step:] * b[:, :-step]), dim=1))
            step *= 2
        return a, b

    @staticmethod
    def _parallel_scan(decay, drive):
        return S6._parallel_prefix(decay, drive)[1]

    @staticmethod
    def _chunked_scan(decay, drive, chunk_size=16):
        from torch.utils.checkpoint import checkpoint
        previous = drive.new_zeros(drive.shape[0], drive.shape[2], drive.shape[3])
        chunks = []

        def run_chunk(a, b, initial):
            prefix_a, prefix_b = S6._parallel_prefix(a, b)
            return prefix_a * initial.unsqueeze(1) + prefix_b

        for start in range(0, drive.shape[1], chunk_size):
            block = checkpoint(run_chunk, decay[:, start:start + chunk_size],
                               drive[:, start:start + chunk_size], previous,
                               use_reentrant=False)
            chunks.append(block)
            previous = block[:, -1]
        return torch.cat(chunks, dim=1)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class ParallelRecalibration(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ParallelRecalibration, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        x1 = x * channel_attention
        spatial_attention = self.spatial_attention(x)
        x2 = x * spatial_attention
        return x1, x2

class MambaAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, state_size=64):
        super(MambaAttention, self).__init__()

        self.mamba_block = MambaBlock(2 * in_channels, state_size)
        self.attention = ParallelRecalibration(in_channels, reduction)

        # 用于存储前向特征和梯度
        self.features = None
        self.gradients = None

    def forward(self, x):
        B, C, H, W = x.shape
        channel_attention, spatial_attention = self.attention(x)
        x = torch.cat((channel_attention, spatial_attention), dim=1)
        # Eq. (7): one channel descriptor per spatial location, row-major.
        in_mb = x.flatten(2).transpose(1, 2)
        out_mb = self.mamba_block(in_mb)
        out_mb = out_mb.transpose(1, 2).reshape(B, 2 * C, H, W)

        # 保存特征图以用于梯度捕获
        self.features = out_mb
        #self.features = x   #第三个消融

        return out_mb
        #return x



    def backward_hook(self, grad):
        """反向传播时捕获梯度"""
        self.gradients = grad

    def register_hooks(self):
        """为特征图注册钩子"""
        if self.features is not None:
            self.features.register_hook(self.backward_hook)
        else:
            raise ValueError("Features have not been computed. Run a forward pass before registering hooks.")

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
        self.attention = ParallelRecalibration(in_channels, reduction)
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

def build_model(num_classes=12):
    """Build the complete model used in the paper (depth 18, context width 128)."""
    model = ResNet_mamba_attention(
        BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
        context_dim=128, context_residual=True, context_gated=True,
        depthwise=True, refine_stage="layer4", context_dt_init=True,
        context_a_max=None, pool="gap", context_pos_enc=True,
        context_scan_grid=None,
    )
    nn.init.constant_(model.mamba_attention.gate.bias, math.log(0.05 / 0.95))
    model.mamba_attention.spatial_encoder = nn.Conv2d(
        128, 128, 3, padding=1, groups=128, bias=False
    )
    for module in model.modules():
        if hasattr(module, "_parallel_scan") and not hasattr(module, "scan_backend"):
            module.scan_backend = "parallel_chunked"
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = build_model(args.num_classes).to(args.device).eval()
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, 224, 224, device=args.device))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Logits shape:", tuple(logits.shape))
