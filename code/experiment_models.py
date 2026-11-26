"""Explicit experimental variants. No pretrained weights are downloaded."""
import math
import torch
from torch import nn
from torchvision import models
from our_models import (CBAMBlock, RMSNorm,
                        resnet18_mamba_attention,resnet34_mamba_attention,resnet50_mamba_attention)
from bottleneck_attention import BottleneckMambaAttention

class TwoViews(nn.Module):
    def forward(self,x):return x,x

class AttentionOnly(nn.Module):
    """去掉 Mamba 的对照：只保留双路通道-空间注意力拼接，维度与含 Mamba 版本一致。"""
    def __init__(self,c,reduction=16):
        super().__init__();self.attention=CBAMBlock(c,reduction)
    def forward(self,x):
        fc,fs=self.attention(x);return torch.cat((fc,fs),1)

class LeftOnly(nn.Module):
    def __init__(self,left):super().__init__();self.left=left
    def forward(self,x):return torch.cat(self.left(x),1)

class StandardCBAMViews(nn.Module):
    """Standard avg/max channel then avg/max spatial CBAM, duplicated to 2C."""
    def __init__(self,c):
        super().__init__();self.mlp=nn.Sequential(nn.Conv2d(c,c//16,1,bias=False),nn.ReLU(),nn.Conv2d(c//16,c,1,bias=False));self.spatial=nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self,x):
        x=x*torch.sigmoid(self.mlp(x.mean((2,3),keepdim=True))+self.mlp(x.amax((2,3),keepdim=True)))
        x=x*torch.sigmoid(self.spatial(torch.cat((x.mean(1,keepdim=True),x.amax(1,keepdim=True)),1)))
        return x,x

class TokenMLP(nn.Module):
    """Token-wise negative control; intentionally no spatial mixing."""
    def __init__(self,k,target_params):
        super().__init__();hidden=max(1,round(target_params/(2*k)))
        self.net=nn.Sequential(nn.LayerNorm(k),nn.Linear(k,hidden,bias=False),nn.GELU(),nn.Linear(hidden,k,bias=False))
    def forward(self,x):return self.net(x)

class AttentionMixer(nn.Module):
    def __init__(self,k,target_params):
        super().__init__();self.norm=nn.LayerNorm(k);self.attn=nn.MultiheadAttention(k,4,bias=False,batch_first=True)
        hidden=max(1,round((target_params-4*k*k)/(2*k)))
        self.ffn=nn.Sequential(nn.LayerNorm(k),nn.Linear(k,hidden,bias=False),nn.GELU(),nn.Linear(hidden,k,bias=False))
    def forward(self,x):
        n=self.norm(x);x=x+self.attn(n,n,n,need_weights=False)[0];return x+self.ffn(x)

class ScanAttention(BottleneckMambaAttention):
    def __init__(self,c,k,directions):super().__init__(c,k);self.directions=directions
    def forward(self,x):
        fc,fs=self.attention(x);base=torch.cat((fc,fs),1)
        z=torch.cat((self.channel_proj(fc),self.spatial_proj(fs)),1)
        b,k,h,w=z.shape;ys=[]
        for axis,reverse in self.directions:
            zz=z if axis=='row' else z.transpose(2,3)
            seq=zz.flatten(2).transpose(1,2)
            if reverse:seq=seq.flip(1)
            out=self.mamba_block(seq)
            if reverse:out=out.flip(1)
            out=out.transpose(1,2).reshape_as(zz)
            if axis=='column':out=out.transpose(2,3)
            ys.append(out)
        correction=self.restore(torch.stack(ys).mean(0))
        self.features=base+torch.sigmoid(self.gate(z))*correction
        return self.features

def build_model(method,depth,classes,k=256):
    if method.startswith("l4_ablation_"):
        from ablation_mixers import replace_mixer
        return replace_mixer(build_model("l4_resgate_spatial_k128", depth, classes, 128), method[len("l4_ablation_"):])

    if method in ("efficientvim_m1", "vision_mamba_tiny", "vmamba_tiny", "rsmamba_b"):
        from reviewer_models import build_reviewer_model
        return build_reviewer_model(method, classes)

    if method in ("vit_b16", "swin_t"):
        from torchvision.models import vit_b_16, swin_t
        return (vit_b_16 if method == "vit_b16" else swin_t)(weights=None, num_classes=classes)

    # Explicit layer4 pilot: preserve the paper's parallel left-side fusion,
    # then refine its compressed 7x7 tokens with a gated residual Mamba.
    if method == 'l4_resgate_spatial_k128':
        return build_model('serial_spatial_dtref_resgate_posenc_parallel_chunked',
                           depth,classes,128)
    if method == 'l4_without_attention':
        return build_model('backbone_only',depth,classes,k)
    if method.endswith('_parallel_chunked'):
        model=build_model(method[:-len('_parallel_chunked')],depth,classes,k)
        for module in model.modules():
            if hasattr(module,'_parallel_scan') and not hasattr(module,'scan_backend'):
                module.scan_backend='parallel_chunked'
        return model
    if method.endswith('_parallel_checkpoint'):
        model=build_model(method[:-len('_parallel_checkpoint')],depth,classes,k)
        for module in model.modules():
            if hasattr(module,'_parallel_scan') and not hasattr(module,'scan_backend'):
                module.scan_backend='parallel_checkpoint'
        return model
    if method.endswith('_parallel'):
        # Explicit experiment ID keeps archived sequential checkpoints intact.
        model=build_model(method[:-len('_parallel')],depth,classes,k)
        for module in model.modules():
            if hasattr(module,'_parallel_scan') and hasattr(module,'scan_backend') is False:
                module.scan_backend='parallel'
        return model
    factory={18:resnet18_mamba_attention,34:resnet34_mamba_attention,50:resnet50_mamba_attention}[depth]
    # 名字以 _fullconv 结尾 = 主干残差单元用全卷积（等同 ResNet-18/34/50 原始单元），
    # 其余设置与同名方法完全一致：用于比较“同一主干、只差模块”。
    depthwise=not method.endswith('_fullconv')
    base=method if depthwise else method[:-len('_fullconv')]
    # 名字以 l3_ 开头 = 把 Mamba 精修从 layer4（7x7=49 token，感受野已全局）前移到
    # layer3（14x14=196 token，感受野尚未覆盖全图），其余设置与同名方法一致。
    refine_stage='layer4'
    if base.startswith('l3bridge_'):
        refine_stage='layer3_then_layer4'
        base=base[len('l3bridge_'):]
    elif base.startswith('l3_'):
        refine_stage='layer3'
        base=base[3:]
    # 名字以 _headflat 结尾 = 分类头不做全局池化，直接展开 7x7x2C（位置感知头）。
    # 用于检验「空间排布是否被 GAP 抹平」这一假设。
    pool='gap'
    if base.endswith('_headflat'):
        pool='flat'
        base=base[:-len('_headflat')]
    # 后缀可组合（例如 ..._posenc_nonsel），因此循环剥离：
    # _posenc   = 展平后给 token 加可学习 2D 位置编码（前端补回空间排布）
    # _nonsel   = 把扫描里的 Δ/B/C 换成不依赖输入的静态参数（S4 式非选择性），
    #             其余结构完全一致：Reviewer 2 点名要的同参数机制对照
    pos_enc=False;nonselective=False
    while True:
        if base.endswith('_posenc') and not pos_enc:
            pos_enc=True;base=base[:-len('_posenc')];continue
        if base.endswith('_nonsel') and not nonselective:
            nonselective=True;base=base[:-len('_nonsel')];continue
        break
    stage_channels=(256 if refine_stage in ('layer3','layer3_then_layer4') else 512)*(4 if depth==50 else 1)
    if base in ('serial','serial_dtref','serial_spatial','serial_spatial_dt','serial_spatial_norm','serial_spatial_mlp','serial_spatial_identity',
                'serial_spatial_dtref','serial_spatial_dtref_a16','serial_spatial_dtref_resgate',
                'serial_spatial_dtref_resgate_pool10',
                'serial_spatial_dtref_resgate_pool7'):
        # 允许把同样的串联模块接到 layer3（14x14=196 token）：那里 token 的空间
        # 位置仍然携带信息，是「把空间网格当序列扫」真正能起作用的分辨率。
        # serial_spatial_dtref：把 Δ 的参数化改回参考实现（softplus 带对数均匀偏置），
        # 使 Δ·|A| 落在 1e-3~6 量级而不是 ~45，状态不再每步清零。
        # serial_spatial_dtref_a16：在其基础上把 |A| 量级压到 [1,16]，进一步延长记忆。
        # serial_dtref：与 serial 完全相同（不加二维编码），只修 Δ 的参数化。
        dt_init = base in ('serial_dtref', 'serial_spatial_dtref', 'serial_spatial_dtref_a16',
                           'serial_spatial_dtref_resgate','serial_spatial_dtref_resgate_pool10',
                           'serial_spatial_dtref_resgate_pool7')
        a_max = 16 if base == 'serial_spatial_dtref_a16' else None
        resgate = base in ('serial_spatial_dtref_resgate','serial_spatial_dtref_resgate_pool10',
                           'serial_spatial_dtref_resgate_pool7')
        scan_grid = (10 if base == 'serial_spatial_dtref_resgate_pool10' else
                     7 if base == 'serial_spatial_dtref_resgate_pool7' else None)
        if scan_grid is not None and refine_stage not in ('layer3','layer3_then_layer4'):
            raise ValueError('pooled scan is defined only for a layer3 attention grid')
        # A new experiment ID: never reinterpret existing ours/no_residual checkpoints.
        # Keep Mamba's internal multiplicative gate; remove the outer bypass and gate.
        m=factory(num_classes=classes,context_dim=k,context_residual=resgate,
                       context_gated=resgate,depthwise=depthwise,refine_stage=refine_stage,
                       context_dt_init=dt_init,context_a_max=a_max,pool=pool,
                       context_pos_enc=pos_enc,context_scan_grid=scan_grid)
        if resgate:
            # Begin close to the stable two-view baseline while allowing the
            # Mamba correction to grow only when training supports it.
            nn.init.constant_(m.mamba_attention.gate.bias, math.log(0.05 / 0.95))
        if base not in ('serial','serial_dtref'):
            m.mamba_attention.spatial_encoder=nn.Conv2d(k,k,3,padding=1,groups=k,bias=False)
        if base == 'serial_spatial_norm':
            m.mamba_attention.output_norm=RMSNorm(k)
        if base == 'serial_spatial_dt':
            # Only change the delta bias; retain identical shared weights/RNG.
            # Log-spaced target steps at zero input, learned independently thereafter.
            ssm=m.mamba_attention.mamba_block.S6
            dt=torch.logspace(-3,-1,ssm.d_model)
            ssm.fc1.bias=nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        if base == 'serial_spatial_mlp':
            target=sum(p.numel() for p in m.mamba_attention.mamba_block.parameters())
            m.mamba_attention.mamba_block=TokenMLP(k,target)
        if base == 'serial_spatial_identity':
            m.mamba_attention.mamba_block=nn.Identity()
        if nonselective:
            for module in m.modules():
                if type(module).__name__ == 'S6':
                    module.to_nonselective()
        return m
    if base=='resnet':
        m={18:models.resnet18,34:models.resnet34,50:models.resnet50}[depth](weights=None);m.fc=nn.Linear(m.fc.in_features,classes);return m
    if base=='mobilenet_v2':
        m=models.mobilenet_v2(weights=None);m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,classes);return m
    # 只有主干、完全没有模块：双路重标定与 Mamba 都不存在，分类头直接接主干输出。
    # 这是「模块整体是否有效」的唯一基准，之前一直没有。
    if base=='backbone_only':
        m=factory(num_classes=classes,depthwise=depthwise,refine_stage=refine_stage)
        m.mamba_attention=nn.Identity()
        m.fc=nn.Linear(stage_channels,classes)
        return m
    if base=='fullwidth':return factory(num_classes=classes,depthwise=depthwise,refine_stage=refine_stage)
    if base=='no_mamba':
        m=factory(num_classes=classes,depthwise=depthwise,refine_stage=refine_stage)
        m.mamba_attention=AttentionOnly(stage_channels)
        return m
    valid={'ours','fixed_gate','left_only','no_gate','no_residual','no_left','cbam','mlp','self_attention','bidirectional','fourway'}
    if base not in valid:raise ValueError(method)
    # 单因素消融：no_residual 只关闭残差路径，门控保持开启；no_gate 只关闭门控。
    # 之前 no_residual 会连带把门控也关掉，导致该项同时改了两个变量。
    m=factory(num_classes=classes,context_dim=k,context_residual=base!='no_residual',context_gated=base!='no_gate',depthwise=depthwise,refine_stage=refine_stage)
    c=stage_channels
    if base=='fixed_gate':
        for p in m.mamba_attention.gate.parameters():p.requires_grad_(False)
    if base=='left_only':m.mamba_attention=LeftOnly(m.mamba_attention.attention)
    if base=='no_left':m.mamba_attention.attention=TwoViews()
    if base=='cbam':m.mamba_attention.attention=StandardCBAMViews(c)
    if base in ('mlp','self_attention'):
        target=sum(p.numel() for p in m.mamba_attention.mamba_block.parameters())
        m.mamba_attention.mamba_block=(TokenMLP if base=='mlp' else AttentionMixer)(k,target)
    if base in ('bidirectional','fourway'):
        directions=[('row',False),('row',True)]
        if base=='fourway':directions += [('column',False),('column',True)]
        m.mamba_attention=ScanAttention(c,k,directions)
    return m
