import math
import torch
from torch import nn

# -------------------------------
# SHARED COMPONENTS
# -------------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            Swish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

    def forward(self, time):
        device = time.device
        half_dim = self.time_emb_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.mlp(embeddings)

class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim),
            Swish(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
    def forward(self, cond):
        return self.mlp(cond)

# -------------------------------
# V3-A: NEAREST NEIGHBOR (BASELINE)
# -------------------------------
class BlockA(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.cond_mlp = nn.Linear(time_emb_dim, out_ch)
        
        if up:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, 3, padding=1)
            )
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            
        self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1) if not up else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gnorm1 = nn.GroupNorm(8, out_ch)
        self.gnorm2 = nn.GroupNorm(8, out_ch)
        self.act = Swish()

    def forward(self, x, t_emb, c_emb):
        x = self.conv1(x)
        x = self.act(x)
        x = self.gnorm1(x)
        
        # Additive conditioning
        t_emb_out = self.time_mlp(t_emb)[:, :, None, None]
        c_emb_out = self.cond_mlp(c_emb)[:, :, None, None]
        x = x + t_emb_out + c_emb_out
        
        x = self.conv2(x)
        x = self.act(x)
        x = self.gnorm2(x)
        return self.transform(x)

class ConditionalUNetV3A(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, channel_mults=(1, 2, 4, 8), cond_dim=8, time_emb_dim=32):
        super().__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.cond_emb = ConditionEmbedding(cond_dim, time_emb_dim)
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        channels = [base_channels]
        now_ch = base_channels
        
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(BlockA(now_ch, out_ch, time_emb_dim))
            now_ch = out_ch
            channels.append(now_ch)
            
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.ups.append(BlockA(now_ch + channels.pop(), out_ch, time_emb_dim, up=True))
            now_ch = out_ch
            
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            Swish(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

    def forward(self, x, time, cond):
        t_emb = self.time_emb(time)
        c_emb = self.cond_emb(cond)
        
        x = self.init_conv(x)
        skips = [x]
        
        for down in self.downs:
            x = down(x, t_emb, c_emb)
            skips.append(x)
            
        for up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = up(x, t_emb, c_emb)
            
        return self.out_conv(x)

# -------------------------------
# V3-B: BILINEAR UPSAMPLING
# -------------------------------
class BlockB(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.cond_mlp = nn.Linear(time_emb_dim, out_ch)
        
        if up:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1)
            )
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            
        self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1) if not up else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gnorm1 = nn.GroupNorm(8, out_ch)
        self.gnorm2 = nn.GroupNorm(8, out_ch)
        self.act = Swish()

    def forward(self, x, t_emb, c_emb):
        x = self.conv1(x)
        x = self.act(x)
        x = self.gnorm1(x)
        
        # Additive conditioning
        t_emb_out = self.time_mlp(t_emb)[:, :, None, None]
        c_emb_out = self.cond_mlp(c_emb)[:, :, None, None]
        x = x + t_emb_out + c_emb_out
        
        x = self.conv2(x)
        x = self.act(x)
        x = self.gnorm2(x)
        return self.transform(x)

class ConditionalUNetV3B(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, channel_mults=(1, 2, 4, 8), cond_dim=8, time_emb_dim=32):
        super().__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.cond_emb = ConditionEmbedding(cond_dim, time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = [base_channels]
        now_ch = base_channels
        
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(BlockB(now_ch, out_ch, time_emb_dim))
            now_ch = out_ch
            channels.append(now_ch)
            
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.ups.append(BlockB(now_ch + channels.pop(), out_ch, time_emb_dim, up=True))
            now_ch = out_ch
            
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            Swish(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

    def forward(self, x, time, cond):
        t_emb = self.time_emb(time)
        c_emb = self.cond_emb(cond)
        x = self.init_conv(x)
        skips = [x]
        for down in self.downs:
            x = down(x, t_emb, c_emb)
            skips.append(x)
        for up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = up(x, t_emb, c_emb)
        return self.out_conv(x)

# -------------------------------
# V3-C: BILINEAR + FiLM CONDITIONING
# -------------------------------
class BlockC(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        # FiLM condition projection: outputs (scale, shift)
        self.cond_film_mlp = nn.Linear(time_emb_dim, out_ch * 2)
        
        if up:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1)
            )
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            
        self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1) if not up else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gnorm1 = nn.GroupNorm(8, out_ch)
        self.gnorm2 = nn.GroupNorm(8, out_ch)
        self.act = Swish()

    def forward(self, x, t_emb, c_emb):
        x = self.conv1(x)
        x = self.act(x)
        x = self.gnorm1(x)
        
        # Time injection (Additive)
        t_emb_out = self.time_mlp(t_emb)[:, :, None, None]
        x = x + t_emb_out
        
        # FiLM Condition Injection (Multiplicative Scale + Shift)
        film_params = self.cond_film_mlp(c_emb)[:, :, None, None]
        scale, shift = film_params.chunk(2, dim=1)
        x = x * (1 + scale) + shift
        
        x = self.conv2(x)
        x = self.act(x)
        x = self.gnorm2(x)
        return self.transform(x)

class ConditionalUNetV3C(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, channel_mults=(1, 2, 4, 8), cond_dim=8, time_emb_dim=32):
        super().__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.cond_emb = ConditionEmbedding(cond_dim, time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = [base_channels]
        now_ch = base_channels
        
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(BlockC(now_ch, out_ch, time_emb_dim))
            now_ch = out_ch
            channels.append(now_ch)
            
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.ups.append(BlockC(now_ch + channels.pop(), out_ch, time_emb_dim, up=True))
            now_ch = out_ch
            
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            Swish(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

    def forward(self, x, time, cond):
        t_emb = self.time_emb(time)
        c_emb = self.cond_emb(cond)
        x = self.init_conv(x)
        skips = [x]
        for down in self.downs:
            x = down(x, t_emb, c_emb)
            skips.append(x)
        for up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = up(x, t_emb, c_emb)
        return self.out_conv(x)
