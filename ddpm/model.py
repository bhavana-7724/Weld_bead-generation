import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        if up:
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gnorm1 = nn.GroupNorm(8, out_ch)
        self.gnorm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, t):
        h = self.gnorm1(self.act(self.conv1(x)))
        time_emb = self.act(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.gnorm2(self.act(self.conv2(h)))
        return self.transform(h)

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, channel_mults=(1, 2, 4), cond_dim=8, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Track skip channels
        skip_channels = [base_channels]
        now_ch = base_channels
        
        # Down
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(Block(now_ch, out_ch, time_emb_dim))
            now_ch = out_ch
            skip_channels.append(now_ch)
            
        # Bottleneck
        self.mid_conv1 = nn.Conv2d(now_ch, now_ch, 3, padding=1)
        self.mid_time_mlp = nn.Linear(time_emb_dim, now_ch)
        self.mid_conv2 = nn.Conv2d(now_ch, now_ch, 3, padding=1)
        self.mid_act = nn.SiLU()
        self.mid_gnorm1 = nn.GroupNorm(8, now_ch)
        self.mid_gnorm2 = nn.GroupNorm(8, now_ch)
        
        # Up
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            skip_ch = skip_channels.pop()
            # x has now_ch, s has skip_ch
            self.ups.append(Block(now_ch + skip_ch, out_ch, time_emb_dim, up=True))
            now_ch = out_ch
            
        self.out_conv = nn.Sequential(
            nn.Conv2d(now_ch + skip_channels.pop(), base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 1)
        )

    def forward(self, x, t, c):
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(c)
        emb = t_emb + c_emb  # Fuse timestep and conditioning
        
        x = self.init_conv(x)
        skips = [x]
        
        for down in self.downs:
            x = down(x, emb)
            skips.append(x)
            
        x = self.mid_gnorm1(self.mid_act(self.mid_conv1(x)))
        x = x + self.mid_act(self.mid_time_mlp(emb))[(...,) + (None,) * 2]
        x = self.mid_gnorm2(self.mid_act(self.mid_conv2(x)))
        
        for up in self.ups:
            s = skips.pop()
            # Ensure dimensions match before cat, assuming power of 2 sized images
            x = torch.cat((x, s), dim=1)
            x = up(x, emb)
            
        # Final block uses the very first skip connection (init_conv output)
        x = torch.cat((x, skips.pop()), dim=1)
        return self.out_conv(x)
