"""
geometry_models.py

Comprehensive PyTorch Architecture Suite for Phase V9:
M1 — Direct Conditional ResUp CNN
M2 — Conditional Variational Autoencoder (CVAE)
M3 — GC-WPG-v2 (Multi-Scale Spatial Prior CNN)
M4 — GeoSDF-CVAE (Continuous Signed Distance Field Variational Architecture)
M5 — Geometry-Conditioned cGAN (Spatial Geometry Conditioning + PatchGAN)
Differentiable loss functions: SDF Loss, Sigmoid Mask Loss, Apex Loss, ROI Loss.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================================
# Shared Building Blocks
# =========================================================================
class ResUpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_up = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(self.act(self.bn_up(self.up(x))))

class MultiScaleSpatialFusion(nn.Module):
    def __init__(self, channels, prior_channels=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels + prior_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x, prior):
        return x + self.block(torch.cat([x, prior], dim=1))

# =========================================================================
# M1: Direct Conditional CNN
# =========================================================================
class DirectCNN(nn.Module):
    def __init__(self, cond_dim=6, noise_dim=16):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        in_dim = cond_dim + noise_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up1 = ResUpBlock(128, 64)
        self.up2 = ResUpBlock(64, 32)
        self.up3 = ResUpBlock(32, 16)
        self.up4 = ResUpBlock(16, 16)
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, c, z=None):
        if z is None:
            z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
        inp = torch.cat([c, z], dim=-1)
        h = self.fc(inp).reshape(-1, 128, 8, 8)
        feat = self.up4(self.up3(self.up2(self.up1(h))))
        return self.out_conv(feat)

# =========================================================================
# M2: Conditional VAE (Authentic Architecture)
# =========================================================================
class ConditionalVAE(nn.Module):
    def __init__(self, cond_dim=6, latent_dim=24):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc_mu = nn.Linear(128 * 8 * 8 + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8 + cond_dim, latent_dim)
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up1 = ResUpBlock(128, 64)
        self.up2 = ResUpBlock(64, 32)
        self.up3 = ResUpBlock(32, 16)
        self.up4 = ResUpBlock(16, 16)
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def encode(self, x, c):
        h = self.enc_conv(x).reshape(x.size(0), -1)
        hc = torch.cat([h, c], dim=-1)
        return self.fc_mu(hc), self.fc_logvar(hc)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z, c):
        zc = torch.cat([z, c], dim=-1)
        h = self.dec_fc(zc).reshape(-1, 128, 8, 8)
        feat = self.up4(self.up3(self.up2(self.up1(h))))
        return self.out_conv(feat)
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    def generate(self, c, z=None):
        self.eval()
        with torch.no_grad():
            if z is None:
                z = torch.zeros((c.size(0), self.latent_dim), dtype=torch.float32, device=c.device)
            return self.decode(z, c)

# =========================================================================
# M3: GC-WPG-v2 (Multi-Scale Spatial Prior)
# =========================================================================
class GC_WPG_v2(nn.Module):
    def __init__(self, cond_dim=6, noise_dim=16, scaler_params=None):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.scaler = scaler_params if scaler_params is not None else {
            'w_min': 6.82, 'w_max': 12.02, 'r_min': 1.20, 'r_max': 4.85
        }
        self.geom_head = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        in_dim = cond_dim + noise_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up1 = ResUpBlock(128, 64)
        self.fuse1 = MultiScaleSpatialFusion(64, prior_channels=2)
        self.up2 = ResUpBlock(64, 32)
        self.fuse2 = MultiScaleSpatialFusion(32, prior_channels=2)
        self.up3 = ResUpBlock(32, 16)
        self.fuse3 = MultiScaleSpatialFusion(16, prior_channels=2)
        self.up4 = ResUpBlock(16, 16)
        self.fuse4 = MultiScaleSpatialFusion(16, prior_channels=2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        y_grid, x_grid = torch.meshgrid(torch.arange(128, dtype=torch.float32),
                                        torch.arange(128, dtype=torch.float32), indexing='ij')
        self.register_buffer("y_coords", y_grid.contiguous().clone(), persistent=False)
        self.register_buffer("x_coords", x_grid.contiguous().clone(), persistent=False)

    def generate_spatial_geometry_prior(self, w_norm, reinf_norm):
        batch_size = w_norm.size(0)
        px_scale = 6.0
        w_px = (w_norm * (self.scaler['w_max'] - self.scaler['w_min']) + self.scaler['w_min']) * px_scale
        r_px = (reinf_norm * (self.scaler['r_max'] - self.scaler['r_min']) + self.scaler['r_min']) * px_scale
        w_px = w_px.reshape(batch_size, 1, 1)
        r_px = r_px.reshape(batch_size, 1, 1)
        y_dist = (self.y_coords.unsqueeze(0) - 64.0) / (w_px / 2.0 + 1e-6)
        profile_factor = torch.clamp(1.0 - y_dist**2, min=0.0)
        x_crown_bnd = 64.0 - r_px * profile_factor
        crown_prior = torch.sigmoid(1.5 * (self.x_coords.unsqueeze(0) - x_crown_bnd)) * torch.sigmoid(1.5 * (64.0 - self.x_coords.unsqueeze(0)))
        plate_prior = torch.sigmoid(1.5 * (self.x_coords.unsqueeze(0) - 64.0)).expand(batch_size, -1, -1)
        return torch.stack([crown_prior, plate_prior], dim=1)

    def forward(self, c, z=None):
        if z is None:
            z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
        pred_geom = self.geom_head(c)
        prior_128 = self.generate_spatial_geometry_prior(pred_geom[:, 0:1], pred_geom[:, 1:2])
        prior_64 = F.interpolate(prior_128, size=(64, 64), mode='bilinear', align_corners=False)
        prior_32 = F.interpolate(prior_128, size=(32, 32), mode='bilinear', align_corners=False)
        prior_16 = F.interpolate(prior_128, size=(16, 16), mode='bilinear', align_corners=False)
        inp = torch.cat([c, z], dim=-1)
        h = self.fc(inp).reshape(-1, 128, 8, 8)
        h1 = self.fuse1(self.up1(h), prior_16)
        h2 = self.fuse2(self.up2(h1), prior_32)
        h3 = self.fuse3(self.up3(h2), prior_64)
        h4 = self.fuse4(self.up4(h3), prior_128)
        return self.out_conv(h4), pred_geom, prior_128

# =========================================================================
# M4: GeoSDF-CVAE (Signed Distance Field Continuous Geometry VAE)
# =========================================================================
class GeoSDF_CVAE(nn.Module):
    """
    Predicts continuous Signed Distance Field (SDF) of the weld crown geometry
    under a variational latent formulation, converts SDF to a differentiable mask,
    and synthesizes an RGB image conditioned on the continuous geometry field.
    """
    def __init__(self, cond_dim=6, latent_dim=16, beta=1.0):
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.beta = beta  # Transition steepness for smooth mask

        # Variational Geometry Encoder: Encodes (SDF, c) -> mu, logvar
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc_mu = nn.Linear(128 * 8 * 8 + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8 + cond_dim, latent_dim)

        # Condition prior net: predicts prior mu and logvar from parameters c alone
        self.prior_net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, latent_dim * 2)
        )

        # Geometry SDF Decoder: Maps (z, c) -> Continuous SDF Field (128x128)
        self.sdf_fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.sdf_up1 = ResUpBlock(128, 64)
        self.sdf_up2 = ResUpBlock(64, 32)
        self.sdf_up3 = ResUpBlock(32, 16)
        self.sdf_up4 = ResUpBlock(16, 16)
        self.sdf_head = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        # Apex and Projection Head directly from latent & conditioning
        self.apex_head = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 2) # predicts apex_x, apex_y in pixels
        )

        # RGB Image Generator: Conditioned on (c, z, predicted_SDF, predicted_mask)
        self.rgb_fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64 * 8 * 8),
            nn.BatchNorm1d(64 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.rgb_up1 = ResUpBlock(64, 64)
        self.rgb_up2 = ResUpBlock(64, 32)
        self.rgb_up3 = ResUpBlock(32, 16)
        self.rgb_up4 = ResUpBlock(16, 16)
        
        # Spatial geometry modulation at full resolution
        self.rgb_head = nn.Sequential(
            nn.Conv2d(16 + 2, 32, kernel_size=3, padding=1), # 16 feats + 1 SDF + 1 Mask
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, sdf_gt, c):
        h = self.enc_conv(sdf_gt).reshape(sdf_gt.size(0), -1)
        hc = torch.cat([h, c], dim=-1)
        return self.fc_mu(hc), self.fc_logvar(hc)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_sdf(self, z, c):
        zc = torch.cat([z, c], dim=-1)
        h = self.sdf_fc(zc).reshape(-1, 128, 8, 8)
        feat = self.sdf_up4(self.sdf_up3(self.sdf_up2(self.sdf_up1(h))))
        pred_sdf = self.sdf_head(feat)
        # Convert SDF to differentiable smooth binary mask
        pred_mask = torch.sigmoid(-self.beta * pred_sdf)
        pred_apex = self.apex_head(zc)
        return pred_sdf, pred_mask, pred_apex

    def decode_rgb(self, z, c, pred_sdf, pred_mask):
        zc = torch.cat([z, c], dim=-1)
        h = self.rgb_fc(zc).reshape(-1, 64, 8, 8)
        feat = self.rgb_up4(self.rgb_up3(self.rgb_up2(self.rgb_up1(h))))
        geom_cat = torch.cat([feat, pred_sdf, pred_mask], dim=1)
        pred_rgb = self.rgb_head(geom_cat)
        return pred_rgb

    def forward(self, sdf_gt, c):
        mu, logvar = self.encode(sdf_gt, c)
        z = self.reparameterize(mu, logvar)
        pred_sdf, pred_mask, pred_apex = self.decode_sdf(z, c)
        pred_rgb = self.decode_rgb(z, c, pred_sdf, pred_mask)
        return pred_rgb, pred_sdf, pred_mask, pred_apex, mu, logvar

    def generate(self, c, z=None):
        self.eval()
        with torch.no_grad():
            if z is None:
                # Canonical deterministic evaluation: z = 0
                z = torch.zeros((c.size(0), self.latent_dim), dtype=torch.float32, device=c.device)
            pred_sdf, pred_mask, pred_apex = self.decode_sdf(z, c)
            pred_rgb = self.decode_rgb(z, c, pred_sdf, pred_mask)
            return pred_rgb, pred_sdf, pred_mask, pred_apex

# =========================================================================
# M5: Geometry-Conditioned cGAN (Spatial Conditioning + PatchGAN)
# =========================================================================
class Geometry_cGAN_Generator(nn.Module):
    """
    Two-stage generator:
    Stage 1: Predicts continuous crown mask M_pred in [0, 1] from parameters c.
    Stage 2: Translates (M_pred, c) into high-fidelity RGB macrograph.
    """
    def __init__(self, cond_dim=6):
        super().__init__()
        self.cond_dim = cond_dim

        # Geometry Mask Head
        self.geom_fc = nn.Sequential(
            nn.Linear(cond_dim, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.geom_up1 = ResUpBlock(128, 64)
        self.geom_up2 = ResUpBlock(64, 32)
        self.geom_up3 = ResUpBlock(32, 16)
        self.geom_up4 = ResUpBlock(16, 16)
        self.mask_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Image Synthesis Decoder: Conditioned on spatial mask + parameter vector
        self.img_fc = nn.Sequential(
            nn.Linear(cond_dim, 64 * 8 * 8),
            nn.BatchNorm1d(64 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_up1 = ResUpBlock(64, 64)
        self.img_up2 = ResUpBlock(64, 32)
        self.img_up3 = ResUpBlock(32, 16)
        self.img_up4 = ResUpBlock(16, 16)
        
        # Merge spatial mask at 128x128
        self.final_conv = nn.Sequential(
            nn.Conv2d(16 + 1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, c):
        # Stage 1: Geometry mask prediction
        h_g = self.geom_fc(c).reshape(-1, 128, 8, 8)
        feat_g = self.geom_up4(self.geom_up3(self.geom_up2(self.geom_up1(h_g))))
        pred_mask = self.mask_head(feat_g)

        # Stage 2: Photorealistic translation
        h_i = self.img_fc(c).reshape(-1, 64, 8, 8)
        feat_i = self.img_up4(self.img_up3(self.img_up2(self.img_up1(h_i))))
        merged = torch.cat([feat_i, pred_mask], dim=1)
        pred_rgb = self.final_conv(merged)

        return pred_rgb, pred_mask

class PatchGAN_Discriminator(nn.Module):
    """
    Compact PatchGAN discriminator evaluating pairs: (Mask, RGB Image) -> [0, 1]
    Small capacity + spectral normalization to prevent memorization on small N=36 datasets.
    """
    def __init__(self, in_channels=4): # 1 channel mask + 3 channel RGB
        super().__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)), # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)), # 64 -> 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)), # 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1) # 16 -> 15 (PatchGAN logits)
        )
    def forward(self, mask, img):
        x = torch.cat([mask, img], dim=1)
        return self.model(x)
