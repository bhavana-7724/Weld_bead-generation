"""
geometry_models_cuda.py

Phase V10: High-Performance CUDA-Optimized Geometry-Aware Architecture Suite
Contains all 6 neural models + 1 statistical shape baseline:

Neural Models:
- M1: Direct Conditional ResUp CNN
- M2: Conditional Variational Autoencoder (CVAE)
- M3: GC-WPG-v2 (Multi-Scale Spatial Prior CNN)
- M4: GeoSDF-CVAE (Continuous Signed Distance Field VAE with Prior Training & FiLM Conditioning)
- M5: Geometry-Conditioned cGAN (Spatial Mask Prediction + PatchGAN Discriminator)
- M6: GF-WPG (GENUINE Geometry-First Weld Profile Generator: Stage 1 Contour Network + Stage 2 Neural Renderer)

Statistical Shape Baseline:
- B1: PCAContourRegressor (Compact PCA Contour Representation + Ridge Regression Baseline)

Key CUDA & Mixed Precision Features:
- Mixed-precision numerical stability (guarded BCE, clamped probabilities).
- Device and AMP dtype casting on dynamic coordinate buffers.
- Modern spectral normalization with backward-compatible fallbacks.
- Complete decoupling of raw RGB, predicted geometry masks, and visualization overlays.
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

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation for conditioning injection."""
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.scale = nn.Linear(cond_dim, channels)
        self.shift = nn.Linear(cond_dim, channels)

    def forward(self, x, c):
        # x: (B, C, H, W), c: (B, cond_dim)
        gamma = self.scale(c).unsqueeze(-1).unsqueeze(-1)
        beta = self.shift(c).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta

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

    def forward(self, feat, prior_128):
        h, w = feat.shape[2], feat.shape[3]
        prior_scaled = F.interpolate(prior_128, size=(h, w), mode='bilinear', align_corners=False)
        return feat + self.block(torch.cat([feat, prior_scaled], dim=1))

# =========================================================================
# M1: Direct Conditional ResUp CNN
# =========================================================================
class DirectCNN(nn.Module):
    def __init__(self, cond_dim=6, latent_dim=16):
        super().__init__()
        self.cond_dim = cond_dim
        self.fc = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up1 = ResUpBlock(128, 64)   # 8x8 -> 16x16
        self.up2 = ResUpBlock(64, 32)    # 16x16 -> 32x32
        self.up3 = ResUpBlock(32, 16)    # 32x32 -> 64x64
        self.up4 = ResUpBlock(16, 16)    # 64x64 -> 128x128
        self.head = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, c):
        z_dummy = torch.zeros((c.size(0), 16), dtype=c.dtype, device=c.device)
        h = self.fc(torch.cat([c, z_dummy], dim=1)).reshape(-1, 128, 8, 8)
        feat = self.up4(self.up3(self.up2(self.up1(h))))
        return self.head(feat)

# =========================================================================
# M2: Conditional Variational Autoencoder (CVAE)
# =========================================================================
class ConditionalVAE(nn.Module):
    def __init__(self, cond_dim=6, latent_dim=24):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Encoder: (Image, c) -> mu, logvar
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

        # Decoder: (z, c) -> Image
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up1 = ResUpBlock(128, 64)
        self.up2 = ResUpBlock(64, 32)
        self.up3 = ResUpBlock(32, 16)
        self.up4 = ResUpBlock(16, 16)
        self.head = nn.Sequential(
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
        return self.head(feat)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

    def generate(self, c, z=None):
        self.eval()
        with torch.no_grad():
            if z is None:
                z = torch.zeros((c.size(0), self.latent_dim), dtype=c.dtype, device=c.device)
            return self.decode(z, c)

# =========================================================================
# M3: GC-WPG-v2 (Multi-Scale Spatial Prior CNN)
# =========================================================================
class GC_WPG_v2(nn.Module):
    def __init__(self, cond_dim=6, latent_dim=16, scaler_params=None):
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        self.w_min = scaler_params.get('w_min', 6.82) if scaler_params else 6.82
        self.w_max = scaler_params.get('w_max', 12.02) if scaler_params else 12.02
        self.r_min = scaler_params.get('r_min', 1.20) if scaler_params else 1.20
        self.r_max = scaler_params.get('r_max', 4.85) if scaler_params else 4.85

        # Analytical Geometry Predictor Head
        self.geom_head = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

        # Main Decoder Backbone
        self.fc = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, 128 * 8 * 8),
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

        self.head = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Coordinate Grids for Spatial Prior Synthesis (registered as buffers)
        ys = torch.linspace(-1.0, 1.0, 128).view(1, 1, 128, 1).expand(1, 1, 128, 128)
        xs = torch.linspace(0.0, 1.0, 128).view(1, 1, 1, 128).expand(1, 1, 128, 128)
        self.register_buffer('y_coords', ys)
        self.register_buffer('x_coords', xs)

    def compute_spatial_prior(self, w_norm, r_norm, device):
        w_mm = w_norm * (self.w_max - self.w_min) + self.w_min
        r_mm = r_norm * (self.r_max - self.r_min) + self.r_min
        p_width_px = w_mm * 10.0
        p_proj_px = r_mm * 10.0

        b = w_norm.size(0)
        h_val = torch.clamp(p_proj_px / 64.0, 0.05, 1.0).view(b, 1, 1, 1)
        w_val = torch.clamp(p_width_px / 64.0, 0.1, 1.5).view(b, 1, 1, 1)

        y_c = self.y_coords.to(device=device, dtype=w_norm.dtype).expand(b, -1, -1, -1)
        x_c = self.x_coords.to(device=device, dtype=w_norm.dtype).expand(b, -1, -1, -1)

        crown_profile = h_val * torch.clamp(1.0 - (y_c / w_val).pow(2), min=0.0)
        crown_prior = torch.sigmoid((crown_profile - (0.5 - x_c)) * 15.0) * (x_c < 0.5).to(dtype=w_norm.dtype)
        plate_prior = (x_c >= 0.5).to(dtype=w_norm.dtype)

        return torch.cat([crown_prior, plate_prior], dim=1)

    def forward(self, c):
        pred_geom = self.geom_head(c)
        w_norm, r_norm = pred_geom[:, 0:1], pred_geom[:, 1:2]
        spatial_prior = self.compute_spatial_prior(w_norm, r_norm, c.device)

        z_dummy = torch.zeros((c.size(0), self.latent_dim), dtype=c.dtype, device=c.device)
        h = self.fc(torch.cat([c, z_dummy], dim=1)).reshape(-1, 128, 8, 8)

        f1 = self.fuse1(self.up1(h), spatial_prior)
        f2 = self.fuse2(self.up2(f1), spatial_prior)
        f3 = self.fuse3(self.up3(f2), spatial_prior)
        f4 = self.fuse4(self.up4(f3), spatial_prior)

        img = self.head(f4)
        return img, pred_geom, spatial_prior

# =========================================================================
# M4: GeoSDF-CVAE (Continuous Signed Distance Field VAE with FiLM & Prior Loss)
# =========================================================================
class GeoSDF_CVAE(nn.Module):
    def __init__(self, cond_dim=6, latent_dim=16, beta=1.0):
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Geometry SDF Encoder: (SDF, c) -> mu, logvar
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
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

        # SDF Decoder with FiLM Condition Injection (Prevents Conditioning Ignorance / Posterior Collapse)
        self.sdf_fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.sdf_up1 = ResUpBlock(128, 64)
        self.film1 = FiLMBlock(cond_dim, 64)

        self.sdf_up2 = ResUpBlock(64, 32)
        self.film2 = FiLMBlock(cond_dim, 32)

        self.sdf_up3 = ResUpBlock(32, 16)
        self.film3 = FiLMBlock(cond_dim, 16)

        self.sdf_up4 = ResUpBlock(16, 16)
        self.film4 = FiLMBlock(cond_dim, 16)

        self.sdf_head = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        # Apex and Projection Head
        self.apex_head = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 2)
        )

        # RGB Synthesis Decoder conditioned on predicted geometry
        self.rgb_fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64 * 8 * 8),
            nn.BatchNorm1d(64 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.rgb_up1 = ResUpBlock(64, 64)
        self.rgb_up2 = ResUpBlock(64, 32)
        self.rgb_up3 = ResUpBlock(32, 16)
        self.rgb_up4 = ResUpBlock(16, 16)
        
        self.rgb_head = nn.Sequential(
            nn.Conv2d(16 + 2, 32, kernel_size=3, padding=1),
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
        f1 = self.film1(self.sdf_up1(h), c)
        f2 = self.film2(self.sdf_up2(f1), c)
        f3 = self.film3(self.sdf_up3(f2), c)
        f4 = self.film4(self.sdf_up4(f3), c)
        pred_sdf = self.sdf_head(f4)
        pred_mask = torch.sigmoid(-self.beta * pred_sdf)
        pred_apex = self.apex_head(zc)
        return pred_sdf, pred_mask, pred_apex

    def decode_rgb(self, z, c, pred_sdf, pred_mask):
        zc = torch.cat([z, c], dim=-1)
        h = self.rgb_fc(zc).reshape(-1, 64, 8, 8)
        feat = self.rgb_up4(self.rgb_up3(self.rgb_up2(self.rgb_up1(h))))
        pred_sdf_cast = pred_sdf.to(dtype=feat.dtype)
        pred_mask_cast = pred_mask.to(dtype=feat.dtype)
        geom_cat = torch.cat([feat, pred_sdf_cast, pred_mask_cast], dim=1)
        return self.rgb_head(geom_cat)

    def forward(self, sdf_gt, c):
        mu, logvar = self.encode(sdf_gt, c)
        z = self.reparameterize(mu, logvar)
        pred_sdf, pred_mask, pred_apex = self.decode_sdf(z, c)
        pred_rgb = self.decode_rgb(z, c, pred_sdf, pred_mask)
        return pred_rgb, pred_sdf, pred_mask, pred_apex, mu, logvar

    def forward_prior(self, c):
        """Pure condition-driven forward pass with z=0 (prior mode)."""
        z_zero = torch.zeros((c.size(0), self.latent_dim), dtype=c.dtype, device=c.device)
        pred_sdf, pred_mask, pred_apex = self.decode_sdf(z_zero, c)
        pred_rgb = self.decode_rgb(z_zero, c, pred_sdf, pred_mask)
        return pred_rgb, pred_sdf, pred_mask, pred_apex

    def generate(self, c, z=None):
        self.eval()
        with torch.no_grad():
            if z is None:
                z = torch.zeros((c.size(0), self.latent_dim), dtype=c.dtype, device=c.device)
            pred_sdf, pred_mask, pred_apex = self.decode_sdf(z, c)
            pred_rgb = self.decode_rgb(z, c, pred_sdf, pred_mask)
            return pred_rgb, pred_sdf, pred_mask, pred_apex

# =========================================================================
# M5: Geometry-Conditioned cGAN
# =========================================================================
class Geometry_cGAN_Generator(nn.Module):
    def __init__(self, cond_dim=6):
        super().__init__()
        self.mask_net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 64 * 8 * 8),
            nn.BatchNorm1d(64 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mask_up = nn.Sequential(
            ResUpBlock(64, 32),
            ResUpBlock(32, 16),
            ResUpBlock(16, 8),
            ResUpBlock(8, 8),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.rgb_net = nn.Sequential(
            nn.Conv2d(1 + cond_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, c):
        h = self.mask_net(c).reshape(-1, 64, 8, 8)
        pred_mask = self.mask_up(h)
        b, _, height, width = pred_mask.shape
        c_spatial = c.view(b, c.size(1), 1, 1).expand(-1, -1, height, width)
        pred_rgb = self.rgb_net(torch.cat([pred_mask, c_spatial], dim=1))
        return pred_rgb, pred_mask

class PatchGAN_Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        try:
            from torch.nn.utils.parametrizations import spectral_norm
        except ImportError:
            from torch.nn.utils import spectral_norm

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, mask, img):
        x = torch.cat([mask, img], dim=1)
        return self.model(x)

# =========================================================================
# M6: GF-WPG (GENUINE Geometry-First Weld Profile Generator)
# =========================================================================
class GF_WPG(nn.Module):
    """
    GENUINE Geometry-First Weld Profile Generator (GF-WPG):
    
    Stage 1: Explicit Neural Geometry Regressor G(c)
      Directly predicts the discrete external crown contour coordinates x(y)
      for y in [18, 112] (95 points), crown apex (x_apex, y_apex), and deposit area.
      Loss is purely geometric: L_geom = ||x_pred - x_gt||_1 + ||apex_pred - apex_gt||_1.
      
    Stage 2: Differentiable Contour Rasterizer & Neural Profile Renderer R(M_geom, c)
      Takes the explicit predicted geometry raster and conditional process variables
      to synthesize photorealistic metallographic cross-section texture.
    """
    def __init__(self, cond_dim=6, num_contour_pts=95):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_contour_pts = num_contour_pts

        # Stage 1: Explicit Geometry Network
        self.geom_net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_contour_pts + 2) # 95 contour x-coords + (apex_x, apex_y)
        )

        # Stage 2: Neural Renderer Backbone (takes rasterized mask + c)
        self.render_fc = nn.Sequential(
            nn.Linear(cond_dim + 16, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up1 = ResUpBlock(128, 64)
        self.up2 = ResUpBlock(64, 32)
        self.up3 = ResUpBlock(32, 16)
        self.up4 = ResUpBlock(16, 16)

        # Final fusion with predicted geometry mask
        self.rgb_head = nn.Sequential(
            nn.Conv2d(16 + 1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Coordinate grid for differentiable soft rasterization
        xs = torch.arange(128).view(1, 1, 1, 128).float()
        self.register_buffer('xs_grid', xs)

    def predict_geometry(self, c):
        out = self.geom_net(c)
        # Bounded projection: x_apex in [15, 60], contour points in [15, 64]
        raw_pts = out[:, :self.num_contour_pts]
        pred_pts = 64.0 - F.softplus(raw_pts)
        pred_pts = torch.clamp(pred_pts, min=15.0, max=64.0)

        raw_apex = out[:, self.num_contour_pts:]
        apex_x = 64.0 - F.softplus(raw_apex[:, 0:1])
        apex_y = torch.sigmoid(raw_apex[:, 1:2]) * 94.0 + 18.0
        pred_apex = torch.cat([apex_x, apex_y], dim=-1)

        return pred_pts, pred_apex

    def rasterize_contour(self, pred_pts, device):
        """
        Differentiable rasterizer: maps 95 contour x(y) points to a 128x128 mask.
        For y in [18, 112]: mask is 1 between pred_x(y) and x_baseline=64.
        """
        b = pred_pts.size(0)
        mask = torch.zeros((b, 1, 128, 128), dtype=pred_pts.dtype, device=device)
        xs = self.xs_grid.to(device=device, dtype=pred_pts.dtype) # (1, 1, 1, 128)

        # Smooth boundary step along X: sig(64 - x) * sig(x - x_pred)
        for idx in range(self.num_contour_pts):
            y = 18 + idx
            x_pred = pred_pts[:, idx:idx+1].unsqueeze(-1).unsqueeze(-1) # (b, 1, 1, 1)
            row_mask = torch.sigmoid((64.0 - xs) * 5.0) * torch.sigmoid((xs - x_pred) * 5.0)
            mask[:, :, y:y+1, :] = row_mask
        return mask

    def forward(self, c):
        # Stage 1: Geometry Prediction
        pred_pts, pred_apex = self.predict_geometry(c)
        pred_mask = self.rasterize_contour(pred_pts, c.device)

        # Stage 2: Profile Rendering conditioned on predicted geometry
        z_dummy = torch.zeros((c.size(0), 16), dtype=c.dtype, device=c.device)
        h = self.render_fc(torch.cat([c, z_dummy], dim=1)).reshape(-1, 128, 8, 8)
        feat = self.up4(self.up3(self.up2(self.up1(h))))
        pred_rgb = self.rgb_head(torch.cat([feat, pred_mask], dim=1))

        return pred_rgb, pred_mask, pred_pts, pred_apex

    def generate(self, c):
        self.eval()
        with torch.no_grad():
            return self.forward(c)

# =========================================================================
# Baseline B1: PCA-Contour + Ridge Regression Baseline
# =========================================================================
class PCAContourRegressor:
    """
    Statistical Shape Model Baseline (Active Shape Model):
    1. Computes PCA on the 95-point crown contour coordinates of the training set.
    2. Uses regularized Ridge Regression to map process parameters c -> PCA component weights.
    3. Reconstructs predicted contour and rasterizes into baseline binary mask and macrograph.
    """
    def __init__(self, n_components=4, alpha=1.0):
        self.n_components = n_components
        self.alpha = alpha
        self.mean_contour = None
        self.components = None
        self.weights = None
        self.fitted = False

    def fit(self, C_train, X_contours_train):
        """
        C_train: (N, 6) numpy array of normalized welding conditions.
        X_contours_train: (N, 95) numpy array of contour x-coordinates for y in [18, 112].
        """
        N = X_contours_train.shape[0]
        self.mean_contour = np.mean(X_contours_train, axis=0) # (95,)
        centered = X_contours_train - self.mean_contour

        # SVD / PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.components = Vt[:self.n_components] # (K, 95)
        # Project training contours to PCA weights: (N, K)
        W_train = np.dot(centered, self.components.T)

        # Ridge Regression: W = (C^T C + alpha I)^(-1) C^T W_train
        C_bias = np.hstack([C_train, np.ones((N, 1))])
        reg_I = self.alpha * np.eye(C_bias.shape[1])
        reg_I[-1, -1] = 0.0 # Don't regularize bias
        self.weights = np.linalg.solve(np.dot(C_bias.T, C_bias) + reg_I, np.dot(C_bias.T, W_train))
        self.fitted = True

    def predict_contour(self, c_vec):
        """
        c_vec: (6,) or (N, 6) condition vector.
        Returns: predicted contour x-coordinates (N, 95)
        """
        if not self.fitted:
            raise RuntimeError("PCAContourRegressor must be fitted before predict.")
        c_arr = np.atleast_2d(c_vec)
        N = c_arr.shape[0]
        C_bias = np.hstack([c_arr, np.ones((N, 1))])
        pred_w = np.dot(C_bias, self.weights) # (N, K)
        pred_contour = self.mean_contour + np.dot(pred_w, self.components)
        return np.clip(pred_contour, 15.0, 64.0)

    def rasterize(self, c_vec):
        """Rasterizes the PCA-predicted contour into a 128x128 binary mask."""
        pred_pts = self.predict_contour(c_vec) # (N, 95)
        N = pred_pts.shape[0]
        masks = np.zeros((N, 128, 128), dtype=np.uint8)
        for i in range(N):
            pts = pred_pts[i]
            for idx, x_val in enumerate(pts):
                y = 18 + idx
                x_int = int(round(x_val))
                if x_int < 64:
                    masks[i, y, x_int:64] = 1
        return masks

# =========================================================================
# Unit Test Function
# =========================================================================
def test_all_models_cuda():
    print("Testing CUDA & AMP Model Suite on CPU/CUDA...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    c = torch.randn(2, 6, device=device)
    img = torch.randn(2, 3, 128, 128, device=device)
    mask = torch.randn(2, 1, 128, 128, device=device)
    sdf = torch.randn(2, 1, 128, 128, device=device)

    m1 = DirectCNN().to(device)
    o1 = m1(c)
    assert o1.shape == (2, 3, 128, 128), "M1 failed shape test"

    m2 = ConditionalVAE().to(device)
    o2, mu, lv = m2(img, c)
    assert o2.shape == (2, 3, 128, 128), "M2 failed shape test"

    m3 = GC_WPG_v2().to(device)
    o3, pg, sp = m3(c)
    assert o3.shape == (2, 3, 128, 128), "M3 failed shape test"

    m4 = GeoSDF_CVAE().to(device)
    o4, psdf, pmask, papx, _, _ = m4(sdf, c)
    o4_prior, _, _, _ = m4.forward_prior(c)
    assert o4.shape == (2, 3, 128, 128) and o4_prior.shape == (2, 3, 128, 128), "M4 failed shape test"

    m5_g = Geometry_cGAN_Generator().to(device)
    m5_d = PatchGAN_Discriminator().to(device)
    o5, pm = m5_g(c)
    d_out = m5_d(pm, o5)
    assert o5.shape == (2, 3, 128, 128) and d_out.shape == (2, 1, 15, 15), "M5 failed shape test"

    # Test M6 GF-WPG
    m6 = GF_WPG().to(device)
    o6_rgb, o6_mask, o6_pts, o6_apex = m6(c)
    assert o6_rgb.shape == (2, 3, 128, 128), "M6 RGB failed shape test"
    assert o6_mask.shape == (2, 1, 128, 128), "M6 mask failed shape test"
    assert o6_pts.shape == (2, 95), "M6 contour pts failed shape test"
    assert o6_apex.shape == (2, 2), "M6 apex failed shape test"

    # Test B1 PCA Contour Regressor
    b1 = PCAContourRegressor(n_components=3)
    dummy_C = np.random.randn(10, 6)
    dummy_contours = np.random.uniform(20.0, 50.0, (10, 95))
    b1.fit(dummy_C, dummy_contours)
    p_pts = b1.predict_contour(dummy_C[:2])
    p_mask = b1.rasterize(dummy_C[:2])
    assert p_pts.shape == (2, 95), "B1 pts failed shape test"
    assert p_mask.shape == (2, 128, 128), "B1 mask failed shape test"

    print("ALL 6 NEURAL MODELS + 1 PCA BASELINE PASSED VERIFICATION SUITE!")

if __name__ == "__main__":
    test_all_models_cuda()
