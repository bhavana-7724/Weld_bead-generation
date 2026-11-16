"""
models_v13.py

Phase V13: Enhanced Geometry-First Weld Profile Architecture (GF-WPG v2)
and Comparative Model Suite.

Key Architectural Enhancements in GF-WPG v2:
1. Explicit Crown Geometry + Landmark Predictor (Stage 1):
   - Discrete scanline crown contour x(y) for y in [18, 112] (95 points).
   - Explicit Crown Apex coordinates (x_apex, y_apex).
   - Explicit Upper and Lower Toe coordinates (y_toe_top, y_toe_bot).
   - Geometric loss formulation directly optimizing contour MAE, apex error,
     and toe attachment points.

2. Toe-Gated Differentiable Soft Rasterizer (Stage 2):
   - Smooth horizontal sigmoid transition: sigma(64 - x) * sigma(x - x_pred(y)).
   - Vertical toe gating: sigma(y - y_toe_top) * sigma(y_toe_bot - y).
   - Fully differentiable end-to-end, guaranteeing clean substrate attachment
     without unconstrained vertical protrusions.

3. High-Fidelity Multi-Scale Neural Renderer with FiLM (Stage 3):
   - Multi-scale Feature-wise Linear Modulation (FiLM) conditioned on process vector c.
   - Multi-scale spatial geometry fusion at 16x16, 32x32, 64x64, and 128x128.
   - Multi-term loss: L1 RGB + Crown-weighted L1 + Differentiable SSIM + Spatial Gradient
     loss to eliminate haze, preserve crisp metallic contrast, and avoid oversmoothing.

4. Baseline Wrappers:
   - DirectCNN (M1), GC-WPG-v2 (M3), Geometry_cGAN_Generator (M5), PCAContourRegressor (B1).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =========================================================================
# Shared Building Blocks
# =========================================================================
class ResUpBlock(nn.Module):
    """Residual upsampling block with ConvTranspose2d and dual Conv2d."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_up = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        h = self.act(self.bn_up(self.up(x)))
        res = self.act(self.bn1(self.conv1(h)))
        res = self.bn2(self.conv2(res))
        return self.act(h + res)


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: conditions feature channels on vector c."""
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.scale = nn.Linear(cond_dim, channels)
        self.shift = nn.Linear(cond_dim, channels)
        # Initialize identity modulation
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x, c):
        gamma = self.scale(c).unsqueeze(-1).unsqueeze(-1)
        beta = self.shift(c).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class MultiScaleSpatialFusion(nn.Module):
    """Fuses intermediate convolutional feature maps with downsampled mask."""
    def __init__(self, channels, prior_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels + prior_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, feat, mask_128):
        h, w = feat.shape[2], feat.shape[3]
        mask_scaled = F.interpolate(mask_128, size=(h, w), mode='bilinear', align_corners=False)
        return feat + self.conv(torch.cat([feat, mask_scaled], dim=1))


# =========================================================================
# Differentiable SSIM and Gradient Loss Functions
# =========================================================================
def create_gaussian_window(window_size=11, sigma=1.5, channel=3):
    """Creates a 2D Gaussian filter kernel for SSIM."""
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


class DifferentiableSSIM(nn.Module):
    """Differentiable SSIM loss for PyTorch."""
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', create_gaussian_window(window_size, 1.5, channel))

    def forward(self, img1, img2):
        window = self.window.to(dtype=img1.dtype, device=img1.device)
        padding = self.window_size // 2

        mu1 = F.conv2d(img1, window, padding=padding, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=padding, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=self.channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
        return torch.clamp(ssim_map.mean(), 0.0, 1.0)


class SpatialGradientLoss(nn.Module):
    """Penalizes loss of sharp edges using image gradient differences."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        loss_x = F.l1_loss(pred_dx, target_dx)
        loss_y = F.l1_loss(pred_dy, target_dy)
        return loss_x + loss_y


# =========================================================================
# GF-WPG v2 Stage 1: Explicit Geometry Network
# =========================================================================
class GeometryNet_v2(nn.Module):
    """
    Predicts:
    1. 95 scanline x-coordinates x(y) for y in [18, 112].
    2. Crown Apex coordinates (x_apex, y_apex).
    3. Crown Toe coordinates (y_toe_top, y_toe_bot).
    """
    def __init__(self, cond_dim=6, num_contour_pts=95):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_contour_pts = num_contour_pts

        self.net = nn.Sequential(
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
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Total outputs: 95 (scanlines) + 2 (apex_x, apex_y) + 2 (toe_top, toe_bot) = 99
            nn.Linear(128, num_contour_pts + 4)
        )

    def forward(self, c):
        raw = self.net(c)
        
        # 1. 95 scanline contour x-coordinates
        raw_pts = raw[:, :self.num_contour_pts]
        pred_pts = 64.0 - F.softplus(raw_pts)
        pred_pts = torch.clamp(pred_pts, min=15.0, max=64.0)

        # 2. Apex coordinates: x_apex in [15, 60], y_apex in [18, 112]
        raw_apex = raw[:, self.num_contour_pts : self.num_contour_pts + 2]
        apex_x = 64.0 - F.softplus(raw_apex[:, 0:1])
        apex_x = torch.clamp(apex_x, min=15.0, max=60.0)
        apex_y = torch.sigmoid(raw_apex[:, 1:2]) * 94.0 + 18.0
        pred_apex = torch.cat([apex_x, apex_y], dim=-1)

        # 3. Toe coordinates: y_toe_top in [18, 58], y_toe_bot in [70, 112]
        raw_toes = raw[:, self.num_contour_pts + 2 : self.num_contour_pts + 4]
        toe_top = torch.sigmoid(raw_toes[:, 0:1]) * 40.0 + 18.0
        toe_bot = torch.sigmoid(raw_toes[:, 1:2]) * 42.0 + 70.0
        pred_toes = torch.cat([toe_top, toe_bot], dim=-1)

        return pred_pts, pred_apex, pred_toes


# =========================================================================
# GF-WPG v2 Stage 2: Toe-Gated Differentiable Soft Rasterizer
# =========================================================================
class DifferentiableRasterizer_v2(nn.Module):
    """
    Maps (pred_pts, pred_toes) to a smooth (B, 1, 128, 128) binary mask.
    Enforces toe bounds so the crown deposit terminates naturally at the substrate baseline.
    """
    def __init__(self, num_contour_pts=95, y_start=18):
        super().__init__()
        self.num_contour_pts = num_contour_pts
        self.y_start = y_start

        # Static coordinate grid: (1, 1, 1, 128)
        xs = torch.arange(128).view(1, 1, 1, 128).float()
        self.register_buffer('xs_grid', xs)

    def forward(self, pred_pts, pred_toes):
        b = pred_pts.size(0)
        device = pred_pts.device
        dtype = pred_pts.dtype
        mask = torch.zeros((b, 1, 128, 128), dtype=dtype, device=device)
        xs = self.xs_grid.to(device=device, dtype=dtype)

        toe_top = pred_toes[:, 0:1].unsqueeze(-1).unsqueeze(-1) # (b, 1, 1, 1)
        toe_bot = pred_toes[:, 1:2].unsqueeze(-1).unsqueeze(-1) # (b, 1, 1, 1)

        for idx in range(self.num_contour_pts):
            y = self.y_start + idx
            x_pred = pred_pts[:, idx:idx+1].unsqueeze(-1).unsqueeze(-1)

            # Horizontal crown profile gate: 1 between x_pred(y) and x_baseline=64
            h_gate = torch.sigmoid((64.0 - xs) * 6.0) * torch.sigmoid((xs - x_pred) * 6.0)

            # Vertical toe gate: smooth falloff outside [toe_top, toe_bot]
            y_tensor = torch.tensor(y, dtype=dtype, device=device)
            v_gate = torch.sigmoid((y_tensor - toe_top) * 2.5) * torch.sigmoid((toe_bot - y_tensor) * 2.5)

            mask[:, :, y:y+1, :] = h_gate * v_gate

        return mask


# =========================================================================
# GF-WPG v2 Stage 3: High-Fidelity Multi-Scale Neural Renderer
# =========================================================================
class NeuralRenderer_v2(nn.Module):
    """
    Synthesizes crisp, high-contrast metallographic RGB cross-sections.
    Employs multi-scale FiLM conditioning and progressive spatial geometry fusion.
    """
    def __init__(self, cond_dim=6):
        super().__init__()
        self.cond_dim = cond_dim

        # Condition expansion
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling backbone with FiLM and Spatial Fusion
        self.up1 = ResUpBlock(128, 64)       # 8x8 -> 16x16
        self.film1 = FiLMBlock(cond_dim, 64)
        self.fuse1 = MultiScaleSpatialFusion(64, 1)

        self.up2 = ResUpBlock(64, 32)        # 16x16 -> 32x32
        self.film2 = FiLMBlock(cond_dim, 32)
        self.fuse2 = MultiScaleSpatialFusion(32, 1)

        self.up3 = ResUpBlock(32, 16)        # 32x32 -> 64x64
        self.film3 = FiLMBlock(cond_dim, 16)
        self.fuse3 = MultiScaleSpatialFusion(16, 1)

        self.up4 = ResUpBlock(16, 16)        # 64x64 -> 128x128
        self.film4 = FiLMBlock(cond_dim, 16)
        self.fuse4 = MultiScaleSpatialFusion(16, 1)

        # High-Fidelity RGB Head
        self.rgb_head = nn.Sequential(
            nn.Conv2d(16 + 1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, c, pred_mask):
        h = self.fc(c).reshape(-1, 128, 8, 8)

        f1 = self.fuse1(self.film1(self.up1(h), c), pred_mask)
        f2 = self.fuse2(self.film2(self.up2(f1), c), pred_mask)
        f3 = self.fuse3(self.film3(self.up3(f2), c), pred_mask)
        f4 = self.fuse4(self.film4(self.up4(f3), c), pred_mask)

        # Concat final mask with f4
        out_cat = torch.cat([f4, pred_mask], dim=1)
        pred_rgb = self.rgb_head(out_cat)
        return pred_rgb


# =========================================================================
# Integrated GF-WPG v2 Model
# =========================================================================
class GF_WPG_v2(nn.Module):
    """
    Integrated Enhanced Geometry-First Weld Profile Generator (GF-WPG v2).
    
    Forward pipeline:
      c (6-dim)
        --> Stage 1: GeometryNet_v2 predicts (pred_pts, pred_apex, pred_toes)
        --> Stage 2: DifferentiableRasterizer_v2 rasterizes to pred_mask
        --> Stage 3: NeuralRenderer_v2 renders pred_rgb conditioned on (c, pred_mask)
    """
    def __init__(self, cond_dim=6, num_contour_pts=95):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_contour_pts = num_contour_pts

        self.geom_net = GeometryNet_v2(cond_dim=cond_dim, num_contour_pts=num_contour_pts)
        self.rasterizer = DifferentiableRasterizer_v2(num_contour_pts=num_contour_pts, y_start=18)
        self.renderer = NeuralRenderer_v2(cond_dim=cond_dim)

    def predict_geometry(self, c):
        return self.geom_net(c)

    def forward(self, c):
        pred_pts, pred_apex, pred_toes = self.geom_net(c)
        pred_mask = self.rasterizer(pred_pts, pred_toes)
        pred_rgb = self.renderer(c, pred_mask)
        return pred_rgb, pred_mask, pred_pts, pred_apex, pred_toes

    def generate(self, c):
        """Evaluation inference mode."""
        self.eval()
        with torch.no_grad():
            pred_rgb, pred_mask, pred_pts, pred_apex, pred_toes = self.forward(c)
            return pred_rgb, pred_mask, pred_pts, pred_apex, pred_toes


# =========================================================================
# GF-WPG v2 Loss Module
# =========================================================================
class GF_WPG_Loss(nn.Module):
    """
    Composite Multi-Objective Loss Formulation for GF-WPG v2:
      L = L_geom + 0.5*L_apex + 0.5*L_toe + 2.0*L_mask + 2.0*L_rgb + 2.0*L_crown + 1.0*L_ssim + 0.5*L_grad
    """
    def __init__(self):
        super().__init__()
        self.ssim_loss = DifferentiableSSIM(window_size=11, channel=3)
        self.grad_loss = SpatialGradientLoss()

    def forward(self, pred_rgb, pred_mask, pred_pts, pred_apex, pred_toes,
                gt_rgb, gt_mask, gt_pts, gt_apex, gt_toes):
        # 1. Geometry losses
        l_pts = F.l1_loss(pred_pts, gt_pts)
        l_apex = F.l1_loss(pred_apex, gt_apex)
        l_toe = F.l1_loss(pred_toes, gt_toes)

        # 2. Mask losses: BCE + Soft Dice
        bce_mask = F.binary_cross_entropy(pred_mask, gt_mask)
        inter = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
        union = pred_mask.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))
        dice_mask = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
        l_mask = bce_mask + dice_mask.mean()

        # 3. RGB Photometric losses
        l_rgb = F.l1_loss(pred_rgb, gt_rgb)
        l_crown = F.l1_loss(pred_rgb * gt_mask, gt_rgb * gt_mask)
        l_ssim = 1.0 - self.ssim_loss(pred_rgb, gt_rgb)
        l_grad = self.grad_loss(pred_rgb, gt_rgb)

        total_loss = (
            1.0 * l_pts +
            0.5 * l_apex +
            0.5 * l_toe +
            2.0 * l_mask +
            2.0 * l_rgb +
            2.0 * l_crown +
            1.0 * l_ssim +
            0.5 * l_grad
        )

        loss_dict = {
            "total": float(total_loss.item()),
            "pts": float(l_pts.item()),
            "apex": float(l_apex.item()),
            "toe": float(l_toe.item()),
            "mask": float(l_mask.item()),
            "rgb": float(l_rgb.item()),
            "crown": float(l_crown.item()),
            "ssim": float(l_ssim.item()),
            "grad": float(l_grad.item())
        }
        return total_loss, loss_dict


# =========================================================================
# Baseline Architectures for Comparison (M1, M3, M5, B1)
# =========================================================================
class DirectCNN(nn.Module):
    """M1: Direct Conditional ResUp CNN."""
    def __init__(self, cond_dim=6, latent_dim=16):
        super().__init__()
        self.cond_dim = cond_dim
        self.fc = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, 128 * 8 * 8),
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

    def forward(self, c):
        z_dummy = torch.zeros((c.size(0), 16), dtype=c.dtype, device=c.device)
        h = self.fc(torch.cat([c, z_dummy], dim=1)).reshape(-1, 128, 8, 8)
        feat = self.up4(self.up3(self.up2(self.up1(h))))
        return self.head(feat)

    def generate(self, c):
        self.eval()
        with torch.no_grad():
            return self.forward(c)


class GC_WPG_v2(nn.Module):
    """M3: Multi-Scale Spatial Prior CNN."""
    def __init__(self, cond_dim=6):
        super().__init__()
        self.prior_net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mask_head = nn.Sequential(
            ResUpBlock(128, 64),
            ResUpBlock(64, 32),
            ResUpBlock(32, 16),
            ResUpBlock(16, 8),
            nn.Conv2d(8, 2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.rgb_fc = nn.Sequential(
            nn.Linear(cond_dim, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.rgb_up1 = ResUpBlock(128, 64)
        self.rgb_up2 = ResUpBlock(64, 32)
        self.rgb_up3 = ResUpBlock(32, 16)
        self.rgb_up4 = ResUpBlock(16, 16)
        self.rgb_head = nn.Sequential(
            nn.Conv2d(16 + 2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, c):
        h_prior = self.prior_net(c).reshape(-1, 128, 8, 8)
        spatial_prior = self.mask_head(h_prior)

        h_rgb = self.rgb_fc(c).reshape(-1, 128, 8, 8)
        feat = self.rgb_up4(self.rgb_up3(self.rgb_up2(self.rgb_up1(h_rgb))))
        rgb = self.rgb_head(torch.cat([feat, spatial_prior], dim=1))
        return rgb, spatial_prior

    def generate(self, c):
        self.eval()
        with torch.no_grad():
            rgb, spatial_prior = self.forward(c)
            return rgb, spatial_prior[:, 0:1]


class Geometry_cGAN_Generator(nn.Module):
    """M5: Geometry-Conditioned cGAN Generator."""
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

    def generate(self, c):
        self.eval()
        with torch.no_grad():
            return self.forward(c)


class PCAContourRegressor:
    """B1: PCA-Contour + Ridge Regression Baseline."""
    def __init__(self, n_components=4, alpha=1.0):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        self.pca = PCA(n_components=n_components)
        self.reg = Ridge(alpha=alpha)
        self.fitted = False

    def fit(self, X, y):
        # y: (N, 95)
        coeffs = self.pca.fit_transform(y)
        self.reg.fit(X, coeffs)
        self.fitted = True

    def predict(self, X):
        pred_coeffs = self.reg.predict(X)
        pred_y = self.pca.inverse_transform(pred_coeffs)
        return np.clip(pred_y, 15.0, 64.0)
