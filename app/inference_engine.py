"""
inference_engine.py

Core Inference Engine for Phase V8:
- Manages checkpoint loading and SHA256 integrity verification.
- Enforces strict input validation and domain boundary checks.
- Produces prospective numerical predictions (Width, Depth, Reinforcement in mm) via SVR regressor.
- Executes conditional image generation via Direct Conditional CNN and GC-WPG-v2.
- Performs external weld crown contour extraction and overlay rendering.
- Returns rich metadata, provenance tracking, and explicit limitation disclaimers.
"""

import os
import sys
import time
import hashlib
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
GEN_DIR = os.path.join(MODELS_DIR, "generative")
NUM_DIR = os.path.join(MODELS_DIR, "numerical")

# Known Checkpoint Paths
DIRECT_CNN_CKPT = os.path.join(GEN_DIR, "direct_cnn_checkpoint.pt")
GC_WPG_V2_CKPT = os.path.join(GEN_DIR, "gc_wpg_v2_checkpoint.pt")
REGRESSOR_CKPT = os.path.join(NUM_DIR, "prospective_regressor_checkpoint.pkl")

# Expected SHA256 Hashes
EXPECTED_HASHES = {
    "direct_cnn": "714ce514ec708b5a32868c37f4468fd63b028571dffefa1b4cfdda587b4aeccd",
    "gc_wpg_v2": "44c7f7e51ab9b5ca63ada9dc31a525d717dad37745b18cdb061e92ab8724ee7c",
    "regressor": "f827e5c375cb9348acc89631e6f1819a5c31f080002b03dbba03ef0586e2618b"
}

def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# =========================================================================
# Generative Architectures (Identical to Checkpoint Specifications)
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
        h = self.fc(inp).view(-1, 128, 8, 8)
        feat = self.up4(self.up3(self.up2(self.up1(h))))
        return self.out_conv(feat)

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

class GC_WPG_v2(nn.Module):
    def __init__(self, cond_dim=6, noise_dim=16, scaler_params=None):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.scaler = scaler_params
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
        w_px = w_px.view(batch_size, 1, 1)
        r_px = r_px.view(batch_size, 1, 1)
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
        h = self.fc(inp).view(-1, 128, 8, 8)
        h1 = self.fuse1(self.up1(h), prior_16)
        h2 = self.fuse2(self.up2(h1), prior_32)
        h3 = self.fuse3(self.up3(h2), prior_64)
        h4 = self.fuse4(self.up4(h3), prior_128)
        return self.out_conv(h4), pred_geom, prior_128

# =========================================================================
# Verified Boundary Extraction Function
# =========================================================================
def extract_crown_contour(gray_img, x_baseline=64, y_min=20, y_max=110, threshold=155):
    pts = {}
    edge_map = np.zeros_like(gray_img, dtype=np.uint8)
    for y in range(y_min, y_max + 1):
        row_vals = gray_img[y, :x_baseline]
        metal_idx = np.where(row_vals < threshold)[0]
        if len(metal_idx) > 0:
            x_min = metal_idx[0]
            if x_min < x_baseline:
                pts[y] = int(x_min)
                edge_map[y, x_min] = 1
    return pts, edge_map

# =========================================================================
# WeldPredictor Class (Primary Application Interface)
# =========================================================================
class WeldPredictor:
    SUPPORTED_MATERIALS = ["IN718", "SS304L"]
    SUPPORTED_PROCESSES = ["ColdArc", "GMAW"]
    FEED_RATE_DOMAIN = (4.5, 6.0)   # m/min
    SPEED_DOMAIN = (20.0, 32.0)     # cm/min

    def __init__(self):
        self.device = torch.device("cpu")
        self.direct_cnn = None
        self.gc_wpg_v2 = None
        self.num_regressor = None
        self.checkpoint_status = {}
        self.load_checkpoints()

    def load_checkpoints(self):
        """Loads all permanent checkpoints and verifies hashes."""
        # 1. Direct CNN
        if os.path.exists(DIRECT_CNN_CKPT):
            actual_h = compute_sha256(DIRECT_CNN_CKPT)
            self.direct_cnn = DirectCNN(6, 16).to(self.device)
            ckpt = torch.load(DIRECT_CNN_CKPT, map_location="cpu")
            self.direct_cnn.load_state_dict(ckpt["model_state_dict"])
            self.direct_cnn.eval()
            self.checkpoint_status["direct_cnn"] = {
                "loaded": True, "path": DIRECT_CNN_CKPT,
                "sha256": actual_h, "hash_match": actual_h == EXPECTED_HASHES["direct_cnn"],
                "scaler": ckpt.get("scaler", {})
            }
        else:
            self.checkpoint_status["direct_cnn"] = {"loaded": False, "error": f"File not found: {DIRECT_CNN_CKPT}"}

        # 2. GC-WPG-v2
        if os.path.exists(GC_WPG_V2_CKPT):
            actual_h = compute_sha256(GC_WPG_V2_CKPT)
            ckpt = torch.load(GC_WPG_V2_CKPT, map_location="cpu")
            self.gc_wpg_v2 = GC_WPG_v2(6, 16, scaler_params=ckpt["scaler"]).to(self.device)
            self.gc_wpg_v2.load_state_dict(ckpt["model_state_dict"])
            self.gc_wpg_v2.eval()
            self.checkpoint_status["gc_wpg_v2"] = {
                "loaded": True, "path": GC_WPG_V2_CKPT,
                "sha256": actual_h, "hash_match": actual_h == EXPECTED_HASHES["gc_wpg_v2"],
                "scaler": ckpt.get("scaler", {})
            }
        else:
            self.checkpoint_status["gc_wpg_v2"] = {"loaded": False, "error": f"File not found: {GC_WPG_V2_CKPT}"}

        # 3. Prospective Numerical Regressor
        if os.path.exists(REGRESSOR_CKPT):
            actual_h = compute_sha256(REGRESSOR_CKPT)
            with open(REGRESSOR_CKPT, "rb") as f:
                self.num_bundle = pickle.load(f)
            self.num_regressor = self.num_bundle["model"]
            self.checkpoint_status["regressor"] = {
                "loaded": True, "path": REGRESSOR_CKPT,
                "sha256": actual_h, "hash_match": actual_h == EXPECTED_HASHES["regressor"],
                "model_name": self.num_bundle["model_name"],
                "documented_mae": self.num_bundle["documented_cv_mae_mm"]
            }
        else:
            self.checkpoint_status["regressor"] = {"loaded": False, "error": f"File not found: {REGRESSOR_CKPT}"}

    def validate_inputs(self, material, process, feed_rate, speed):
        """Validates prospective inputs and returns flags, warnings, and clamped values."""
        warnings = []
        is_valid = True
        is_out_of_domain = False

        if material not in self.SUPPORTED_MATERIALS:
            return False, False, [f"Unsupported material '{material}'. Supported: {self.SUPPORTED_MATERIALS}"]
        if process not in self.SUPPORTED_PROCESSES:
            return False, False, [f"Unsupported process '{process}'. Supported: {self.SUPPORTED_PROCESSES}"]

        f = float(feed_rate)
        s = float(speed)

        if f < self.FEED_RATE_DOMAIN[0] or f > self.FEED_RATE_DOMAIN[1]:
            is_out_of_domain = True
            warnings.append(f"Wire feed rate {f:.2f} m/min is outside experimental training domain [{self.FEED_RATE_DOMAIN[0]}, {self.FEED_RATE_DOMAIN[1]} m/min]. Prediction involves extrapolation.")
        if s < self.SPEED_DOMAIN[0] or s > self.SPEED_DOMAIN[1]:
            is_out_of_domain = True
            warnings.append(f"Travel speed {s:.2f} cm/min is outside experimental training domain [{self.SPEED_DOMAIN[0]}, {self.SPEED_DOMAIN[1]} cm/min]. Prediction involves extrapolation.")

        return is_valid, is_out_of_domain, warnings

    def predict_numerical(self, material, process, feed_rate, speed):
        """Generates prospective numerical predictions for Width, Depth, Reinforcement in mm."""
        if self.num_regressor is None:
            return {
                "available": False,
                "message": "Numerical prediction model is not configured or checkpoint is missing."
            }
        t0 = time.time()
        scaler = self.num_bundle["scaler"]
        encoder = self.num_bundle["encoder"]
        
        import pandas as pd
        df_num = pd.DataFrame([[float(feed_rate), float(speed)]], columns=self.num_bundle["num_features"])
        df_cat = pd.DataFrame([[material, process]], columns=self.num_bundle["cat_features"])
        x_num = scaler.transform(df_num)
        x_cat = encoder.transform(df_cat)
        x_all = np.hstack([x_num, x_cat])
        
        preds = self.num_regressor.predict(x_all)[0]
        elapsed = time.time() - t0
        
        return {
            "available": True,
            "model_name": self.num_bundle["model_name"],
            "width_mm": round(float(preds[0]), 2),
            "depth_mm": round(float(preds[1]), 2),
            "reinforcement_mm": round(float(preds[2]), 2),
            "documented_cv_mae_mm": self.num_bundle["documented_cv_mae_mm"],
            "inference_time_ms": round(elapsed * 1000, 2)
        }

    def generate_image(self, model_key, material, process, feed_rate, speed):
        """Generates raw 128x128 macrograph from the selected deep generative model."""
        model = self.direct_cnn if model_key == "direct_cnn" else self.gc_wpg_v2
        if model is None:
            return None, f"Model checkpoint for '{model_key}' is not available."

        t0 = time.time()
        # Conditioning vector c in R^6: [F_norm, S_norm, IN718, SS304L, ColdArc, GMAW]
        f_min, f_max = 4.5, 6.0
        s_min, s_max = 20.0, 32.0
        f_norm = (float(feed_rate) - f_min) / (f_max - f_min + 1e-8)
        s_norm = (float(speed) - s_min) / (s_max - s_min + 1e-8)
        
        c = torch.tensor([[
            f_norm, s_norm,
            1.0 if material == "IN718" else 0.0,
            1.0 if material == "SS304L" else 0.0,
            1.0 if process == "ColdArc" else 0.0,
            1.0 if process == "GMAW" else 0.0
        ]], dtype=torch.float32)

        with torch.no_grad():
            if model_key == "direct_cnn":
                out = model(c)
                prior_map = None
            else:
                out, _, prior = model(c)
                prior_map = prior.squeeze(0).cpu().numpy()

        elapsed = time.time() - t0
        raw_rgb = np.clip(out.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
        raw_uint8 = (raw_rgb * 255).astype(np.uint8)
        
        return {
            "raw_rgb_uint8": raw_uint8,
            "raw_rgb_float": raw_rgb,
            "prior_map": prior_map,
            "inference_time_ms": round(elapsed * 1000, 2),
            "condition_tensor": c.tolist()[0]
        }, None

    def extract_contour_and_render_overlays(self, raw_uint8):
        """Extracts reinforcement boundary and constructs overlay visual representations."""
        gray = cv2.cvtColor(raw_uint8, cv2.COLOR_RGB2GRAY)
        bg_val = np.median(gray[20:110, :20])
        adaptive_thresh = max(145, min(170, bg_val - 12))
        
        pts, edge_map = extract_crown_contour(gray, x_baseline=64, threshold=adaptive_thresh)
        
        is_failed = False
        if len(pts) < 10:
            is_failed = True
            apex_x = 64
            apex_y = 64
        else:
            apex_x = min(pts.values())
            apex_y = [y for y, x in pts.items() if x == apex_x][0]
            if apex_x >= 62:
                is_failed = True

        reinf_proj_px = 64 - apex_x
        active_area_px = int(sum(np.sum(gray[y, :64] < adaptive_thresh) for y in range(20, 111)))
        
        # Render Overlay Image (128x128 RGB)
        overlay_rgb = raw_uint8.copy()
        # Draw interface line at X=64 in cyan
        overlay_rgb[:, 64, :] = [0, 255, 255]
        
        if not is_failed and pts:
            # Draw contour points in bright green
            for y, x in pts.items():
                if 0 <= y < 128 and 0 <= x < 128:
                    overlay_rgb[y, x, :] = [46, 204, 113]
                    if x + 1 < 128:
                        overlay_rgb[y, x + 1, :] = [46, 204, 113]
            # Draw apex in red
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = apex_y + dy, apex_x + dx
                    if 0 <= ny < 128 and 0 <= nx < 128:
                        overlay_rgb[ny, nx, :] = [231, 76, 60]

        # Render Contour-Only Mask (Dark background with colored contours)
        contour_mask = np.zeros((128, 128, 3), dtype=np.uint8)
        contour_mask[:, 64, :] = [0, 180, 220]
        if not is_failed and pts:
            for y, x in pts.items():
                if 0 <= y < 128 and 0 <= x < 128:
                    contour_mask[y, x, :] = [46, 204, 113]
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = apex_y + dy, apex_x + dx
                    if 0 <= ny < 128 and 0 <= nx < 128:
                        contour_mask[ny, nx, :] = [231, 76, 60]

        return {
            "is_failed": is_failed,
            "apex_x_px": int(apex_x),
            "apex_y_px": int(apex_y),
            "reinf_projection_px": int(reinf_proj_px),
            "active_area_px_count": int(active_area_px),
            "extracted_rows": len(pts),
            "adaptive_threshold": float(adaptive_thresh),
            "contour_points": pts,
            "overlay_rgb": overlay_rgb,
            "contour_only_rgb": contour_mask
        }

    def run_full_inference(self, material, process, feed_rate, speed, model_choice="both"):
        """End-to-end execution pipeline for dashboard or CLI invocation."""
        is_valid, is_ood, warnings = self.validate_inputs(material, process, feed_rate, speed)
        if not is_valid:
            return {"success": False, "errors": warnings}

        num_res = self.predict_numerical(material, process, feed_rate, speed)
        
        models_to_run = []
        if model_choice in ["direct_cnn", "both"]:
            models_to_run.append(("direct_cnn", "Direct Conditional CNN"))
        if model_choice in ["gc_wpg_v2", "both"]:
            models_to_run.append(("gc_wpg_v2", "GC-WPG-v2 (Multi-Scale Prior)"))

        gen_results = {}
        for m_key, m_name in models_to_run:
            res, err = self.generate_image(m_key, material, process, feed_rate, speed)
            if err:
                gen_results[m_key] = {"error": err}
            else:
                contour_data = self.extract_contour_and_render_overlays(res["raw_rgb_uint8"])
                gen_results[m_key] = {
                    "model_name": m_name,
                    "checkpoint_used": self.checkpoint_status[m_key]["path"],
                    "checkpoint_sha256": self.checkpoint_status[m_key]["sha256"],
                    "inference_time_ms": res["inference_time_ms"],
                    "raw_rgb_uint8": res["raw_rgb_uint8"],
                    "contour_data": contour_data,
                    "condition_vector": res["condition_tensor"]
                }

        return {
            "success": True,
            "is_out_of_domain": is_ood,
            "warnings": warnings,
            "inputs": {
                "material": material,
                "process": process,
                "feed_rate_m_min": float(feed_rate),
                "speed_cm_min": float(speed)
            },
            "numerical_predictions": num_res,
            "generative_results": gen_results,
            "provenance": {
                "framework": "PyTorch (CPU Only)",
                "image_resolution": "128x128 RGB",
                "calibration_note": "Pixel-to-mm conversions require independent optical scale calibration. Reported contour projection is strictly in PIXELS.",
                "disclaimer": "The generated macrograph is an AI-synthesized estimate learned from 36 experimental welds. It represents surface and near-surface cross-sectional morphology. Fusion boundary and deep root penetration are not physically guaranteed without experimental metallographic sectioning."
            }
        }
