"""
RIFE (Real-Time Intermediate Flow Estimation) Pretrained Inference Module.
Phase V15: Sharp Weld Macrograph Pretrained AI Study.

Wraps the official ECCV 2022 RIFE IFNet checkpoint for optical flow estimation
and frame synthesis between experimental weld macrograph donors.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Ensure rife_model can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from rife_model.IFNet import IFNet
    from rife_model.warplayer import warp
except ImportError:
    try:
        from scripts.phase_v15_pretrained.rife_model.IFNet import IFNet
        from scripts.phase_v15_pretrained.rife_model.warplayer import warp
    except ImportError:
        from model.IFNet import IFNet
        from model.warplayer import warp


class RIFEInterpolator:
    def __init__(self, checkpoint_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            # Default location in phase_v15 checkpoints manifest
            base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
            checkpoint_path = os.path.join(
                base_dir,
                "results",
                "phase_v15_pretrained_models",
                "checkpoints_manifest",
                "rife",
                "RIFE_train_log",
                "flownet.pkl"
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"RIFE checkpoint not found at: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.model = IFNet().to(self.device)
        self._load_weights(checkpoint_path)
        self.model.eval()

    def _load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        # Strip DataParallel 'module.' prefix if present
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_sd, strict=True)

    def _preprocess(self, img_rgb):
        """Convert HxWxC uint8 RGB image to 1x3xHxW float tensor in [0, 1]."""
        if img_rgb.dtype == np.uint8:
            tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        else:
            tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
        return tensor.unsqueeze(0).to(self.device)

    def _postprocess(self, tensor):
        """Convert 1x3xHxW float tensor in [0, 1] to HxWxC uint8 RGB image."""
        arr = tensor.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        return (arr * 255.0).round().astype(np.uint8)

    def _pad_if_needed(self, img0_t, img1_t):
        """Pad images to multiples of 32 for IFNet multi-scale pyramid."""
        _, _, h, w = img0_t.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        pad_bottom = ph - h
        pad_right = pw - w
        if pad_bottom > 0 or pad_right > 0:
            p0 = F.pad(img0_t, (0, pad_right, 0, pad_bottom), mode='replicate')
            p1 = F.pad(img1_t, (0, pad_right, 0, pad_bottom), mode='replicate')
            return p0, p1, (h, w)
        return img0_t, img1_t, (h, w)

    @torch.no_grad()
    def interpolate(self, img0_rgb, img1_rgb, timestep=0.5, return_flow=False):
        """
        Synthesize intermediate frame between img0_rgb and img1_rgb at relative timestep t in [0, 1].
        
        Args:
            img0_rgb: np.ndarray (H, W, 3) uint8 RGB
            img1_rgb: np.ndarray (H, W, 3) uint8 RGB
            timestep: float in [0, 1]
            return_flow: bool, whether to return estimated optical flow field
            
        Returns:
            interp_rgb: np.ndarray (H, W, 3) uint8 RGB
            (optional) flow: np.ndarray (H, W, 4)
        """
        if timestep <= 0.001:
            if return_flow:
                return img0_rgb.copy(), np.zeros((*img0_rgb.shape[:2], 4), dtype=np.float32)
            return img0_rgb.copy()
        if timestep >= 0.999:
            if return_flow:
                return img1_rgb.copy(), np.zeros((*img1_rgb.shape[:2], 4), dtype=np.float32)
            return img1_rgb.copy()

        t0 = self._preprocess(img0_rgb)
        t1 = self._preprocess(img1_rgb)
        p0, p1, orig_shape = self._pad_if_needed(t0, t1)
        h, w = orig_shape

        inp = torch.cat((p0, p1), dim=1)
        # IFNet forward pass returns: flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill
        out = self.model(inp)
        flow_list = out[0]
        final_flow = flow_list[-1]  # shape (1, 4, H, W): [0:2] is F_0->t, [2:4] is F_1->t
        mask = out[1]               # shape (1, 1, H, W)
        merged_list = out[2]
        mid_frame = merged_list[-1] # shape (1, 3, H, W)

        if abs(timestep - 0.5) < 1e-4:
            # Exact midpoint
            res_t = mid_frame[:, :, :h, :w]
        else:
            # General timestep t in (0, 1):
            # Scale bidirectional flows proportionally:
            # At t=0.5, flow[:, :2] is half-flow from 0 to 1, flow[:, 2:4] is half-flow from 1 to 0.
            # For general t, displacement from 0 is 2*t*flow[:, :2]
            # and displacement from 1 is 2*(1-t)*flow[:, 2:4]
            scale0 = 2.0 * timestep
            scale1 = 2.0 * (1.0 - timestep)
            f0 = final_flow[:, :2] * scale0
            f1 = final_flow[:, 2:4] * scale1
            warped0 = warp(p0, f0)
            warped1 = warp(p1, f1)
            # Blend using timestep weight and learned mask
            weight0 = (1.0 - timestep)
            weight1 = timestep
            res_t = warped0 * (weight0 * mask + (1.0 - timestep) * (1.0 - mask)) + \
                    warped1 * (weight1 * (1.0 - mask) + timestep * mask)
            # Normalize blend weights
            res_t = torch.clamp(res_t[:, :, :h, :w], 0.0, 1.0)

        res_rgb = self._postprocess(res_t)

        if return_flow:
            flow_np = final_flow[:, :, :h, :w].squeeze(0).permute(1, 2, 0).cpu().numpy()
            return res_rgb, flow_np

        return res_rgb

    @staticmethod
    def flow_to_rgb(flow_2d):
        """Convert a 2-channel flow field (H, W, 2) [dx, dy] to an HSV-colored RGB image."""
        h, w = flow_2d.shape[:2]
        fx, fy = flow_2d[:, :, 0], flow_2d[:, :, 1]
        mag, ang = cv2.cartToPolar(fx, fy)
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    print("Testing RIFEInterpolator on synthetic data...")
    interp = RIFEInterpolator()
    img0 = np.full((128, 128, 3), 50, dtype=np.uint8)
    img1 = np.full((128, 128, 3), 200, dtype=np.uint8)
    res, flow = interp.interpolate(img0, img1, timestep=0.5, return_flow=True)
    print(f"Interpolated shape: {res.shape}, mean: {res.mean():.1f}, flow shape: {flow.shape}")
    print("RIFEInterpolator self-test PASSED!")
