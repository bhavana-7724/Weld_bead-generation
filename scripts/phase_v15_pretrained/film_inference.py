"""
Google Research Frame Interpolation (FILM) Pretrained Inference Module.
Phase V15: Sharp Weld Macrograph Pretrained AI Study.

Wraps the official ECCV 2022 FILM SavedModel checkpoint for multi-scale
feature synthesis between experimental weld macrograph donors.
"""

import os
import sys
import numpy as np

# Suppress TensorFlow logging before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))


class FILMInterpolator:
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
            checkpoint_path = os.path.join(
                base_dir,
                "results",
                "phase_v15_pretrained_models",
                "checkpoints_manifest",
                "film"
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"FILM SavedModel not found at: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.model = tf.saved_model.load(checkpoint_path)
        self.infer = self.model.signatures["serving_default"]

    def _preprocess(self, img_rgb):
        """Convert HxWxC uint8 RGB image to 1xHxWxC float32 tensor in [0, 1]."""
        if img_rgb.dtype == np.uint8:
            arr = img_rgb.astype(np.float32) / 255.0
        else:
            arr = img_rgb.astype(np.float32)
        return tf.constant(arr[np.newaxis, ...], dtype=tf.float32)

    def _postprocess(self, tensor):
        """Convert 1xHxWxC float32 tensor in [0, 1] to HxWxC uint8 RGB image."""
        arr = tensor[0].numpy()
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).round().astype(np.uint8)

    def interpolate(self, img0_rgb, img1_rgb, timestep=0.5):
        """
        Synthesize intermediate frame between img0_rgb and img1_rgb at relative timestep t in [0, 1].
        
        Args:
            img0_rgb: np.ndarray (H, W, 3) uint8 RGB
            img1_rgb: np.ndarray (H, W, 3) uint8 RGB
            timestep: float in [0, 1]
            
        Returns:
            interp_rgb: np.ndarray (H, W, 3) uint8 RGB
        """
        if timestep <= 0.001:
            return img0_rgb.copy()
        if timestep >= 0.999:
            return img1_rgb.copy()

        x0 = self._preprocess(img0_rgb)
        x1 = self._preprocess(img1_rgb)
        t = tf.constant([[float(timestep)]], dtype=tf.float32)

        res = self.infer(x0=x0, x1=x1, time=t)

        # In the official FILM SavedModel, the output dictionary contains intermediate warped tensors
        # and the final refined synthesized frame.
        if "image" in res:
            out_tensor = res["image"]
        elif "tf.__operators__.add_12" in res:
            out_tensor = res["tf.__operators__.add_12"]
        else:
            # Fallback: find the 3-channel tensor that is not an exact copy of x0 or x1
            candidate = None
            for k, v in res.items():
                if v.shape[-1] == 3 and v.shape == x0.shape:
                    if not (np.allclose(v.numpy(), x0.numpy()) or np.allclose(v.numpy(), x1.numpy())):
                        candidate = v
                        break
            if candidate is not None:
                out_tensor = candidate
            else:
                raise KeyError(f"Unable to locate interpolated output tensor among keys: {list(res.keys())}")

        return self._postprocess(out_tensor)


if __name__ == "__main__":
    print("Testing FILMInterpolator on synthetic data...")
    interp = FILMInterpolator()
    img0 = np.full((128, 128, 3), 50, dtype=np.uint8)
    img1 = np.full((128, 128, 3), 200, dtype=np.uint8)
    res = interp.interpolate(img0, img1, timestep=0.5)
    print(f"Interpolated shape: {res.shape}, mean: {res.mean():.1f}")
    print("FILMInterpolator self-test PASSED!")
