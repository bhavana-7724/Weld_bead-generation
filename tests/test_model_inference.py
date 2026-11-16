"""
test_model_inference.py

Automated Test Suite for Weld Bead Prediction and Generative Architecture Inference.
Tests:
1. Checkpoint loading and SHA256 integrity verification.
2. Input parameter validation and domain boundary checks.
3. Deterministic repeatability of inference outputs (L_inf = 0).
4. Output tensor dimensions and dynamic range (128x128x3, uint8 [0, 255]).
5. External crown contour extraction and apex localization.
6. Prospective numerical regression inference (Width, Depth, Reinforcement in mm).
7. Out-of-domain warning triggers and graceful degradation.
"""

import os
import sys
import unittest
import numpy as np

# Ensure app directory is importable
APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from inference_engine import (
    WeldPredictor,
    compute_sha256,
    EXPECTED_HASHES,
    DIRECT_CNN_CKPT,
    GC_WPG_V2_CKPT,
    REGRESSOR_CKPT
)

class TestWeldInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = WeldPredictor()

    def test_01_checkpoints_exist_and_hashes_match(self):
        """Verifies all checkpoints exist and match SHA256 expected hashes."""
        self.assertTrue(os.path.exists(DIRECT_CNN_CKPT), f"Missing {DIRECT_CNN_CKPT}")
        self.assertTrue(os.path.exists(GC_WPG_V2_CKPT), f"Missing {GC_WPG_V2_CKPT}")
        self.assertTrue(os.path.exists(REGRESSOR_CKPT), f"Missing {REGRESSOR_CKPT}")

        h_direct = compute_sha256(DIRECT_CNN_CKPT)
        h_gc = compute_sha256(GC_WPG_V2_CKPT)
        h_reg = compute_sha256(REGRESSOR_CKPT)

        self.assertEqual(h_direct, EXPECTED_HASHES["direct_cnn"], "Direct CNN hash mismatch")
        self.assertEqual(h_gc, EXPECTED_HASHES["gc_wpg_v2"], "GC-WPG-v2 hash mismatch")
        self.assertEqual(h_reg, EXPECTED_HASHES["regressor"], "Regressor hash mismatch")

    def test_02_checkpoint_loading_status(self):
        """Verifies checkpoint status dictionary confirms all models loaded successfully."""
        status = self.predictor.checkpoint_status
        self.assertTrue(status["direct_cnn"]["loaded"])
        self.assertTrue(status["gc_wpg_v2"]["loaded"])
        self.assertTrue(status["regressor"]["loaded"])

    def test_03_input_validation_in_domain(self):
        """Tests that standard in-domain parameters pass validation without warnings."""
        is_valid, is_ood, warnings = self.predictor.validate_inputs("IN718", "ColdArc", 5.0, 25.0)
        self.assertTrue(is_valid)
        self.assertFalse(is_ood)
        self.assertEqual(len(warnings), 0)

    def test_04_input_validation_out_of_domain(self):
        """Tests that out-of-domain parameters trigger warnings but remain valid."""
        is_valid, is_ood, warnings = self.predictor.validate_inputs("SS304L", "GMAW", 6.5, 35.0)
        self.assertTrue(is_valid)
        self.assertTrue(is_ood)
        self.assertGreaterEqual(len(warnings), 1)

    def test_05_input_validation_invalid_categorical(self):
        """Tests that unknown material or process is rejected."""
        is_valid, is_ood, warnings = self.predictor.validate_inputs("Titanium", "ColdArc", 5.0, 25.0)
        self.assertFalse(is_valid)
        is_valid2, is_ood2, warnings2 = self.predictor.validate_inputs("IN718", "Laser", 5.0, 25.0)
        self.assertFalse(is_valid2)

    def test_06_numerical_prediction_properties(self):
        """Tests that prospective SVR numerical prediction yields positive, realistic bead dimensions."""
        res = self.predictor.predict_numerical("IN718", "ColdArc", 5.0, 25.0)
        self.assertTrue(res["available"])
        self.assertIn("width_mm", res)
        self.assertIn("depth_mm", res)
        self.assertIn("reinforcement_mm", res)
        self.assertGreater(res["width_mm"], 2.0)
        self.assertGreater(res["depth_mm"], 0.5)
        self.assertGreater(res["reinforcement_mm"], 0.5)
        self.assertLess(res["width_mm"], 20.0)
        self.assertLess(res["depth_mm"], 10.0)
        self.assertLess(res["reinforcement_mm"], 10.0)

    def test_07_generative_direct_cnn_dimensions_and_types(self):
        """Verifies Direct CNN outputs exact 128x128x3 uint8 array in [0, 255]."""
        res, err = self.predictor.generate_image("direct_cnn", "IN718", "ColdArc", 5.0, 25.0)
        self.assertIsNone(err)
        img = res["raw_rgb_uint8"]
        self.assertEqual(img.shape, (128, 128, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertGreaterEqual(img.min(), 0)
        self.assertLessEqual(img.max(), 255)

    def test_08_generative_gc_wpg_v2_dimensions_and_prior(self):
        """Verifies GC-WPG-v2 outputs exact 128x128x3 uint8 array and spatial prior."""
        res, err = self.predictor.generate_image("gc_wpg_v2", "SS304L", "GMAW", 5.25, 24.5)
        self.assertIsNone(err)
        img = res["raw_rgb_uint8"]
        self.assertEqual(img.shape, (128, 128, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertIsNotNone(res["prior_map"])
        self.assertEqual(res["prior_map"].shape, (2, 128, 128))

    def test_09_deterministic_repeatability(self):
        """Verifies that running the model multiple times on identical inputs produces identical results."""
        res1, _ = self.predictor.generate_image("direct_cnn", "IN718", "ColdArc", 5.0, 25.0)
        res2, _ = self.predictor.generate_image("direct_cnn", "IN718", "ColdArc", 5.0, 25.0)
        diff = np.max(np.abs(res1["raw_rgb_uint8"].astype(int) - res2["raw_rgb_uint8"].astype(int)))
        self.assertEqual(diff, 0, "Direct CNN output must be strictly deterministic")

        res_gc1, _ = self.predictor.generate_image("gc_wpg_v2", "SS304L", "GMAW", 5.25, 24.5)
        res_gc2, _ = self.predictor.generate_image("gc_wpg_v2", "SS304L", "GMAW", 5.25, 24.5)
        diff_gc = np.max(np.abs(res_gc1["raw_rgb_uint8"].astype(int) - res_gc2["raw_rgb_uint8"].astype(int)))
        self.assertEqual(diff_gc, 0, "GC-WPG-v2 output must be strictly deterministic")

    def test_10_contour_extraction_behavior(self):
        """Verifies crown contour extraction locates apex and non-negative projection."""
        res, _ = self.predictor.generate_image("direct_cnn", "IN718", "ColdArc", 5.0, 25.0)
        contour_info = self.predictor.extract_contour_and_render_overlays(res["raw_rgb_uint8"])
        self.assertFalse(contour_info["is_failed"])
        self.assertGreater(contour_info["reinf_projection_px"], 0)
        self.assertLess(contour_info["apex_x_px"], 64)
        self.assertGreater(contour_info["extracted_rows"], 20)
        self.assertGreater(contour_info["active_area_px_count"], 100)

    def test_11_full_pipeline_execution(self):
        """Verifies full inference pipeline runs seamlessly and returns complete payload."""
        full_res = self.predictor.run_full_inference("IN718", "ColdArc", 5.0, 25.0, model_choice="both")
        self.assertTrue(full_res["success"])
        self.assertIn("numerical_predictions", full_res)
        self.assertIn("generative_results", full_res)
        self.assertIn("direct_cnn", full_res["generative_results"])
        self.assertIn("gc_wpg_v2", full_res["generative_results"])
        self.assertIn("provenance", full_res)

if __name__ == "__main__":
    unittest.main()
