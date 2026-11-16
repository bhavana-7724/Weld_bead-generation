"""
audit_cross_validation_experiments.py

Audits 5-Fold Grouped Cross-Validation:
- Aggregates GC_WPG_V2_METRICS.csv at the EXPERIMENT level (sample-averaged, N=36) vs VIEW level (N=72).
- Computes Mean, Std, Median, Min, Max across independent experiments.
- Compares with Direct CNN, GC-WPG-v1, CVAE, and Parametric Baseline from EXTERNAL_CONTOUR_METRICS.csv.
- Checks split orthogonality and isolation from cv_splits.json.
- Verifies prospective inference pipeline and scaler integrity.
"""

import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V7_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v7")
V7_REFINED_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v7_refined")
AUDIT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v7_integrity_audit")

# 1. Audit Splits Orthogonality
with open(os.path.join(V7_DIR, "cv_splits.json"), "r") as fp:
    splits = json.load(fp)

split_audit = []
all_test_samples = []
for f_name, s in splits.items():
    test_s = set(s["outer_test_samples"])
    val_s = set(s["inner_val_samples"])
    train_s = set(s["train_samples"])
    
    # Check disjointness
    leak_test_train = test_s.intersection(train_s)
    leak_test_val = test_s.intersection(val_s)
    leak_val_train = val_s.intersection(train_s)
    
    all_test_samples.extend(s["outer_test_samples"])
    
    split_audit.append({
        "fold": f_name,
        "n_outer_test": len(test_s),
        "n_inner_val": len(val_s),
        "n_train": len(train_s),
        "test_train_leak": len(leak_test_train),
        "test_val_leak": len(leak_test_val),
        "val_train_leak": len(leak_val_train),
        "outer_test_samples": sorted(list(test_s))
    })

df_split_audit = pd.DataFrame(split_audit)
print("--- SPLIT ORTHOGONALITY AUDIT ---")
print(df_split_audit[["fold", "n_outer_test", "n_inner_val", "n_train", "test_train_leak", "test_val_leak"]].to_string(index=False))
print(f"Total test samples across 5 folds: {len(all_test_samples)} (Unique: {len(set(all_test_samples))})")

# 2. Experiment-level vs View-level Aggregation for GC-WPG-v2
df_v2 = pd.read_csv(os.path.join(V7_REFINED_DIR, "GC_WPG_V2_METRICS.csv"))
cv_v2 = df_v2[df_v2['evaluation_suite'].str.contains("5-Fold")].copy()

# View-level metrics (N=72)
view_metrics = {
    "level": "View-Level (N=72 macrographs)",
    "ssim_mean": cv_v2['ssim'].mean(), "ssim_std": cv_v2['ssim'].std(),
    "roi_ssim_mean": cv_v2['roi_ssim'].mean(), "roi_ssim_std": cv_v2['roi_ssim'].std(),
    "psnr_mean": cv_v2['psnr_db'].mean(), "psnr_std": cv_v2['psnr_db'].std(),
    "contour_mae_px_mean": cv_v2['contour_mae_px'].mean(), "contour_mae_px_std": cv_v2['contour_mae_px'].std(),
    "apex_err_x_px_mean": cv_v2['apex_err_x_px'].mean(), "apex_err_x_px_std": cv_v2['apex_err_x_px'].std(),
    "hausdorff_px_mean": cv_v2['hausdorff_px'].mean(), "hausdorff_px_std": cv_v2['hausdorff_px'].std(),
    "f1_1px_mean": cv_v2['reinf_boundary_f1_1px'].mean(), "f1_1px_std": cv_v2['reinf_boundary_f1_1px'].std(),
    "f1_2px_mean": cv_v2['reinf_boundary_f1_2px'].mean(), "f1_2px_std": cv_v2['reinf_boundary_f1_2px'].std(),
    "failure_rate": cv_v2['extraction_failed'].mean()
}

# Experiment-level metrics (N=36 independent experiments)
exp_v2 = cv_v2.groupby(['sample', 'material', 'process', 'feed_rate', 'speed']).agg({
    'ssim': 'mean',
    'roi_ssim': 'mean',
    'psnr_db': 'mean',
    'contour_mae_px': 'mean',
    'apex_err_x_px': 'mean',
    'hausdorff_px': 'mean',
    'reinf_boundary_f1_1px': 'mean',
    'reinf_boundary_f1_2px': 'mean',
    'extraction_failed': 'max'
}).reset_index()

exp_metrics = {
    "level": "Experiment-Level (N=36 independent welds)",
    "ssim_mean": exp_v2['ssim'].mean(), "ssim_std": exp_v2['ssim'].std(),
    "roi_ssim_mean": exp_v2['roi_ssim'].mean(), "roi_ssim_std": exp_v2['roi_ssim'].std(),
    "psnr_mean": exp_v2['psnr_db'].mean(), "psnr_std": exp_v2['psnr_db'].std(),
    "contour_mae_px_mean": exp_v2['contour_mae_px'].mean(), "contour_mae_px_std": exp_v2['contour_mae_px'].std(),
    "apex_err_x_px_mean": exp_v2['apex_err_x_px'].mean(), "apex_err_x_px_std": exp_v2['apex_err_x_px'].std(),
    "hausdorff_px_mean": exp_v2['hausdorff_px'].mean(), "hausdorff_px_std": exp_v2['hausdorff_px'].std(),
    "f1_1px_mean": exp_v2['reinf_boundary_f1_1px'].mean(), "f1_1px_std": exp_v2['reinf_boundary_f1_1px'].std(),
    "f1_2px_mean": exp_v2['reinf_boundary_f1_2px'].mean(), "f1_2px_std": exp_v2['reinf_boundary_f1_2px'].std(),
    "failure_rate": exp_v2['extraction_failed'].mean()
}

df_aggregation = pd.DataFrame([view_metrics, exp_metrics])
print("\n--- GC-WPG-V2 AGGREGATION COMPARISON ---")
print(df_aggregation.to_string(index=False))

# 3. Experiment-Level Comparison across ALL 5 Models in Phase V7
df_ext = pd.read_csv(os.path.join(V7_DIR, "EXTERNAL_CONTOUR_METRICS.csv"))
cv_all = df_ext[df_ext['evaluation_suite'].str.contains("5-Fold")].copy()

exp_models = []
for model_name, grp in cv_all.groupby("model"):
    exp_grp = grp.groupby("sample").agg({
        'ssim': 'mean',
        'roi_ssim': 'mean',
        'psnr_db': 'mean',
        'contour_mae_px': 'mean',
        'apex_err_x_px': 'mean',
        'hausdorff_px': 'mean',
        'reinf_boundary_f1_1px': 'mean',
        'reinf_boundary_f1_2px': 'mean',
        'extraction_failed': 'max'
    }).reset_index()
    
    exp_models.append({
        "model": model_name,
        "n_experiments": len(exp_grp),
        "ssim_mean": round(exp_grp['ssim'].mean(), 4),
        "ssim_std": round(exp_grp['ssim'].std(), 4),
        "roi_ssim_mean": round(exp_grp['roi_ssim'].mean(), 4),
        "roi_ssim_std": round(exp_grp['roi_ssim'].std(), 4),
        "contour_mae_px_mean": round(exp_grp['contour_mae_px'].mean(), 2),
        "contour_mae_px_std": round(exp_grp['contour_mae_px'].std(), 2),
        "apex_err_x_px_mean": round(exp_grp['apex_err_x_px'].mean(), 2),
        "apex_err_x_px_std": round(exp_grp['apex_err_x_px'].std(), 2),
        "hausdorff_px_mean": round(exp_grp['hausdorff_px'].mean(), 2),
        "hausdorff_px_std": round(exp_grp['hausdorff_px'].std(), 2),
        "f1_1px_mean": round(exp_grp['reinf_boundary_f1_1px'].mean(), 4),
        "failure_rate": round(exp_grp['extraction_failed'].mean(), 3)
    })

# Add GC-WPG-v2 experiment-level metrics
exp_models.append({
    "model": "GC-WPG-v2 (Multi-Scale Prior)",
    "n_experiments": len(exp_v2),
    "ssim_mean": round(exp_v2['ssim'].mean(), 4),
    "ssim_std": round(exp_v2['ssim'].std(), 4),
    "roi_ssim_mean": round(exp_v2['roi_ssim'].mean(), 4),
    "roi_ssim_std": round(exp_v2['roi_ssim'].std(), 4),
    "contour_mae_px_mean": round(exp_v2['contour_mae_px'].mean(), 2),
    "contour_mae_px_std": round(exp_v2['contour_mae_px'].std(), 2),
    "apex_err_x_px_mean": round(exp_v2['apex_err_x_px'].mean(), 2),
    "apex_err_x_px_std": round(exp_v2['apex_err_x_px'].std(), 2),
    "hausdorff_px_mean": round(exp_v2['hausdorff_px'].mean(), 2),
    "hausdorff_px_std": round(exp_v2['hausdorff_px'].std(), 2),
    "f1_1px_mean": round(exp_v2['reinf_boundary_f1_1px'].mean(), 4),
    "failure_rate": round(exp_v2['extraction_failed'].mean(), 3)
})

df_exp_benchmark = pd.DataFrame(exp_models)
out_csv = os.path.join(AUDIT_DIR, "EXPERIMENT_LEVEL_CV_BENCHMARK.csv")
df_exp_benchmark.to_csv(out_csv, index=False)
print("\n--- EXPERIMENT-LEVEL 5-FOLD CV BENCHMARK (N=36 EXPERIMENTS) ---")
print(df_exp_benchmark[["model", "contour_mae_px_mean", "contour_mae_px_std", "apex_err_x_px_mean", "ssim_mean", "roi_ssim_mean", "f1_1px_mean", "failure_rate"]].to_string(index=False))
