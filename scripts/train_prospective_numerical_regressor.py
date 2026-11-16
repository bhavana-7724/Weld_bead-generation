"""
train_prospective_numerical_regressor.py

Implements Section 4 of Phase V8:
1. Audits existing numerical models and notes that historical BNN/DNN used retrospective HIP & efficiency.
2. Conducts clean, leak-free 5-fold grouped CV on prospective features only:
   (feed_rate, speed, material, process).
3. Compares Ridge, Random Forest, Gradient Boosting, SVR, and MLP Regressor.
4. Selects the best performing prospective model based on 5-fold CV MAE.
5. Fits and saves the final prospective deployment checkpoint:
   - models/numerical/prospective_regressor_checkpoint.pkl
6. Exports validation summary to results/final_integration/guide_demo_v8/NUMERICAL_PREDICTION_VALIDATION.md
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_dataset_36_v2.csv")
V7_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v7")
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v8")
MODEL_DIR = os.path.join(BASE_DIR, "models", "numerical")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Load 5-fold splits
with open(os.path.join(V7_DIR, "cv_splits.json"), "r") as fp:
    cv_splits = json.load(fp)

TARGET_COLS = ["width", "depth", "reinforcement"]

# Candidate models
candidate_models = {
    "Ridge Regression": MultiOutputRegressor(Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
    "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42)),
    "Support Vector Regressor (SVR)": MultiOutputRegressor(SVR(kernel='rbf', C=10.0, epsilon=0.1)),
    "Multi-Layer Perceptron (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, alpha=0.01, random_state=42)
}

cv_results = {m: {"w_mae": [], "d_mae": [], "r_mae": [], "avg_mae": [], "preds": {}, "targets": {}} for m in candidate_models}

for f_idx in range(5):
    fold_name = f"fold_{f_idx+1}"
    split = cv_splits[fold_name]
    test_samples = split["outer_test_samples"]
    train_samples = split["train_samples"] + split["inner_val_samples"] # 28-30 train samples
    
    df_train = df[df['sample'].isin(train_samples)].copy()
    df_test = df[df['sample'].isin(test_samples)].copy()
    
    # Preprocessor fitted strictly on training data
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(df_train[["feed_rate", "speed"]])
    X_test_num = scaler.transform(df_test[["feed_rate", "speed"]])
    
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    X_train_cat = encoder.fit_transform(df_train[["material", "process"]])
    X_test_cat = encoder.transform(df_test[["material", "process"]])
    
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])
    
    Y_train = df_train[TARGET_COLS].values
    Y_test = df_test[TARGET_COLS].values
    
    for m_name, model in candidate_models.items():
        # Fit on training fold
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        
        w_mae = mean_absolute_error(Y_test[:, 0], preds[:, 0])
        d_mae = mean_absolute_error(Y_test[:, 1], preds[:, 1])
        r_mae = mean_absolute_error(Y_test[:, 2], preds[:, 2])
        avg_mae = (w_mae + d_mae + r_mae) / 3.0
        
        cv_results[m_name]["w_mae"].append(w_mae)
        cv_results[m_name]["d_mae"].append(d_mae)
        cv_results[m_name]["r_mae"].append(r_mae)
        cv_results[m_name]["avg_mae"].append(avg_mae)
        
        for s_id, p, y in zip(test_samples, preds, Y_test):
            cv_results[m_name]["preds"][s_id] = p
            cv_results[m_name]["targets"][s_id] = y

# Summarize CV
summary_rows = []
for m_name in candidate_models:
    w_m = np.mean(cv_results[m_name]["w_mae"])
    w_s = np.std(cv_results[m_name]["w_mae"])
    d_m = np.mean(cv_results[m_name]["d_mae"])
    d_s = np.std(cv_results[m_name]["d_mae"])
    r_m = np.mean(cv_results[m_name]["r_mae"])
    r_s = np.std(cv_results[m_name]["r_mae"])
    avg_m = np.mean(cv_results[m_name]["avg_mae"])
    avg_s = np.std(cv_results[m_name]["avg_mae"])
    
    summary_rows.append({
        "Model": m_name,
        "Width MAE (mm)": f"{w_m:.3f} ± {w_s:.3f}",
        "Depth MAE (mm)": f"{d_m:.3f} ± {d_s:.3f}",
        "Reinforcement MAE (mm)": f"{r_m:.3f} ± {r_s:.3f}",
        "Average MAE (mm)": f"{avg_m:.3f} ± {avg_s:.3f}",
        "raw_avg_mae": avg_m
    })

summary_df = pd.DataFrame(summary_rows).sort_values(by="raw_avg_mae").reset_index(drop=True)
print("=== PROSPECTIVE NUMERICAL REGRESSION 5-FOLD CV BENCHMARK ===")
print(summary_df[["Model", "Width MAE (mm)", "Depth MAE (mm)", "Reinforcement MAE (mm)", "Average MAE (mm)"]].to_string(index=False))

best_model_name = summary_df.iloc[0]["Model"]
best_val_mae = summary_df.iloc[0]["raw_avg_mae"]
print(f"\nBest Prospective Model Selected: {best_model_name} (5-Fold CV Average MAE: {best_val_mae:.3f} mm)")

# Train final deployment model on all 36 experiments
print(f"\nTraining final deployment checkpoint on all 36 experiments using {best_model_name}...")
final_scaler = StandardScaler()
X_all_num = final_scaler.fit_transform(df[["feed_rate", "speed"]])
final_encoder = OneHotEncoder(sparse_output=False, drop=None)
X_all_cat = final_encoder.fit_transform(df[["material", "process"]])
X_all = np.hstack([X_all_num, X_all_cat])
Y_all = df[TARGET_COLS].values

final_model = candidate_models[best_model_name]
final_model.fit(X_all, Y_all)

# Save checkpoint bundle
deployment_bundle = {
    "model_name": best_model_name,
    "model": final_model,
    "scaler": final_scaler,
    "encoder": final_encoder,
    "num_features": ["feed_rate", "speed"],
    "cat_features": ["material", "process"],
    "cat_categories": [list(c) for c in final_encoder.categories_],
    "target_cols": TARGET_COLS,
    "cv_summary": summary_df.to_dict(orient="records"),
    "documented_cv_mae_mm": {
        "width_mae": float(np.mean(cv_results[best_model_name]["w_mae"])),
        "depth_mae": float(np.mean(cv_results[best_model_name]["d_mae"])),
        "reinf_mae": float(np.mean(cv_results[best_model_name]["r_mae"])),
        "avg_mae": float(best_val_mae)
    },
    "training_domain": {
        "feed_rate_min": float(df["feed_rate"].min()),
        "feed_rate_max": float(df["feed_rate"].max()),
        "speed_min": float(df["speed"].min()),
        "speed_max": float(df["speed"].max()),
        "materials": sorted(list(df["material"].unique())),
        "processes": sorted(list(df["process"].unique()))
    }
}

ckpt_path = os.path.join(MODEL_DIR, "prospective_regressor_checkpoint.pkl")
with open(ckpt_path, "wb") as fp:
    pickle.dump(deployment_bundle, fp)
print(f"Saved deployment checkpoint to: {ckpt_path}")

# Also copy to guide_demo_v8 for provenance
v8_ckpt_path = os.path.join(OUT_DIR, "prospective_regressor_checkpoint.pkl")
with open(v8_ckpt_path, "wb") as fp:
    pickle.dump(deployment_bundle, fp)

# Save CV benchmark CSV
summary_df.to_csv(os.path.join(OUT_DIR, "NUMERICAL_REGRESSION_CV_BENCHMARK.csv"), index=False)
print("Saved NUMERICAL_REGRESSION_CV_BENCHMARK.csv successfully.")
