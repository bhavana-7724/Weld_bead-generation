"""
run_leakage_free_regression.py

Executes rigorous, leakage-free nested cross-validation for weld bead geometry regression.
Addresses Phase V5 Objective A1:
- Outer 5-fold CV on 32 training experiments (excluding 4 reserved holdouts: 5, 14, 22, 32).
- Inner train/validation split (80/20) inside each outer training fold for neural network early stopping.
- Zero test-fold leakage during scaling, training, or checkpoint selection.
- Evaluates: Ridge, SVR, RandomForest, GradientBoosting, XGBoost, GaussianProcess, MLP, Proposed DNN, BNN-MCDO, and Baselines (Mean Predictor).
- Runs both Prospective (feed_rate, speed, material, process) and Retrospective (including heat input and efficiency) feature sets.
- Also performs diagnostic sensitivity analysis with and without Sample 32.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_dataset_36_v2.csv")
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "meeting_demo")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Categorical encoding
material_le = LabelEncoder()
process_le = LabelEncoder()
df["material_encoded"] = material_le.fit_transform(df["material"])
df["process_encoded"] = process_le.fit_transform(df["process"])

HOLDOUTS = [5, 14, 22, 32]
TARGET_COLS = ["width", "depth", "reinforcement"]

PROSP_FEATURES = ["feed_rate", "speed", "material_encoded", "process_encoded"]
RETRO_FEATURES = ["feed_rate", "speed", "HIP", "efficiency", "material_encoded", "process_encoded"]

# ---------------------------------------------------------
# Neural Network Architectures
# ---------------------------------------------------------
class ProposedDNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)

class BNNMCDropout(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)

def train_torch_model_nested(model_cls, in_features, X_inner_tr, y_inner_tr, X_inner_val, y_inner_val, epochs=400, lr=0.01, patience=30):
    """
    Trains PyTorch model with early stopping monitored STRICTLY on inner validation set.
    Outer test set is completely untouched.
    """
    model = model_cls(in_features)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    X_tr_t = torch.tensor(X_inner_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_inner_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_inner_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_inner_val, dtype=torch.float32)
    
    best_loss = float('inf')
    best_weights = None
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            v_out = model(X_val_t)
            v_loss = criterion(v_out, y_val_t).item()
            
        if v_loss < best_loss:
            best_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model

def predict_bnn_mc(model, X_test, n_forward_passes=100):
    """MC Dropout inference with dropout layers kept active."""
    model.train() # keeps dropout active
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(n_forward_passes):
            preds.append(model(X_test_t).numpy())
    preds = np.stack(preds, axis=0) # (100, N, 3)
    mean_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0)
    return mean_pred, std_pred

# ---------------------------------------------------------
# Evaluation Harness
# ---------------------------------------------------------
def get_model_dict():
    return {
        "Mean_Baseline": "mean_baseline",
        "Ridge": MultiOutputRegressor(Ridge(alpha=1.0)),
        "SVR": MultiOutputRegressor(SVR(C=10.0, epsilon=0.1)),
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=5, random_state=SEED)),
        "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=SEED)),
        "XGBoost": MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=SEED)),
        "GaussianProcess": MultiOutputRegressor(GaussianProcessRegressor(kernel=C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1), random_state=SEED, n_restarts_optimizer=2)),
        "MLP": MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=SEED, early_stopping=True, validation_fraction=0.15)),
        "Proposed_DNN": "torch_dnn",
        "BNN_MCDropout": "torch_bnn"
    }

def run_nested_cv(df_pool, feature_cols, cont_cols, n_splits=5, label="primary"):
    """
    Executes 5-fold CV with inner early stopping split.
    Returns detailed fold metrics and out-of-fold predictions.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    model_names = list(get_model_dict().keys())
    
    oof_predictions = {m: np.zeros((len(df_pool), 3)) for m in model_names}
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(df_pool)):
        df_outer_tr = df_pool.iloc[train_idx].copy().reset_index(drop=True)
        df_outer_te = df_pool.iloc[test_idx].copy().reset_index(drop=True)
        
        # Fit scaler on outer train only
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_tr = df_outer_tr[feature_cols].values.astype(np.float32)
        y_tr = df_outer_tr[TARGET_COLS].values.astype(np.float32)
        X_te = df_outer_te[feature_cols].values.astype(np.float32)
        y_te = df_outer_te[TARGET_COLS].values.astype(np.float32)
        
        # Continuous feature indices
        cont_indices = [feature_cols.index(c) for c in cont_cols]
        scaler_cont = StandardScaler()
        X_tr[:, cont_indices] = scaler_cont.fit_transform(X_tr[:, cont_indices])
        X_te[:, cont_indices] = scaler_cont.transform(X_te[:, cont_indices])
        
        y_tr_scaled = scaler_y.fit_transform(y_tr)
        
        # Inner train/val split for neural networks
        # 80% inner train, 20% inner val
        inner_tr_idx, inner_val_idx = train_test_split(np.arange(len(X_tr)), test_size=0.20, random_state=SEED + fold)
        X_inner_tr, y_inner_tr = X_tr[inner_tr_idx], y_tr_scaled[inner_tr_idx]
        X_inner_val, y_inner_val = X_tr[inner_val_idx], y_tr_scaled[inner_val_idx]
        
        models = get_model_dict()
        
        for m_name, model in models.items():
            if m_name == "Mean_Baseline":
                pred_scaled = np.tile(np.mean(y_tr_scaled, axis=0), (len(X_te), 1))
                pred = scaler_y.inverse_transform(pred_scaled)
            elif m_name == "Proposed_DNN":
                trained_net = train_torch_model_nested(
                    ProposedDNN, len(feature_cols), X_inner_tr, y_inner_tr, X_inner_val, y_inner_val,
                    epochs=400, lr=0.01, patience=30
                )
                trained_net.eval()
                with torch.no_grad():
                    pred_scaled = trained_net(torch.tensor(X_te, dtype=torch.float32)).numpy()
                pred = scaler_y.inverse_transform(pred_scaled)
            elif m_name == "BNN_MCDropout":
                trained_net = train_torch_model_nested(
                    BNNMCDropout, len(feature_cols), X_inner_tr, y_inner_tr, X_inner_val, y_inner_val,
                    epochs=400, lr=0.01, patience=30
                )
                pred_scaled, _ = predict_bnn_mc(trained_net, X_te, n_forward_passes=100)
                pred = scaler_y.inverse_transform(pred_scaled)
            else:
                model.fit(X_tr, y_tr_scaled)
                pred_scaled = model.predict(X_te)
                pred = scaler_y.inverse_transform(pred_scaled)
                
            oof_predictions[m_name][test_idx] = pred
            
            w_mae = mean_absolute_error(y_te[:, 0], pred[:, 0])
            d_mae = mean_absolute_error(y_te[:, 1], pred[:, 1])
            r_mae = mean_absolute_error(y_te[:, 2], pred[:, 2])
            avg_mae = (w_mae + d_mae + r_mae) / 3.0
            
            fold_metrics.append({
                "label": label,
                "model": m_name,
                "fold": fold + 1,
                "width_mae": w_mae,
                "depth_mae": d_mae,
                "reinf_mae": r_mae,
                "avg_mae": avg_mae
            })
            
    # Compute overall OOF metrics
    y_all = df_pool[TARGET_COLS].values
    summary = []
    for m_name in model_names:
        preds = oof_predictions[m_name]
        w_mae = mean_absolute_error(y_all[:, 0], preds[:, 0])
        d_mae = mean_absolute_error(y_all[:, 1], preds[:, 1])
        r_mae = mean_absolute_error(y_all[:, 2], preds[:, 2])
        avg_mae = (w_mae + d_mae + r_mae) / 3.0
        
        w_rmse = np.sqrt(mean_squared_error(y_all[:, 0], preds[:, 0]))
        d_rmse = np.sqrt(mean_squared_error(y_all[:, 1], preds[:, 1]))
        r_rmse = np.sqrt(mean_squared_error(y_all[:, 2], preds[:, 2]))
        avg_rmse = (w_rmse + d_rmse + r_rmse) / 3.0
        
        w_r2 = r2_score(y_all[:, 0], preds[:, 0])
        d_r2 = r2_score(y_all[:, 1], preds[:, 1])
        r_r2 = r2_score(y_all[:, 2], preds[:, 2])
        avg_r2 = (w_r2 + d_r2 + r_r2) / 3.0
        
        # Fold standard deviation
        f_sub = [f['avg_mae'] for f in fold_metrics if f['model'] == m_name and f['label'] == label]
        fold_std = np.std(f_sub)
        
        summary.append({
            "label": label,
            "model": m_name,
            "width_mae": w_mae,
            "depth_mae": d_mae,
            "reinf_mae": r_mae,
            "avg_mae": avg_mae,
            "avg_rmse": avg_rmse,
            "avg_r2": avg_r2,
            "fold_std_mae": fold_std
        })
        
    return pd.DataFrame(summary), pd.DataFrame(fold_metrics), oof_predictions

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("===============================================================")
    print("Phase V5 Objective A1: Leakage-Free Nested Cross-Validation")
    print("===============================================================")
    
    train_pool_32 = df[~df["sample"].isin(HOLDOUTS)].copy().reset_index(drop=True)
    holdouts_df = df[df["sample"].isin(HOLDOUTS)].copy().reset_index(drop=True)
    
    print(f"Primary Training Pool: {len(train_pool_32)} experiments.")
    print(f"Reserved Development Holdouts: {list(holdouts_df['sample'].values)}")
    
    # 1. Prospective Primary 5-Fold Nested CV (32 samples)
    print("\n--- 1. Prospective Nested 5-Fold CV (feed_rate, speed, material, process) ---")
    sum_prosp, fold_prosp, oof_prosp = run_nested_cv(
        train_pool_32, PROSP_FEATURES, cont_cols=["feed_rate", "speed"], n_splits=5, label="prospective_32"
    )
    print(sum_prosp.to_string(index=False))
    
    # 2. Retrospective Primary 5-Fold Nested CV (32 samples)
    print("\n--- 2. Retrospective Nested 5-Fold CV (with HIP, efficiency) ---")
    sum_retro, fold_retro, oof_retro = run_nested_cv(
        train_pool_32, RETRO_FEATURES, cont_cols=["feed_rate", "speed", "HIP", "efficiency"], n_splits=5, label="retrospective_32"
    )
    print(sum_retro.to_string(index=False))
    
    # 3. Diagnostic Investigation: Sensitivity with vs without Sample 32
    print("\n--- 3. Diagnostic Sensitivity: With vs Without Sample 32 ---")
    # Pool WITH Sample 32: 33 samples (excluding only holdouts 5, 14, 22)
    pool_with_32 = df[~df["sample"].isin([5, 14, 22])].copy().reset_index(drop=True)
    # Pool WITHOUT Sample 32: 32 samples (excluding holdouts 5, 14, 22, 32)
    pool_without_32 = df[~df["sample"].isin([5, 14, 22, 32])].copy().reset_index(drop=True)
    
    sum_with_32, _, _ = run_nested_cv(
        pool_with_32, PROSP_FEATURES, cont_cols=["feed_rate", "speed"], n_splits=5, label="prospective_with_32"
    )
    sum_without_32, _, _ = run_nested_cv(
        pool_without_32, PROSP_FEATURES, cont_cols=["feed_rate", "speed"], n_splits=5, label="prospective_without_32"
    )
    
    # Merge and compare
    comp_df = pd.merge(
        sum_with_32[["model", "width_mae", "depth_mae", "reinf_mae", "avg_mae"]],
        sum_without_32[["model", "width_mae", "depth_mae", "reinf_mae", "avg_mae"]],
        on="model", suffixes=("_with_32", "_without_32")
    )
    comp_df["depth_mae_delta"] = comp_df["depth_mae_without_32"] - comp_df["depth_mae_with_32"]
    comp_df["avg_mae_delta"] = comp_df["avg_mae_without_32"] - comp_df["avg_mae_with_32"]
    print("\nComparison With vs Without Sample 32 (Prospective):")
    print(comp_df[["model", "depth_mae_with_32", "depth_mae_without_32", "depth_mae_delta", "avg_mae_with_32", "avg_mae_without_32", "avg_mae_delta"]].to_string(index=False))
    
    # 4. Save results to CSV
    sum_prosp.to_csv(os.path.join(OUT_DIR, "regression_cv_leakage_free_prospective.csv"), index=False)
    sum_retro.to_csv(os.path.join(OUT_DIR, "regression_cv_leakage_free_retrospective.csv"), index=False)
    comp_df.to_csv(os.path.join(OUT_DIR, "sample32_sensitivity_comparison.csv"), index=False)
    
    # Save OOF predictions for downstream geometry-first validation
    oof_df = train_pool_32[["sample", "material", "process", "width", "depth", "reinforcement"]].copy()
    for m in oof_prosp:
        oof_df[f"{m}_width_oof"] = oof_prosp[m][:, 0]
        oof_df[f"{m}_depth_oof"] = oof_prosp[m][:, 1]
        oof_df[f"{m}_reinf_oof"] = oof_prosp[m][:, 2]
    oof_df.to_csv(os.path.join(OUT_DIR, "oof_predictions_prospective.csv"), index=False)
    
    print("\nSaved all leakage-free regression artifacts to:", OUT_DIR)
