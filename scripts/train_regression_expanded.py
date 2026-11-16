import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
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

# -----------------------------------------------------------------------------
# Configuration and Reproducibility
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_dataset_36_v2.csv")
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# Publication plotting aesthetics
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# Encode categoricals
material_le = LabelEncoder()
process_le = LabelEncoder()
df["material_encoded"] = material_le.fit_transform(df["material"])
df["process_encoded"] = process_le.fit_transform(df["process"])

HOLDOUTS = [5, 14, 22, 32]
TARGET_COLS = ["width", "depth", "reinforcement"]

# Retrospective (6 features) vs Prospective (4 features)
RETRO_FEATURES = ["feed_rate", "speed", "HIP", "efficiency", "material_encoded", "process_encoded"]
PROSP_FEATURES = ["feed_rate", "speed", "material_encoded", "process_encoded"]

# Split train vs holdouts
train_df = df[~df["sample"].isin(HOLDOUTS)].copy().reset_index(drop=True)
test_df = df[df["sample"].isin(HOLDOUTS)].copy().reset_index(drop=True)

print(f"Training pool: {len(train_df)} samples, Holdout pool: {len(test_df)} samples")

# -----------------------------------------------------------------------------
# PyTorch Deep Neural Network (Proposed DNN Architecture)
# -----------------------------------------------------------------------------
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

def train_torch_model(model_cls, in_features, X_tr, y_tr, X_val, y_val, epochs=400, lr=0.01, patience=30):
    model = model_cls(in_features)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
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

# -----------------------------------------------------------------------------
# Cross-Validation Experiment Runner
# -----------------------------------------------------------------------------
def get_model_instances(feature_count):
    return {
        "Ridge": MultiOutputRegressor(Ridge(alpha=1.0)),
        "SVR": MultiOutputRegressor(SVR(C=10.0, epsilon=0.1)),
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=5, random_state=SEED)),
        "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=SEED)),
        "XGBoost": MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=SEED)),
        "GaussianProcess": MultiOutputRegressor(GaussianProcessRegressor(kernel=C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1), random_state=SEED, n_restarts_optimizer=2)),
        "MLP": MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=SEED, early_stopping=True, validation_fraction=0.15)),
        "Proposed_DNN": "torch_dnn",
        "BNN": "torch_bnn"
    }

def run_evaluation(feature_mode="retrospective"):
    feat_cols = RETRO_FEATURES if feature_mode == "retrospective" else PROSP_FEATURES
    cont_count = 4 if feature_mode == "retrospective" else 2
    
    X_train_full = train_df[feat_cols].values.astype(np.float32)
    y_train_full = train_df[TARGET_COLS].values.astype(np.float32)
    
    X_test = test_df[feat_cols].values.astype(np.float32)
    y_test = test_df[TARGET_COLS].values.astype(np.float32)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    models = get_model_instances(len(feat_cols))
    
    cv_records = []
    cv_predictions = []
    
    fold = 1
    for tr_idx, va_idx in kf.split(X_train_full):
        X_tr, X_va = X_train_full[tr_idx], X_train_full[va_idx]
        y_tr, y_va = y_train_full[tr_idx], y_train_full[va_idx]
        
        # Scale continuous features
        scaler_X = StandardScaler()
        X_tr_s = X_tr.copy()
        X_tr_s[:, :cont_count] = scaler_X.fit_transform(X_tr[:, :cont_count])
        X_va_s = X_va.copy()
        X_va_s[:, :cont_count] = scaler_X.transform(X_va[:, :cont_count])
        
        # Scale targets for neural nets
        scaler_y = StandardScaler()
        y_tr_s = scaler_y.fit_transform(y_tr)
        y_va_s = scaler_y.transform(y_va)
        
        for name, m_obj in models.items():
            if m_obj == "torch_dnn":
                model = train_torch_model(ProposedDNN, len(feat_cols), X_tr_s, y_tr_s, X_va_s, y_va_s, epochs=400, lr=0.01)
                model.eval()
                with torch.no_grad():
                    pred_s = model(torch.tensor(X_va_s, dtype=torch.float32)).numpy()
                y_pred = scaler_y.inverse_transform(pred_s)
            elif m_obj == "torch_bnn":
                model = train_torch_model(BNNMCDropout, len(feat_cols), X_tr_s, y_tr_s, X_va_s, y_va_s, epochs=400, lr=0.008)
                model.train() # MC Dropout active
                mc_preds = []
                with torch.no_grad():
                    t_in = torch.tensor(X_va_s, dtype=torch.float32)
                    for _ in range(50):
                        mc_preds.append(model(t_in).numpy())
                pred_s = np.mean(mc_preds, axis=0)
                y_pred = scaler_y.inverse_transform(pred_s)
            else:
                m = m_obj
                m.fit(X_tr_s, y_tr)
                y_pred = m.predict(X_va_s)
                
            # Log CV fold metrics
            w_mae = mean_absolute_error(y_va[:, 0], y_pred[:, 0])
            d_mae = mean_absolute_error(y_va[:, 1], y_pred[:, 1])
            r_mae = mean_absolute_error(y_va[:, 2], y_pred[:, 2])
            avg_mae = (w_mae + d_mae + r_mae) / 3.0
            
            w_rmse = np.sqrt(mean_squared_error(y_va[:, 0], y_pred[:, 0]))
            d_rmse = np.sqrt(mean_squared_error(y_va[:, 1], y_pred[:, 1]))
            r_rmse = np.sqrt(mean_squared_error(y_va[:, 2], y_pred[:, 2]))
            
            w_r2 = r2_score(y_va[:, 0], y_pred[:, 0])
            d_r2 = r2_score(y_va[:, 1], y_pred[:, 1])
            r_r2 = r2_score(y_va[:, 2], y_pred[:, 2])
            
            cv_records.append({
                "Mode": feature_mode,
                "Fold": fold,
                "Model": name,
                "Width_MAE": w_mae,
                "Depth_MAE": d_mae,
                "Reinforcement_MAE": r_mae,
                "Average_MAE": avg_mae,
                "Width_RMSE": w_rmse,
                "Depth_RMSE": d_rmse,
                "Reinforcement_RMSE": r_rmse,
                "Width_R2": w_r2,
                "Depth_R2": d_r2,
                "Reinforcement_R2": r_r2
            })
            
            # Save predictions for residual analysis
            val_samples = train_df.iloc[va_idx]["sample"].values
            for i, samp in enumerate(val_samples):
                cv_predictions.append({
                    "Mode": feature_mode,
                    "Fold": fold,
                    "Sample": samp,
                    "Model": name,
                    "Width_Actual": y_va[i, 0],
                    "Width_Pred": y_pred[i, 0],
                    "Depth_Actual": y_va[i, 1],
                    "Depth_Pred": y_pred[i, 1],
                    "Reinf_Actual": y_va[i, 2],
                    "Reinf_Pred": y_pred[i, 2]
                })
                
        fold += 1
        
    df_cv = pd.DataFrame(cv_records)
    df_preds = pd.DataFrame(cv_predictions)
    
    # -------------------------------------------------------------------------
    # Train Full Training Pool -> Evaluate on the 4 Holdout Samples
    # -------------------------------------------------------------------------
    scaler_X_full = StandardScaler()
    X_train_full_s = X_train_full.copy()
    X_train_full_s[:, :cont_count] = scaler_X_full.fit_transform(X_train_full[:, :cont_count])
    X_test_s = X_test.copy()
    X_test_s[:, :cont_count] = scaler_X_full.transform(X_test[:, :cont_count])
    
    scaler_y_full = StandardScaler()
    y_train_full_s = scaler_y_full.fit_transform(y_train_full)
    
    holdout_records = []
    
    # Validation split for neural network early stopping
    val_cut = int(0.15 * len(X_train_full_s))
    X_t_sub, X_v_sub = X_train_full_s[val_cut:], X_train_full_s[:val_cut]
    y_t_sub_s, y_v_sub_s = y_train_full_s[val_cut:], y_train_full_s[:val_cut]
    
    for name, m_obj in models.items():
        if m_obj == "torch_dnn":
            model = train_torch_model(ProposedDNN, len(feat_cols), X_t_sub, y_t_sub_s, X_v_sub, y_v_sub_s, epochs=400, lr=0.01)
            model.eval()
            with torch.no_grad():
                pred_s = model(torch.tensor(X_test_s, dtype=torch.float32)).numpy()
            y_pred = scaler_y_full.inverse_transform(pred_s)
        elif m_obj == "torch_bnn":
            model = train_torch_model(BNNMCDropout, len(feat_cols), X_t_sub, y_t_sub_s, X_v_sub, y_v_sub_s, epochs=400, lr=0.008)
            model.train()
            mc_preds = []
            with torch.no_grad():
                t_in = torch.tensor(X_test_s, dtype=torch.float32)
                for _ in range(50):
                    mc_preds.append(model(t_in).numpy())
            pred_s = np.mean(mc_preds, axis=0)
            y_pred = scaler_y_full.inverse_transform(pred_s)
        else:
            m = m_obj
            m.fit(X_train_full_s, y_train_full)
            y_pred = m.predict(X_test_s)
            
        for i, s in enumerate(HOLDOUTS):
            w_act, w_p = y_test[i, 0], y_pred[i, 0]
            d_act, d_p = y_test[i, 1], y_pred[i, 1]
            r_act, r_p = y_test[i, 2], y_pred[i, 2]
            
            holdout_records.append({
                "Mode": feature_mode,
                "Model": name,
                "Sample": s,
                "Alloy": test_df.iloc[i]["material"],
                "Process": test_df.iloc[i]["process"],
                "Width_Actual": w_act, "Width_Pred": w_p, "Width_Error": abs(w_act - w_p),
                "Depth_Actual": d_act, "Depth_Pred": d_p, "Depth_Error": abs(d_act - d_p),
                "Reinf_Actual": r_act, "Reinf_Pred": r_p, "Reinf_Error": abs(r_act - r_p),
                "Sample_Avg_MAE": (abs(w_act - w_p) + abs(d_act - d_p) + abs(r_act - r_p)) / 3.0
            })
            
    df_holdout = pd.DataFrame(holdout_records)
    return df_cv, df_preds, df_holdout

print("\n--- Running Retrospective Modeling (6 Features) ---")
cv_retro, preds_retro, holdouts_retro = run_evaluation("retrospective")

print("\n--- Running Prospective Modeling (4 Features) ---")
cv_prosp, preds_prosp, holdouts_prosp = run_evaluation("prospective")

# -----------------------------------------------------------------------------
# Aggregate and Export Data Tables
# -----------------------------------------------------------------------------
cv_all = pd.concat([cv_retro, cv_prosp], ignore_index=True)
preds_all = pd.concat([preds_retro, preds_prosp], ignore_index=True)
holdouts_all = pd.concat([holdouts_retro, holdouts_prosp], ignore_index=True)

preds_all.to_csv(os.path.join(OUT_DIR, "regression_predictions_cv.csv"), index=False)
holdouts_all.to_csv(os.path.join(OUT_DIR, "regression_holdouts_evaluation.csv"), index=False)

# CV Summary across folds
cv_summary = cv_all.groupby(["Mode", "Model"]).agg({
    "Width_MAE": ["mean", "std"],
    "Depth_MAE": ["mean", "std"],
    "Reinforcement_MAE": ["mean", "std"],
    "Average_MAE": ["mean", "std"],
    "Width_RMSE": ["mean"],
    "Depth_RMSE": ["mean"],
    "Reinforcement_RMSE": ["mean"],
    "Width_R2": ["mean"],
    "Depth_R2": ["mean"],
    "Reinforcement_R2": ["mean"]
}).reset_index()

cv_summary.columns = [
    "Mode", "Model",
    "Width_MAE_mean", "Width_MAE_std",
    "Depth_MAE_mean", "Depth_MAE_std",
    "Reinf_MAE_mean", "Reinf_MAE_std",
    "Avg_MAE_mean", "Avg_MAE_std",
    "Width_RMSE_mean", "Depth_RMSE_mean", "Reinf_RMSE_mean",
    "Width_R2_mean", "Depth_R2_mean", "Reinf_R2_mean"
]
cv_summary.to_csv(os.path.join(OUT_DIR, "regression_models_cv_comparison.csv"), index=False)

# Prospective vs Retrospective comparison table
pvt = cv_summary.pivot(index="Model", columns="Mode", values=["Width_MAE_mean", "Depth_MAE_mean", "Reinf_MAE_mean", "Avg_MAE_mean"])
pvt.to_csv(os.path.join(OUT_DIR, "regression_prospective_vs_retrospective.csv"))

print("\n=== 5-FOLD CV MODEL COMPARISON (RETROSPECTIVE) ===")
retro_summary = cv_summary[cv_summary["Mode"] == "retrospective"].sort_values("Avg_MAE_mean")
print(retro_summary[["Model", "Width_MAE_mean", "Depth_MAE_mean", "Reinf_MAE_mean", "Avg_MAE_mean"]].to_string(index=False))

print("\n=== 5-FOLD CV MODEL COMPARISON (PROSPECTIVE) ===")
prosp_summary = cv_summary[cv_summary["Mode"] == "prospective"].sort_values("Avg_MAE_mean")
print(prosp_summary[["Model", "Width_MAE_mean", "Depth_MAE_mean", "Reinf_MAE_mean", "Avg_MAE_mean"]].to_string(index=False))

# -----------------------------------------------------------------------------
# FIGURE 1: Dataset Overview & Physical Welding Correlations
# -----------------------------------------------------------------------------
print("\nGenerating Figure 1: Dataset Overview...")
fig1, axs = plt.subplots(2, 2, figsize=(14, 11))

# Subplot 1: Operational Space (Feed Rate vs Travel Speed, colored by Alloy & Shape by Process)
sns.scatterplot(
    data=df, x="speed", y="feed_rate", hue="material", style="process", s=120,
    palette={"IN718": "#1f77b4", "SS304L": "#ff7f0e"}, ax=axs[0, 0]
)
axs[0, 0].set_title("(a) Experimental Parameter Grid (Operational Space)", fontweight='bold')
axs[0, 0].set_xlabel("Travel Speed (cm/min)")
axs[0, 0].set_ylabel("Wire Feed Rate (m/min)")
axs[0, 0].grid(True, linestyle="--", alpha=0.5)

# Subplot 2: Heat Input vs Deposition Efficiency across Processes
sns.scatterplot(
    data=df, x="HIP", y="efficiency", hue="process", style="material", s=120,
    palette={"ColdArc": "#2ca02c", "GMAW": "#d62728"}, ax=axs[0, 1]
)
axs[0, 1].set_title("(b) Heat Input Parameter vs. Deposition Efficiency", fontweight='bold')
axs[0, 1].set_xlabel("Heat Input Parameter (J/mm)")
axs[0, 1].set_ylabel("Deposition Efficiency (%)")
axs[0, 1].grid(True, linestyle="--", alpha=0.5)

# Subplot 3: Target Geometry Distributions by Process
melt_targets = pd.melt(df, id_vars=["process"], value_vars=["width", "depth", "reinforcement"],
                       var_name="Dimension", value_name="Millimeters")
sns.boxplot(data=melt_targets, x="Dimension", y="Millimeters", hue="process",
            palette={"ColdArc": "#8c564b", "GMAW": "#e377c2"}, ax=axs[1, 0])
axs[1, 0].set_title("(c) Weld Bead Geometry Distributions by Process", fontweight='bold')
axs[1, 0].set_xlabel("Bead Dimension")
axs[1, 0].set_ylabel("Dimension (mm)")
axs[1, 0].grid(True, linestyle="--", alpha=0.5)

# Subplot 4: Pearson Correlation Heatmap
corr_cols = ["feed_rate", "speed", "HIP", "efficiency", "width", "depth", "reinforcement"]
corr_matrix = df[corr_cols].corr()
corr_labels = ["Feed Rate", "Speed", "HIP", "Efficiency", "Width", "Depth", "Reinf"]
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
            xticklabels=corr_labels, yticklabels=corr_labels, ax=axs[1, 1], vmin=-1, vmax=1)
axs[1, 1].set_title("(d) Inter-Variable Pearson Correlation Matrix", fontweight='bold')

plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, "fig1_dataset_overview.png")
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)
print(f"Saved: {fig1_path}")

# -----------------------------------------------------------------------------
# FIGURE 2: Regression Model Comparison (CV MAE & Prospective vs Retrospective)
# -----------------------------------------------------------------------------
print("\nGenerating Figure 2: Regression Model Comparison...")
fig2, axs = plt.subplots(2, 2, figsize=(15, 11))

# Subplot 1: Retrospective Model Comparison (Bar chart of Width, Depth, Reinf, Avg MAE)
models_order = ["Ridge", "SVR", "RandomForest", "GradientBoosting", "XGBoost", "GaussianProcess", "MLP", "BNN", "Proposed_DNN"]
df_retro_plt = cv_summary[cv_summary["Mode"] == "retrospective"].set_index("Model").loc[models_order].reset_index()

x = np.arange(len(models_order))
w = 0.20
axs[0, 0].bar(x - 1.5*w, df_retro_plt["Width_MAE_mean"], width=w, label="Width MAE", color="#1f77b4")
axs[0, 0].bar(x - 0.5*w, df_retro_plt["Depth_MAE_mean"], width=w, label="Depth MAE", color="#ff7f0e")
axs[0, 0].bar(x + 0.5*w, df_retro_plt["Reinf_MAE_mean"], width=w, label="Reinf MAE", color="#2ca02c")
axs[0, 0].bar(x + 1.5*w, df_retro_plt["Avg_MAE_mean"], width=w, label="Avg MAE", color="#d62728")
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(models_order, rotation=35, ha="right")
axs[0, 0].set_ylabel("5-Fold CV MAE (mm)")
axs[0, 0].set_title("(a) Retrospective Models (6 Inputs: F, S, HIP, Eff, Mat, Proc)", fontweight='bold')
axs[0, 0].legend()
axs[0, 0].grid(True, linestyle="--", alpha=0.4)

# Subplot 2: Prospective Model Comparison
df_prosp_plt = cv_summary[cv_summary["Mode"] == "prospective"].set_index("Model").loc[models_order].reset_index()
axs[0, 1].bar(x - 1.5*w, df_prosp_plt["Width_MAE_mean"], width=w, label="Width MAE", color="#1f77b4")
axs[0, 1].bar(x - 0.5*w, df_prosp_plt["Depth_MAE_mean"], width=w, label="Depth MAE", color="#ff7f0e")
axs[0, 1].bar(x + 0.5*w, df_prosp_plt["Reinf_MAE_mean"], width=w, label="Reinf MAE", color="#2ca02c")
axs[0, 1].bar(x + 1.5*w, df_prosp_plt["Avg_MAE_mean"], width=w, label="Avg MAE", color="#d62728")
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(models_order, rotation=35, ha="right")
axs[0, 1].set_ylabel("5-Fold CV MAE (mm)")
axs[0, 1].set_title("(b) Prospective Models (4 Operational Inputs: F, S, Mat, Proc)", fontweight='bold')
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle="--", alpha=0.4)

# Subplot 3: Prospective vs Retrospective Penalty (Avg MAE Delta)
delta_df = pd.DataFrame({
    "Model": models_order,
    "Retrospective": df_retro_plt["Avg_MAE_mean"].values,
    "Prospective": df_prosp_plt["Avg_MAE_mean"].values
})
delta_df["Delta"] = delta_df["Prospective"] - delta_df["Retrospective"]

w_d = 0.35
axs[1, 0].bar(x - w_d/2, delta_df["Retrospective"], width=w_d, label="Retrospective (6 Inputs)", color="#3470a3")
axs[1, 0].bar(x + w_d/2, delta_df["Prospective"], width=w_d, label="Prospective (4 Inputs)", color="#e07a5f")
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(models_order, rotation=35, ha="right")
axs[1, 0].set_ylabel("Average MAE (mm)")
axs[1, 0].set_title("(c) Operational Information Loss (Pre-Weld vs. Post-Weld)", fontweight='bold')
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle="--", alpha=0.4)

# Subplot 4: Feature Importance (Random Forest Gini Importance on Retrospective)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=SEED)
rf_model.fit(train_df[RETRO_FEATURES].values, train_df[TARGET_COLS].values)
feat_names = ["Feed Rate (F)", "Travel Speed (S)", "Heat Input (HIP)", "Dep. Efficiency", "Alloy Type", "Process Type"]

# Multi-output feature importances are computed per estimator
importances = np.mean([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
indices = np.argsort(importances)

axs[1, 1].barh(range(len(indices)), importances[indices], color="#2b8a3e", align="center")
axs[1, 1].set_yticks(range(len(indices)))
axs[1, 1].set_yticklabels([feat_names[i] for i in indices])
axs[1, 1].set_xlabel("Relative Feature Importance (MDI)")
axs[1, 1].set_title("(d) Feature Importance Ranking (Retrospective Random Forest)", fontweight='bold')
axs[1, 1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, "fig2_regression_comparison.png")
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)
print(f"Saved: {fig2_path}")

# -----------------------------------------------------------------------------
# FIGURE 3: Predicted vs Actual & Residual Analysis (Proposed DNN & BNN)
# -----------------------------------------------------------------------------
print("\nGenerating Figure 3: Predicted vs Actual Geometry & Residuals...")
fig3, axs = plt.subplots(2, 3, figsize=(16, 10))

# Filter Proposed_DNN predictions from 5-fold CV (Retrospective)
dnn_preds = preds_all[(preds_all["Model"] == "Proposed_DNN") & (preds_all["Mode"] == "retrospective")]

dims = [
    ("Width", "Width_Actual", "Width_Pred", 0),
    ("Depth", "Depth_Actual", "Depth_Pred", 1),
    ("Reinforcement", "Reinf_Actual", "Reinf_Pred", 2)
]

for label, act_col, pred_col, col_idx in dims:
    act = dnn_preds[act_col].values
    pred = dnn_preds[pred_col].values
    res = act - pred
    
    mae = mean_absolute_error(act, pred)
    rmse = np.sqrt(mean_squared_error(act, pred))
    r2 = r2_score(act, pred)
    
    # Top Row: Actual vs Predicted
    min_v = min(act.min(), pred.min()) * 0.95
    max_v = max(act.max(), pred.max()) * 1.05
    
    axs[0, col_idx].scatter(act, pred, color="#1f77b4", alpha=0.75, s=60, edgecolors="k", linewidth=0.5)
    axs[0, col_idx].plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5, label="Perfect Agreement (y = x)")
    
    # +/- 10% bounds
    axs[0, col_idx].plot([min_v, max_v], [min_v*1.1, max_v*1.1], 'k:', lw=1, alpha=0.6, label="±10% Margin")
    axs[0, col_idx].plot([min_v, max_v], [min_v*0.9, max_v*0.9], 'k:', lw=1, alpha=0.6)
    
    stat_box = f"MAE: {mae:.3f} mm\nRMSE: {rmse:.3f} mm\nR²: {r2:.3f}"
    axs[0, col_idx].text(0.05, 0.92, stat_box, transform=axs[0, col_idx].transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#ced4da'))
    
    axs[0, col_idx].set_title(f"(a{col_idx+1}) Proposed DNN: {label} (Actual vs Predicted)", fontweight='bold')
    axs[0, col_idx].set_xlabel(f"Actual {label} (mm)")
    axs[0, col_idx].set_ylabel(f"Predicted {label} (mm)")
    axs[0, col_idx].set_xlim(min_v, max_v)
    axs[0, col_idx].set_ylim(min_v, max_v)
    axs[0, col_idx].grid(True, linestyle="--", alpha=0.4)
    if col_idx == 0:
        axs[0, col_idx].legend(loc="lower right")

    # Bottom Row: Residuals vs Predicted
    axs[1, col_idx].scatter(pred, res, color="#e65100", alpha=0.75, s=60, edgecolors="k", linewidth=0.5)
    axs[1, col_idx].axhline(0, color="red", linestyle="--", lw=1.5)
    
    # 2-sigma error bounds
    std_res = np.std(res)
    axs[1, col_idx].axhline(2*std_res, color="gray", linestyle=":", lw=1, label="±2σ Error Bound")
    axs[1, col_idx].axhline(-2*std_res, color="gray", linestyle=":", lw=1)
    
    axs[1, col_idx].set_title(f"(b{col_idx+1}) Proposed DNN: {label} Residual Distribution", fontweight='bold')
    axs[1, col_idx].set_xlabel(f"Predicted {label} (mm)")
    axs[1, col_idx].set_ylabel(f"Residual (Actual - Predicted, mm)")
    axs[1, col_idx].grid(True, linestyle="--", alpha=0.4)
    if col_idx == 0:
        axs[1, col_idx].legend(loc="upper right")

plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, "fig3_predicted_vs_actual.png")
fig3.savefig(fig3_path, dpi=300)
plt.close(fig3)
print(f"Saved: {fig3_path}")

print("\n--- Regression Expansion Complete! ---")
