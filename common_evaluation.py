import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp
import torch

# Global deterministic seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

tfd = tfp.distributions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_dataset_36.csv")
SAVE_DIR = os.path.join(BASE_DIR, "results", "evaluation")
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

material_encoder = LabelEncoder()
process_encoder = LabelEncoder()
df["material_encoded"] = material_encoder.fit_transform(df["material"])
df["process_encoded"] = process_encoder.fit_transform(df["process"])

INPUT_COLUMNS = ["feed_rate", "speed", "HIP", "efficiency", "material_encoded", "process_encoded"]
OUTPUT_COLUMNS = ["width", "depth", "reinforcement"]

X = df[INPUT_COLUMNS].values.astype(np.float32)
y = df[OUTPUT_COLUMNS].values.astype(np.float32)

N_TRAIN = int(len(X) * 0.8)
KL_WEIGHT = 1.0 / N_TRAIN

def scaled_kl_divergence(q, p, _):
    return tfp.distributions.kl_divergence(q, p) * KL_WEIGHT

def negative_log_likelihood(y_true, y_pred):
    distribution = tfd.Normal(loc=y_pred, scale=1.0)
    return -tf.reduce_mean(distribution.log_prob(y_true))

def build_mlp():
    model = tf_keras.Sequential([
        tf_keras.layers.Input(shape=(6,)),
        tf_keras.layers.Dense(16, activation="relu"),
        tf_keras.layers.Dense(8, activation="relu"),
        tf_keras.layers.Dense(3)
    ])
    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

def build_bnn():
    model = tf_keras.Sequential([
        tf_keras.layers.Input(shape=(6,)),
        tfp.layers.DenseFlipout(16, activation="relu", kernel_divergence_fn=scaled_kl_divergence, bias_divergence_fn=scaled_kl_divergence),
        tfp.layers.DenseFlipout(8, activation="relu", kernel_divergence_fn=scaled_kl_divergence, bias_divergence_fn=scaled_kl_divergence),
        tfp.layers.DenseFlipout(3, kernel_divergence_fn=scaled_kl_divergence, bias_divergence_fn=scaled_kl_divergence)
    ])
    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.001), loss=negative_log_likelihood, metrics=["mae"])
    return model

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "XGBoost": lambda: MultiOutputRegressor(XGBRegressor(random_state=42)),
    "GradientBoosting": lambda: MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
    "TabNet": lambda: TabNetRegressor(verbose=0, seed=42),
    "MLP": build_mlp,
    "BNN": build_bnn
}

results = []
bnn_uncertainty_results = []
fold = 1

for train_idx, test_idx in kf.split(X):
    print(f"\nEvaluating FOLD {fold}...")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    for model_name, model_fn in models.items():
        print(f"  Training {model_name}...")
        
        tf_keras.backend.clear_session()
        model = model_fn()
        
        if model_name in ["XGBoost", "GradientBoosting"]:
            model.fit(X_train_scaled, y_train_scaled)
            y_pred_scaled = model.predict(X_test_scaled)
            
        elif model_name == "TabNet":
            model.fit(
                X_train=X_train_scaled, y_train=y_train_scaled,
                eval_set=[(X_test_scaled, y_test_scaled)],
                eval_metric=['rmse'],
                max_epochs=200, patience=20, batch_size=8
            )
            y_pred_scaled = model.predict(X_test_scaled)
            
        elif model_name in ["MLP", "BNN"]:
            model.fit(X_train_scaled, y_train_scaled, epochs=500, batch_size=8, verbose=0)
            if model_name == "MLP":
                y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            else:
                y_pred_scaled = model(X_test_scaled, training=False).numpy()
                
                N_MC = 200
                mc_preds = []
                for _ in range(N_MC):
                    pred = model(X_test_scaled, training=True).numpy()
                    pred = y_scaler.inverse_transform(pred)
                    mc_preds.append(pred)
                mc_preds = np.array(mc_preds)
                mc_mean = np.mean(mc_preds, axis=0)
                mc_std = np.std(mc_preds, axis=0)
                
                for i in range(len(test_idx)):
                    sample_idx = test_idx[i]
                    for j, target_name in enumerate(OUTPUT_COLUMNS):
                        bnn_uncertainty_results.append({
                            "fold": fold,
                            "sample_idx": sample_idx,
                            "target": target_name,
                            "actual": y_test[i, j],
                            "mean_pred": mc_mean[i, j],
                            "std_dev": mc_std[i, j],
                            "lower_95": mc_mean[i, j] - 1.96 * mc_std[i, j],
                            "upper_95": mc_mean[i, j] + 1.96 * mc_std[i, j]
                        })

        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        for i, target in enumerate(OUTPUT_COLUMNS):
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            results.append({
                "model": model_name,
                "fold": fold,
                "target": target,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })
            
    fold += 1

results_df = pd.DataFrame(results)
summary_df = results_df.groupby(["model", "target"])[["MAE", "RMSE", "R2"]].mean().reset_index()

print("\n" + "=" * 60)
print("5-FOLD CV SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

results_df.to_csv(os.path.join(SAVE_DIR, "final_5fold_results.csv"), index=False)
summary_df.to_csv(os.path.join(SAVE_DIR, "final_5fold_summary.csv"), index=False)

if bnn_uncertainty_results:
    pd.DataFrame(bnn_uncertainty_results).to_csv(os.path.join(SAVE_DIR, "bnn_uncertainty_results.csv"), index=False)

print("\nSaved evaluation results to: ", SAVE_DIR)
