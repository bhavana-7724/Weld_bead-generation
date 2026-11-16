import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import torch

tfd = tfp.distributions

# ---------------------------------------------------------
# Set random seeds
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_dataset_36.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Data Loading & Verification
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
INPUT_COLUMNS = ["feed_rate", "speed", "HIP", "efficiency", "material_encoded", "process_encoded"]
OUTPUT_COLUMNS = ["width", "depth", "reinforcement"]

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
def get_proposed_dnn():
    model = tf_keras.Sequential([
        tf_keras.layers.Input(shape=(6,)),
        tf_keras.layers.Dense(128, activation='relu'),
        tf_keras.layers.Dropout(0.20),
        tf_keras.layers.Dense(64, activation='relu'),
        tf_keras.layers.Dropout(0.10),
        tf_keras.layers.Dense(32, activation='relu'),
        tf_keras.layers.Dense(3)
    ])
    model.compile(optimizer=tf_keras.optimizers.Adam(), loss='mse')
    return model

def build_mlp():
    model = tf_keras.Sequential([
        tf_keras.layers.Input(shape=(6,)),
        tf_keras.layers.Dense(16, activation="relu"),
        tf_keras.layers.Dense(8, activation="relu"),
        tf_keras.layers.Dense(3)
    ])
    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

# Train size for the BNN KL scaling. 
# We are training on 36 samples, so KL_WEIGHT = 1.0 / 36.0
KL_WEIGHT = 1.0 / len(df)

def scaled_kl_divergence(q, p, _):
    return tfp.distributions.kl_divergence(q, p) * KL_WEIGHT

def negative_log_likelihood(y_true, y_pred):
    distribution = tfd.Normal(loc=y_pred, scale=1.0)
    return -tf.reduce_mean(distribution.log_prob(y_true))

def build_bnn():
    model = tf_keras.Sequential([
        tf_keras.layers.Input(shape=(6,)),
        tfp.layers.DenseFlipout(16, activation="relu", kernel_divergence_fn=scaled_kl_divergence, bias_divergence_fn=scaled_kl_divergence),
        tfp.layers.DenseFlipout(8, activation="relu", kernel_divergence_fn=scaled_kl_divergence, bias_divergence_fn=scaled_kl_divergence),
        tfp.layers.DenseFlipout(3, kernel_divergence_fn=scaled_kl_divergence, bias_divergence_fn=scaled_kl_divergence)
    ])
    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.001), loss=negative_log_likelihood, metrics=["mae"])
    return model

# ---------------------------------------------------------
# Training Preparation
# ---------------------------------------------------------
df_train = df.copy()

material_encoder = LabelEncoder()
process_encoder = LabelEncoder()
df_train["material_encoded"] = material_encoder.fit_transform(df_train["material"])
df_train["process_encoded"] = process_encoder.fit_transform(df_train["process"])

X_train_full = df_train[INPUT_COLUMNS].values.astype(np.float32)
y_train_full = df_train[OUTPUT_COLUMNS].values.astype(np.float32)

# Full training sets (MLP, BNN, Tree models)
scaler_X_full = StandardScaler()
X_train_full_scaled = X_train_full.copy()
X_train_full_scaled[:, :4] = scaler_X_full.fit_transform(X_train_full[:, :4])

scaler_y = StandardScaler()
y_train_full_scaled = scaler_y.fit_transform(y_train_full)

# For early stopping splits (TabNet, Proposed DNN)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_full_scaled, y_train_full, test_size=0.15, random_state=SEED
)

y_train_split_scaled = scaler_y.transform(y_train_split)
y_val_split_scaled = scaler_y.transform(y_val_split)

models_dict = {
    "XGBoost": lambda: MultiOutputRegressor(XGBRegressor(random_state=SEED)),
    "GradientBoosting": lambda: MultiOutputRegressor(GradientBoostingRegressor(random_state=SEED)),
    "TabNet": lambda: TabNetRegressor(verbose=0, seed=SEED),
    "MLP": build_mlp,
    "BNN": build_bnn,
    "Proposed_DNN": get_proposed_dnn
}

# ---------------------------------------------------------
# Target Input Point
# ---------------------------------------------------------
# feed_rate = 5.0, speed = 25.0, HIP = 108.07, efficiency = 64.84, material = IN718, process = GMAW
mat_enc = material_encoder.transform(["IN718"])[0]
proc_enc = process_encoder.transform(["GMAW"])[0]

target_input_raw = np.array([[5.0, 25.0, 108.07, 64.84, mat_enc, proc_enc]], dtype=np.float32)

target_input_scaled = target_input_raw.copy()
target_input_scaled[:, :4] = scaler_X_full.transform(target_input_raw[:, :4])

predictions_output = []
bnn_uncertainty = None

# ---------------------------------------------------------
# Training & Inference
# ---------------------------------------------------------
print("Training models on full dataset (N=36)...")

for model_name, model_fn in models_dict.items():
    tf_keras.backend.clear_session()
    
    if model_name in ["XGBoost", "GradientBoosting"]:
        model = model_fn()
        model.fit(X_train_full, y_train_full) 
        y_pred = model.predict(target_input_raw)[0]
        
        predictions_output.append({
            "Model": model_name,
            "Predicted_Width": y_pred[0],
            "Predicted_Depth": y_pred[1],
            "Predicted_Reinforcement": y_pred[2],
            "Width_Std": np.nan,
            "Depth_Std": np.nan,
            "Reinforcement_Std": np.nan
        })
        
    elif model_name == "TabNet":
        model = model_fn()
        model.fit(
            X_train=X_train_split, y_train=y_train_split_scaled,
            eval_set=[(X_val_split, y_val_split_scaled)],
            eval_metric=['rmse'],
            max_epochs=200, patience=20, batch_size=8
        )
        y_pred_scaled = model.predict(target_input_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
        
        predictions_output.append({
            "Model": model_name,
            "Predicted_Width": y_pred[0],
            "Predicted_Depth": y_pred[1],
            "Predicted_Reinforcement": y_pred[2],
            "Width_Std": np.nan,
            "Depth_Std": np.nan,
            "Reinforcement_Std": np.nan
        })
        
    elif model_name == "MLP":
        model = model_fn()
        model.fit(X_train_full_scaled, y_train_full_scaled, epochs=500, batch_size=8, verbose=0)
        y_pred_scaled = model.predict(target_input_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
        
        predictions_output.append({
            "Model": model_name,
            "Predicted_Width": y_pred[0],
            "Predicted_Depth": y_pred[1],
            "Predicted_Reinforcement": y_pred[2],
            "Width_Std": np.nan,
            "Depth_Std": np.nan,
            "Reinforcement_Std": np.nan
        })
        
    elif model_name == "BNN":
        model = model_fn()
        model.fit(X_train_full_scaled, y_train_full_scaled, epochs=500, batch_size=8, verbose=0)
        N_MC = 200
        mc_preds = []
        for _ in range(N_MC):
            pred = model(target_input_scaled, training=True).numpy()
            pred = scaler_y.inverse_transform(pred)
            mc_preds.append(pred[0])
        mc_preds = np.array(mc_preds)
        y_pred_mean = np.mean(mc_preds, axis=0)
        y_pred_std = np.std(mc_preds, axis=0)
        
        predictions_output.append({
            "Model": model_name,
            "Predicted_Width": y_pred_mean[0],
            "Predicted_Depth": y_pred_mean[1],
            "Predicted_Reinforcement": y_pred_mean[2],
            "Width_Std": y_pred_std[0],
            "Depth_Std": y_pred_std[1],
            "Reinforcement_Std": y_pred_std[2]
        })
        bnn_uncertainty = y_pred_std
        
    elif model_name == "Proposed_DNN":
        model = model_fn()
        early_stopping = tf_keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model.fit(
            X_train_split, y_train_split_scaled,
            validation_data=(X_val_split, y_val_split_scaled),
            epochs=500, batch_size=8, verbose=0,
            callbacks=[early_stopping]
        )
        y_pred_scaled = model.predict(target_input_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
        
        predictions_output.append({
            "Model": model_name,
            "Predicted_Width": y_pred[0],
            "Predicted_Depth": y_pred[1],
            "Predicted_Reinforcement": y_pred[2],
            "Width_Std": np.nan,
            "Depth_Std": np.nan,
            "Reinforcement_Std": np.nan
        })

# ---------------------------------------------------------
# Outputs
# ---------------------------------------------------------
df_preds = pd.DataFrame(predictions_output)
df_preds.to_csv(os.path.join(OUTPUT_DIR, "single_input_predictions.csv"), index=False)

report_text = []
report_text.append("============================================================")
report_text.append("Example Prediction for Common Welding Condition")
report_text.append("============================================================")
report_text.append("Input Conditions:")
report_text.append("  feed_rate  = 5.0")
report_text.append("  speed      = 25.0")
report_text.append("  HIP        = 108.07")
report_text.append("  efficiency = 64.84")
report_text.append("  material   = IN718")
report_text.append("  process    = GMAW")
report_text.append("------------------------------------------------------------")
report_text.append(f"{'MODEL':<20} {'WIDTH (mm)':<15} {'DEPTH (mm)':<15} {'REINF (mm)':<15}")
report_text.append("-" * 60)

for idx, row in df_preds.iterrows():
    if row['Model'] == 'BNN':
        w_str = f"{row['Predicted_Width']:.4f} \u00B1{row['Width_Std']:.4f}"
        d_str = f"{row['Predicted_Depth']:.4f} \u00B1{row['Depth_Std']:.4f}"
        r_str = f"{row['Predicted_Reinforcement']:.4f} \u00B1{row['Reinforcement_Std']:.4f}"
    else:
        w_str = f"{row['Predicted_Width']:.4f}"
        d_str = f"{row['Predicted_Depth']:.4f}"
        r_str = f"{row['Predicted_Reinforcement']:.4f}"
    
    report_text.append(f"{row['Model']:<20} {w_str:<15} {d_str:<15} {r_str:<15}")

report_text.append("============================================================")
final_report = "\\n".join(report_text)

print(final_report)

with open(os.path.join(OUTPUT_DIR, "single_input_predictions.txt"), "w", encoding="utf-8") as f:
    f.write(final_report + "\\n")
