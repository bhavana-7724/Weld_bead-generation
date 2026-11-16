import os
import random
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
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
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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

KL_WEIGHT = 1.0 / 28.0 # Approx since 4/5 * 36 is ~28

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
# CV Loop
# ---------------------------------------------------------
df_cv = df.copy()
material_encoder = LabelEncoder()
process_encoder = LabelEncoder()
df_cv["material_encoded"] = material_encoder.fit_transform(df_cv["material"])
df_cv["process_encoded"] = process_encoder.fit_transform(df_cv["process"])

X = df_cv[INPUT_COLUMNS].values.astype(np.float32)
y = df_cv[OUTPUT_COLUMNS].values.astype(np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

models_dict = {
    "XGBoost": lambda: MultiOutputRegressor(XGBRegressor(random_state=SEED)),
    "GradientBoosting": lambda: MultiOutputRegressor(GradientBoostingRegressor(random_state=SEED)),
    "TabNet": lambda: TabNetRegressor(verbose=0, seed=SEED),
    "MLP": build_mlp,
    "BNN": build_bnn,
    "Proposed_DNN": get_proposed_dnn
}

fold_results = []
predictions = []
fold = 1

for train_idx, test_idx in kf.split(X):
    print(f"\\n--- Fold {fold} ---")
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    
    # Validation split for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=SEED
    )
    
    # Scale continuous features
    scaler_X = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[:, :4] = scaler_X.fit_transform(X_train[:, :4])
    X_val_scaled = X_val.copy()
    X_val_scaled[:, :4] = scaler_X.transform(X_val[:, :4])
    X_test_scaled = X_test.copy()
    X_test_scaled[:, :4] = scaler_X.transform(X_test[:, :4])
    
    # For full training sets (MLP, BNN, Tree models)
    scaler_X_full = StandardScaler()
    X_train_full_scaled = X_train_full.copy()
    X_train_full_scaled[:, :4] = scaler_X_full.fit_transform(X_train_full[:, :4])
    X_test_scaled_full = X_test.copy()
    X_test_scaled_full[:, :4] = scaler_X_full.transform(X_test[:, :4])
    
    # And for early stopping splits:
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full_scaled, y_train_full, test_size=0.15, random_state=SEED
    )
    
    scaler_y = StandardScaler()
    y_train_full_scaled = scaler_y.fit_transform(y_train_full)
    y_train_split_scaled = scaler_y.transform(y_train_split)
    y_val_split_scaled = scaler_y.transform(y_val_split)
    
    for model_name, model_fn in models_dict.items():
        print(f"Training {model_name}...")
        tf_keras.backend.clear_session()
        
        if model_name in ["XGBoost", "GradientBoosting"]:
            model = model_fn()
            model.fit(X_train_full, y_train_full) 
            y_pred = model.predict(X_test)
            
        elif model_name == "TabNet":
            model = model_fn()
            model.fit(
                X_train=X_train_split, y_train=y_train_split_scaled,
                eval_set=[(X_val_split, y_val_split_scaled)],
                eval_metric=['rmse'],
                max_epochs=200, patience=20, batch_size=8
            )
            y_pred_scaled = model.predict(X_test_scaled_full)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
        elif model_name == "MLP":
            model = model_fn()
            model.fit(X_train_full_scaled, y_train_full_scaled, epochs=500, batch_size=8, verbose=0)
            y_pred_scaled = model.predict(X_test_scaled_full, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
        elif model_name == "BNN":
            model = model_fn()
            model.fit(X_train_full_scaled, y_train_full_scaled, epochs=500, batch_size=8, verbose=0)
            N_MC = 200
            mc_preds = []
            for _ in range(N_MC):
                pred = model(X_test_scaled_full, training=True).numpy()
                pred = scaler_y.inverse_transform(pred)
                mc_preds.append(pred)
            mc_preds = np.array(mc_preds)
            y_pred = np.mean(mc_preds, axis=0)
            
        elif model_name == "Proposed_DNN":
            model = model_fn()
            early_stopping = tf_keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(
                X_train_split, y_train_split_scaled,
                validation_data=(X_val_split, y_val_split_scaled),
                epochs=500, batch_size=8, verbose=0,
                callbacks=[early_stopping]
            )
            y_pred_scaled = model.predict(X_test_scaled_full, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
        for i, target in enumerate(OUTPUT_COLUMNS):
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            fold_results.append({
                "fold": fold,
                "model": model_name,
                "target": target,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })
            
        samples_in_test = df_cv.iloc[test_idx]['sample'].values
        for i, s in enumerate(samples_in_test):
            predictions.append({
                "fold": fold,
                "sample": s,
                "model": model_name,
                "actual_width": y_test[i, 0],
                "predicted_width": y_pred[i, 0],
                "actual_depth": y_test[i, 1],
                "predicted_depth": y_pred[i, 1],
                "actual_reinforcement": y_test[i, 2],
                "predicted_reinforcement": y_pred[i, 2]
            })
            
    fold += 1

# ---------------------------------------------------------
# Outputs
# ---------------------------------------------------------
df_res = pd.DataFrame(fold_results)
df_preds = pd.DataFrame(predictions)
df_preds.to_csv(os.path.join(OUTPUT_DIR, "final_six_model_predictions.csv"), index=False)

mean_std = df_res.groupby(['model', 'target']).agg({'MAE': ['mean', 'std'], 'RMSE': ['mean', 'std'], 'R2': ['mean', 'std']}).reset_index()
mean_std.columns = ['Model', 'Target', 'MAE_mean', 'MAE_std', 'RMSE_mean', 'RMSE_std', 'R2_mean', 'R2_std']

table1_rows = []
models_list = ["XGBoost", "GradientBoosting", "TabNet", "MLP", "BNN", "Proposed_DNN"]
for m in models_list:
    row = {'Model': m}
    m_data = mean_std[mean_std['Model'] == m]
    avg_mae_sum = 0
    for t in OUTPUT_COLUMNS:
        t_data = m_data[m_data['Target'] == t].iloc[0]
        row[f'{t.capitalize()}_MAE_mean'] = t_data['MAE_mean']
        row[f'{t.capitalize()}_MAE_std'] = t_data['MAE_std']
        row[f'{t.capitalize()}_RMSE_mean'] = t_data['RMSE_mean']
        row[f'{t.capitalize()}_RMSE_std'] = t_data['RMSE_std']
        row[f'{t.capitalize()}_R2_mean'] = t_data['R2_mean']
        row[f'{t.capitalize()}_R2_std'] = t_data['R2_std']
        avg_mae_sum += t_data['MAE_mean']
    row['Average_MAE'] = avg_mae_sum / 3.0
    table1_rows.append(row)

df_table1 = pd.DataFrame(table1_rows)
df_table1.to_csv(os.path.join(OUTPUT_DIR, "final_six_model_comparison_mean_std.csv"), index=False)

df_table2 = df_table1[['Model', 'Width_MAE_mean', 'Depth_MAE_mean', 'Reinforcement_MAE_mean', 'Average_MAE']]
df_table2.columns = ['Model', 'Width', 'Depth', 'Reinforcement', 'Average']
df_table2.to_csv(os.path.join(OUTPUT_DIR, "final_six_model_comparison.csv"), index=False)

# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------
x = np.arange(len(models_list))
width = 0.12

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - width, df_table1['Width_MAE_mean'], width, label='Width MAE')
rects2 = ax.bar(x, df_table1['Depth_MAE_mean'], width, label='Depth MAE')
rects3 = ax.bar(x + width, df_table1['Reinforcement_MAE_mean'], width, label='Reinforcement MAE')
ax.set_ylabel('Mean CV MAE')
ax.set_title('Six-Model MAE Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'final_six_model_mae_comparison.png'))
plt.close()

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - width, df_table1['Width_RMSE_mean'], width, label='Width RMSE')
rects2 = ax.bar(x, df_table1['Depth_RMSE_mean'], width, label='Depth RMSE')
rects3 = ax.bar(x + width, df_table1['Reinforcement_RMSE_mean'], width, label='Reinforcement RMSE')
ax.set_ylabel('Mean CV RMSE')
ax.set_title('Six-Model RMSE Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'final_six_model_rmse_comparison.png'))
plt.close()

dnn_preds = df_preds[df_preds['model'] == 'Proposed_DNN']
for t in OUTPUT_COLUMNS:
    plt.figure(figsize=(6,6))
    plt.scatter(dnn_preds[f'actual_{t}'], dnn_preds[f'predicted_{t}'], alpha=0.7)
    min_val = min(dnn_preds[f'actual_{t}'].min(), dnn_preds[f'predicted_{t}'].min())
    max_val = max(dnn_preds[f'actual_{t}'].max(), dnn_preds[f'predicted_{t}'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    mae = mean_absolute_error(dnn_preds[f'actual_{t}'], dnn_preds[f'predicted_{t}'])
    rmse = np.sqrt(mean_squared_error(dnn_preds[f'actual_{t}'], dnn_preds[f'predicted_{t}']))
    r2 = r2_score(dnn_preds[f'actual_{t}'], dnn_preds[f'predicted_{t}'])
    n_samples = len(dnn_preds)
    
    textstr = f'n={n_samples}\\nMAE={mae:.4f}\\nRMSE={rmse:.4f}\\nR²={r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)
            
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Proposed DNN: {t.capitalize()} (Actual vs Predicted)')
    plt.savefig(os.path.join(PLOTS_DIR, f'proposed_dnn_{t}_actual_vs_predicted.png'))
    plt.close()

# Summary Output
print("\\n" + "="*60)
print("FINAL CONCISE SIX-MODEL TABLE")
print("="*60)
print(f"{'MODEL':<20} {'WIDTH MAE':<15} {'DEPTH MAE':<15} {'REINF MAE':<15} {'AVG MAE':<15}")
for idx, row in df_table2.iterrows():
    print(f"{row['Model']:<20} {row['Width']:<15.4f} {row['Depth']:<15.4f} {row['Reinforcement']:<15.4f} {row['Average']:<15.4f}")

best_width = df_table2.loc[df_table2['Width'].idxmin(), 'Model']
best_depth = df_table2.loc[df_table2['Depth'].idxmin(), 'Model']
best_reinf = df_table2.loc[df_table2['Reinforcement'].idxmin(), 'Model']
best_avg = df_table2.loc[df_table2['Average'].idxmin(), 'Model']

print("\\nBEST WIDTH MODEL:       ", best_width)
print("BEST DEPTH MODEL:       ", best_depth)
print("BEST REINFORCEMENT MODEL:", best_reinf)
print("BEST OVERALL MODEL:     ", best_avg)
