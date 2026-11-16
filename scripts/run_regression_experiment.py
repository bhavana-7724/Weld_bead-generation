import os
import sys
import random
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Set random seeds
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
import torch
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_dataset_36.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values:\n", df.isnull().sum())
print("Material distribution:\n", df['material'].value_counts())
print("Process distribution:\n", df['process'].value_counts())
print("Target Statistics:\n", df[['width', 'depth', 'reinforcement']].describe())

# Verify known values
assert np.isclose(df.loc[df['sample'] == 27, 'efficiency'].values[0], 60.35)
assert np.isclose(df.loc[df['sample'] == 30, 'efficiency'].values[0], 95.80)
assert np.isclose(df.loc[df['sample'] == 32, 'reinforcement'].values[0], 2.74)

# ---------------------------------------------------------
# Preprocessing Definitions
# ---------------------------------------------------------
INPUT_COLUMNS = ["feed_rate", "speed", "HIP", "efficiency", "material_encoded", "process_encoded"]
OUTPUT_COLUMNS = ["width", "depth", "reinforcement"]
HOLDOUT_SAMPLES = [5, 14, 22, 32]

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
def get_proposed_dnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
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
    # Continuous are indices 0, 1, 2, 3
    scaler_X = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[:, :4] = scaler_X.fit_transform(X_train[:, :4])
    
    X_val_scaled = X_val.copy()
    X_val_scaled[:, :4] = scaler_X.transform(X_val[:, :4])
    
    X_test_scaled = X_test.copy()
    X_test_scaled[:, :4] = scaler_X.transform(X_test[:, :4])
    
    # Target scaling for DNN / TabNet
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    
    for model_name, model_fn in models_dict.items():
        print(f"Training {model_name}...")
        tf.keras.backend.clear_session()
        
        if model_name in ["XGBoost", "GradientBoosting"]:
            model = model_fn()
            # Tree models don't strictly need scaling, but using scaled targets is fine
            # Actually, let's use unscaled targets for tree models to be simpler, 
            # or scaled. Let's use unscaled targets and unscaled X except we pass the scaled X for consistency
            # wait, the prompt says: "For tree-based models: scaling is not required for the continuous variables... use the same transformed inputs for every fold"
            # We'll use unscaled targets for trees, and unscaled X
            model.fit(X_train_full, y_train_full)
            y_pred = model.predict(X_test)
        
        elif model_name == "TabNet":
            model = model_fn()
            # TabNet needs a validation set for early stopping
            model.fit(
                X_train=X_train_scaled, y_train=y_train_scaled,
                eval_set=[(X_val_scaled, y_val_scaled)],
                eval_metric=['rmse'],
                max_epochs=200, patience=20, batch_size=8
            )
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
        elif model_name == "Proposed_DNN":
            model = model_fn()
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            # Use scaled X and scaled Y
            model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=500, batch_size=8, verbose=0,
                callbacks=[early_stopping]
            )
            y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
        # Metrics calculation
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
            
        # Store predictions
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
# Generate outputs
# ---------------------------------------------------------
df_res = pd.DataFrame(fold_results)
df_res.to_csv(os.path.join(OUTPUT_DIR, "fold_metrics.csv"), index=False)

df_preds = pd.DataFrame(predictions)
df_preds.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

# Summary Tables
mean_std = df_res.groupby(['model', 'target']).agg({'MAE': ['mean', 'std'], 'RMSE': ['mean', 'std'], 'R2': ['mean', 'std']}).reset_index()
mean_std.columns = ['Model', 'Target', 'MAE_mean', 'MAE_std', 'RMSE_mean', 'RMSE_std', 'R2_mean', 'R2_std']

# Pivot for Table 1
table1_rows = []
models_list = ["XGBoost", "GradientBoosting", "TabNet", "Proposed_DNN"]
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
df_table1.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_mean_std.csv"), index=False)

df_table2 = df_table1[['Model', 'Width_MAE_mean', 'Depth_MAE_mean', 'Reinforcement_MAE_mean', 'Average_MAE']]
df_table2.columns = ['Model', 'Width', 'Depth', 'Reinforcement', 'Average']
df_table2.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_mae.csv"), index=False)

# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------
dnn_preds = df_preds[df_preds['model'] == 'Proposed_DNN']
for t in OUTPUT_COLUMNS:
    # Actual vs Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(dnn_preds[f'actual_{t}'], dnn_preds[f'predicted_{t}'], alpha=0.7)
    min_val = min(dnn_preds[f'actual_{t}'].min(), dnn_preds[f'predicted_{t}'].min())
    max_val = max(dnn_preds[f'actual_{t}'].max(), dnn_preds[f'predicted_{t}'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate metrics for the plot
    mae = mean_absolute_error(dnn_preds[f'actual_{t}'], dnn_preds[f'predicted_{t}'])
    rmse = np.sqrt(mean_squared_error(dnn_preds[f'actual_{t}'], dnn_preds[f'predicted_{t}']))
    r2 = r2_score(dnn_preds[f'actual_{t}'], dnn_preds[f'predicted_{t}'])
    n_samples = len(dnn_preds)
    
    textstr = f'n={n_samples}\\nMAE={mae:.4f}\\nRMSE={rmse:.4f}\\nR²={r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Proposed DNN: {t.capitalize()} (Actual vs Predicted)')
    plt.savefig(os.path.join(PLOTS_DIR, f'proposed_dnn_{t}_actual_vs_predicted.png'))
    plt.close()
    
    # Residual Plot
    plt.figure(figsize=(6,4))
    residuals = dnn_preds[f'actual_{t}'] - dnn_preds[f'predicted_{t}']
    plt.scatter(dnn_preds[f'predicted_{t}'], residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'Proposed DNN: {t.capitalize()} Residuals')
    plt.savefig(os.path.join(PLOTS_DIR, f'proposed_dnn_{t}_residuals.png'))
    plt.close()

# Model Comparison Graph
x = np.arange(len(models_list))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, df_table1['Width_MAE_mean'], width, label='Width MAE')
rects2 = ax.bar(x, df_table1['Depth_MAE_mean'], width, label='Depth MAE')
rects3 = ax.bar(x + width, df_table1['Reinforcement_MAE_mean'], width, label='Reinforcement MAE')

ax.set_ylabel('Mean CV MAE')
ax.set_title('Model MAE Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_mae_comparison.png'))
plt.close()

# RMSE Comparison Graph
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, df_table1['Width_RMSE_mean'], width, label='Width RMSE')
rects2 = ax.bar(x, df_table1['Depth_RMSE_mean'], width, label='Depth RMSE')
rects3 = ax.bar(x + width, df_table1['Reinforcement_RMSE_mean'], width, label='Reinforcement RMSE')

ax.set_ylabel('Mean CV RMSE')
ax.set_title('Model RMSE Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_rmse_comparison.png'))
plt.close()

# ---------------------------------------------------------
# Interpolation Holdout Evaluation
# ---------------------------------------------------------
print("\\n--- Interpolation Holdout Evaluation ---")
df_interp_train = df[~df['sample'].isin(HOLDOUT_SAMPLES)].copy()
df_interp_test = df[df['sample'].isin(HOLDOUT_SAMPLES)].copy()

df_interp_train["material_encoded"] = material_encoder.transform(df_interp_train["material"])
df_interp_train["process_encoded"] = process_encoder.transform(df_interp_train["process"])
df_interp_test["material_encoded"] = material_encoder.transform(df_interp_test["material"])
df_interp_test["process_encoded"] = process_encoder.transform(df_interp_test["process"])

X_interp_train_full = df_interp_train[INPUT_COLUMNS].values.astype(np.float32)
y_interp_train_full = df_interp_train[OUTPUT_COLUMNS].values.astype(np.float32)
X_interp_test = df_interp_test[INPUT_COLUMNS].values.astype(np.float32)
y_interp_test = df_interp_test[OUTPUT_COLUMNS].values.astype(np.float32)

X_interp_train, X_interp_val, y_interp_train, y_interp_val = train_test_split(
    X_interp_train_full, y_interp_train_full, test_size=0.15, random_state=SEED
)

scaler_X = StandardScaler()
X_interp_train_scaled = X_interp_train.copy()
X_interp_train_scaled[:, :4] = scaler_X.fit_transform(X_interp_train[:, :4])
X_interp_val_scaled = X_interp_val.copy()
X_interp_val_scaled[:, :4] = scaler_X.transform(X_interp_val[:, :4])
X_interp_test_scaled = X_interp_test.copy()
X_interp_test_scaled[:, :4] = scaler_X.transform(X_interp_test[:, :4])

scaler_y = StandardScaler()
y_interp_train_scaled = scaler_y.fit_transform(y_interp_train)
y_interp_val_scaled = scaler_y.transform(y_interp_val)
y_interp_test_scaled = scaler_y.transform(y_interp_test)

interp_results = []

for model_name, model_fn in models_dict.items():
    tf.keras.backend.clear_session()
    
    if model_name in ["XGBoost", "GradientBoosting"]:
        model = model_fn()
        model.fit(X_interp_train_full, y_interp_train_full)
        y_pred = model.predict(X_interp_test)
        
    elif model_name == "TabNet":
        model = model_fn()
        model.fit(
            X_train=X_interp_train_scaled, y_train=y_interp_train_scaled,
            eval_set=[(X_interp_val_scaled, y_interp_val_scaled)],
            eval_metric=['rmse'],
            max_epochs=200, patience=20, batch_size=8
        )
        y_pred_scaled = model.predict(X_interp_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
    elif model_name == "Proposed_DNN":
        model = model_fn()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model.fit(
            X_interp_train_scaled, y_interp_train_scaled,
            validation_data=(X_interp_val_scaled, y_interp_val_scaled),
            epochs=500, batch_size=8, verbose=0, callbacks=[early_stopping]
        )
        y_pred_scaled = model.predict(X_interp_test_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
    for i, target in enumerate(OUTPUT_COLUMNS):
        mae = mean_absolute_error(y_interp_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_interp_test[:, i], y_pred[:, i]))
        r2 = r2_score(y_interp_test[:, i], y_pred[:, i])
        
        interp_results.append({
            "Model": model_name,
            "Target": target,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

df_interp = pd.DataFrame(interp_results)
df_interp.to_csv(os.path.join(OUTPUT_DIR, "interpolation_holdout_results.csv"), index=False)

# ---------------------------------------------------------
# Interpolation Graph
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
interp_mae_w = df_interp[df_interp['Target'] == 'width']['MAE'].values
interp_mae_d = df_interp[df_interp['Target'] == 'depth']['MAE'].values
interp_mae_r = df_interp[df_interp['Target'] == 'reinforcement']['MAE'].values

rects1 = ax.bar(x - width, interp_mae_w, width, label='Width MAE')
rects2 = ax.bar(x, interp_mae_d, width, label='Depth MAE')
rects3 = ax.bar(x + width, interp_mae_r, width, label='Reinforcement MAE')

ax.set_ylabel('Interpolation Holdout MAE')
ax.set_title('Interpolation Holdout Evaluation (MAE)')
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'interpolation_model_comparison.png'))
plt.close()

# ---------------------------------------------------------
# Six-Model Table & Final Output
# ---------------------------------------------------------
historical_mlp = {
    'Model': 'MLP',
    'Width_MAE_mean': 1.0705, 'Width_MAE_std': 0.2576,
    'Depth_MAE_mean': 0.7444, 'Depth_MAE_std': 0.2488,
    'Reinforcement_MAE_mean': 0.7591, 'Reinforcement_MAE_std': 0.1926
}
historical_bnn = {
    'Model': 'BNN',
    'Width_MAE_mean': 0.9808, 'Width_MAE_std': 0.6495,
    'Depth_MAE_mean': 0.7980, 'Depth_MAE_std': 0.1435,
    'Reinforcement_MAE_mean': 0.7374, 'Reinforcement_MAE_std': 0.3347
}
historical_df = pd.DataFrame([historical_mlp, historical_bnn])

# Combine with Table1 (selecting relevant columns)
cols = ['Model', 'Width_MAE_mean', 'Width_MAE_std', 'Depth_MAE_mean', 'Depth_MAE_std', 'Reinforcement_MAE_mean', 'Reinforcement_MAE_std']
df_six = pd.concat([df_table1[cols], historical_df[cols]], ignore_index=True)
df_six['Status'] = df_six['Model'].apply(lambda x: 'Historical' if x in ['MLP', 'BNN'] else 'Newly Verified')
df_six.to_csv(os.path.join(OUTPUT_DIR, "final_six_model_comparison.csv"), index=False)

# Config JSON
config = {
    "dataset_path": DATA_PATH,
    "seed": SEED,
    "n_folds": 5,
    "input_features": INPUT_COLUMNS,
    "categorical_features": ["material", "process"],
    "targets": OUTPUT_COLUMNS,
    "model_configurations": {
        "XGBoost": "MultiOutputRegressor(XGBRegressor(random_state=42))",
        "GradientBoosting": "MultiOutputRegressor(GradientBoostingRegressor(random_state=42))",
        "TabNet": "TabNetRegressor(verbose=0, seed=42) - max_epochs=200, patience=20",
        "Proposed_DNN": "Dense(128)->Dropout(0.2)->Dense(64)->Dropout(0.1)->Dense(32)->Dense(3), Adam, MSE"
    },
    "interpolation_holdout_samples": HOLDOUT_SAMPLES
}
with open(os.path.join(OUTPUT_DIR, "experiment_config.json"), "w") as f:
    json.dump(config, f, indent=4)

# Print Final Summary
print("\\n" + "="*60)
print("FINAL CONCISE TABLE")
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

print("\\nFiles generated in", OUTPUT_DIR, ":")
for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in files:
        print(os.path.join(root, f))
