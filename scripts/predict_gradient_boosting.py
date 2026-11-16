import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

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

# ---------------------------------------------------------
# Data Loading & Verification
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
INPUT_COLUMNS = ["feed_rate", "speed", "HIP", "efficiency", "material_encoded", "process_encoded"]
OUTPUT_COLUMNS = ["width", "depth", "reinforcement"]

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
df_train = df.copy()

material_encoder = LabelEncoder()
process_encoder = LabelEncoder()
df_train["material_encoded"] = material_encoder.fit_transform(df_train["material"])
df_train["process_encoded"] = process_encoder.fit_transform(df_train["process"])

X_train_full = df_train[INPUT_COLUMNS].values.astype(np.float32)
y_train_full = df_train[OUTPUT_COLUMNS].values.astype(np.float32)

scaler_X_full = StandardScaler()
X_train_full_scaled = X_train_full.copy()
X_train_full_scaled[:, :4] = scaler_X_full.fit_transform(X_train_full[:, :4])

scaler_y = StandardScaler()
y_train_full_scaled = scaler_y.fit_transform(y_train_full)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_full_scaled, y_train_full, test_size=0.15, random_state=SEED
)
y_train_split_scaled = scaler_y.transform(y_train_split)
y_val_split_scaled = scaler_y.transform(y_val_split)

# ---------------------------------------------------------
# Target Input Point
# ---------------------------------------------------------
feed = 5.0
speed = 25.0
HIP = 108.07
efficiency = 64.84
material = "IN718"
process = "GMAW"

mat_enc = material_encoder.transform([material])[0]
proc_enc = process_encoder.transform([process])[0]

target_input_raw = np.array([[feed, speed, HIP, efficiency, mat_enc, proc_enc]], dtype=np.float32)

target_input_scaled = target_input_raw.copy()
target_input_scaled[:, :4] = scaler_X_full.transform(target_input_raw[:, :4])

# ---------------------------------------------------------
# Model Definition & Training
# ---------------------------------------------------------
tf_keras.backend.clear_session()

model = MultiOutputRegressor(GradientBoostingRegressor(random_state=SEED))
model.fit(X_train_full, y_train_full)
y_pred = model.predict(target_input_raw)[0]
model_name = 'Gradient Boosting'


# ---------------------------------------------------------
# Output Format
# ---------------------------------------------------------
print(f"WELD BEAD PREDICTION: {model_name}")
print(f"Input: \nFeed Rate:{feed}\nSpeed:{speed}\nHIP:{HIP}\nEfficiency={efficiency}")
print(f"Material={material}\nProcess={process}")
print(f"{'Width':14s} {y_pred[0]:6.2f} mm")
print(f"{'Depth':14s} {y_pred[1]:6.2f} mm")
print(f"{'Reinforcement':14s} {y_pred[2]:6.2f} mm")