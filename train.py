"""
Train the weld-bead model.   Run:  python model/train.py
Does: load data -> (optional) fix data -> leave-one-out CV -> final fit -> save.
"""
import os, sys, pickle, warnings
import numpy as np 
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weld_model import WeldModel, TARGETS, build_features

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# DATA FIXES  -- VERIFY AGAINST YOUR LAB RECORDS, then edit these.
# ------------------------------------------------------------------
# Samples 6-9: depth and reinforcement look swapped in the source table.
SWAP_DEPTH_REINF_SAMPLES = [6, 7, 8, 9]
# Rows that look like typos/outliers. Re-measure or correct them, then
# remove them from this list. Set to [] to keep every row.
DROP_SAMPLES = [1, 13, 14, 20, 32]

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "C:/Users/cool/OneDrive/Desktop/Project/weld_project/data/clean_dataset_36.csv")
SAVE_DIR = os.path.join(BASE, "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

m = df["sample"].isin(SWAP_DEPTH_REINF_SAMPLES)
df.loc[m, ["depth", "reinforcement"]] = df.loc[m, ["reinforcement", "depth"]].values
print(f"Swapped depth/reinforcement for samples {SWAP_DEPTH_REINF_SAMPLES}")

df = df[~df["sample"].isin(DROP_SAMPLES)].reset_index(drop=True)
print(f"Dropped samples {DROP_SAMPLES} -> {len(df)} rows used\n")

X = build_features(df["feed_rate"], df["speed"], df["material"], df["process"])
Y = df[TARGETS].values

# ------------------------------------------------------------------
# Leave-one-out cross-validation = honest expected error on NEW settings
# ------------------------------------------------------------------
print("Running leave-one-out cross-validation (about a minute)...")
pred = np.zeros_like(Y)
for tr, te in LeaveOneOut().split(X):
    mdl = WeldModel().fit(X[tr], Y[tr])
    pred[te] = mdl.predict(X[te])[0]

print("\n" + "=" * 56)
print("LEAVE-ONE-OUT RESULTS (expected error on unseen settings)")
print("=" * 56)
print(f"{'target':15s}{'MAE (mm)':>10s}{'baseline*':>11s}{'R2':>8s}")
for j, t in enumerate(TARGETS):
    base = mean_absolute_error(Y[:, j], np.full(len(Y), Y[:, j].mean()))
    print(f"{t:15s}{mean_absolute_error(Y[:, j], pred[:, j]):10.3f}"
          f"{base:11.3f}{r2_score(Y[:, j], pred[:, j]):8.2f}")
print("*baseline = always predicting the average. Model MAE must be lower to be useful.")

# ------------------------------------------------------------------
# Final fit on ALL data and save
# ------------------------------------------------------------------
final = WeldModel().fit(X, Y)
ranges = {"feed_rate": (df.feed_rate.min(), df.feed_rate.max()),
          "speed": (df.speed.min(), df.speed.max())}
with open(os.path.join(SAVE_DIR, "weld_model.pkl"), "wb") as f:
    pickle.dump({"model": final, "ranges": ranges}, f)

pd.DataFrame({**{f"actual_{t}": Y[:, j] for j, t in enumerate(TARGETS)},
              **{f"loo_pred_{t}": pred[:, j] for j, t in enumerate(TARGETS)}}
             ).to_csv(os.path.join(SAVE_DIR, "loo_predictions.csv"), index=False)

print(f"\nSaved model -> {os.path.join(SAVE_DIR, 'weld_model.pkl')}")
print("Next: python model/predict.py")
