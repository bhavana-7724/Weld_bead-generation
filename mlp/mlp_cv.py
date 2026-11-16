import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras

from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "clean_dataset_36.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("MLP 5-FOLD CROSS-VALIDATION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


INPUT_COLUMNS = [
    "feed_rate",
    "speed",
    "HIP",
    "efficiency"
]

OUTPUT_COLUMNS = [
    "width",
    "depth",
    "reinforcement"
]

X = df[INPUT_COLUMNS].values
y = df[OUTPUT_COLUMNS].values

print(f"Input shape : {X.shape}")
print(f"Output shape: {y.shape}")


# ============================================================
# 3. MLP MODEL
# ============================================================

def build_mlp():

    model = tf_keras.Sequential([
        tf_keras.layers.Input(shape=(4,)),

        tf_keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf_keras.layers.Dense(
            8,
            activation="relu"
        ),

        tf_keras.layers.Dense(
            3
        )
    ])

    model.compile(
        optimizer=tf_keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# 4. 5-FOLD CROSS VALIDATION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Store metrics from every fold
all_results = []


for fold, (train_idx, test_idx) in enumerate(
    kf.split(X),
    start=1
):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"Training samples: {len(train_idx)}")
    print(f"Testing samples : {len(test_idx)}")


    # --------------------------------------------------------
    # Scale INPUTS
    # --------------------------------------------------------

    X_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(
        X_train
    )

    X_test_scaled = X_scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # Scale OUTPUTS
    # --------------------------------------------------------

    y_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(
        y_train
    )

    y_test_scaled = y_scaler.transform(
        y_test
    )


    # --------------------------------------------------------
    # Build fresh model
    # --------------------------------------------------------

    model = build_mlp()


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=500,
        batch_size=8,
        verbose=0
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred_scaled = model.predict(
        X_test_scaled,
        verbose=0
    )


    # --------------------------------------------------------
    # Convert predictions back to physical units
    # --------------------------------------------------------

    y_pred = y_scaler.inverse_transform(
        y_pred_scaled
    )


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    for i, target in enumerate(OUTPUT_COLUMNS):

        mae = mean_absolute_error(
            y_test[:, i],
            y_pred[:, i]
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test[:, i],
                y_pred[:, i]
            )
        )

        r2 = r2_score(
            y_test[:, i],
            y_pred[:, i]
        )

        all_results.append({
            "fold": fold,
            "target": target,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(
            f"{target:15s} | "
            f"MAE: {mae:.4f} | "
            f"RMSE: {rmse:.4f} | "
            f"R²: {r2:.4f}"
        )


# ============================================================
# 5. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(all_results)


# ============================================================
# 6. MEAN ± STD
# ============================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 60)


for target in OUTPUT_COLUMNS:

    target_results = results_df[
        results_df["target"] == target
    ]

    mae_mean = target_results["MAE"].mean()
    mae_std = target_results["MAE"].std()

    rmse_mean = target_results["RMSE"].mean()
    rmse_std = target_results["RMSE"].std()

    r2_mean = target_results["R2"].mean()
    r2_std = target_results["R2"].std()

    print(f"\n{target.upper()}")

    print(
        f"MAE  : {mae_mean:.4f} ± {mae_std:.4f}"
    )

    print(
        f"RMSE : {rmse_mean:.4f} ± {rmse_std:.4f}"
    )

    print(
        f"R²   : {r2_mean:.4f} ± {r2_std:.4f}"
    )


# ============================================================
# 7. SAVE RESULTS
# ============================================================

SAVE_DIR = Path(__file__).resolve().parent / "saved"

SAVE_DIR.mkdir(
    exist_ok=True
)

results_path = SAVE_DIR / "mlp_5fold_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 60)
print("RESULTS SAVED")
print("=" * 60)

print(results_path)