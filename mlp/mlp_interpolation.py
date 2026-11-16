import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "clean_dataset_36.csv"

SAVE_DIR = Path(__file__).resolve().parent / "saved"
SAVE_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

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

# Center points of the four 3x3 experimental grids
HIDDEN_SAMPLES = [5, 14, 22, 32]


print("=" * 70)
print("MLP INTERPOLATION EXPERIMENT")
print("=" * 70)

print("\nHidden samples:", HIDDEN_SAMPLES)


# ============================================================
# 3. HIDE CENTER EXPERIMENTS
# ============================================================

test_df = df[df["sample"].isin(HIDDEN_SAMPLES)].copy()

train_df = df[~df["sample"].isin(HIDDEN_SAMPLES)].copy()

print("\nTraining samples:", len(train_df))
print("Hidden samples  :", len(test_df))


print("\nHidden experimental points:")

print(
    test_df[
        ["sample", "material", "process"] +
        INPUT_COLUMNS +
        OUTPUT_COLUMNS
    ].to_string(index=False)
)


# ============================================================
# 4. PREPARE DATA
# ============================================================

X_train = train_df[INPUT_COLUMNS].values
y_train = train_df[OUTPUT_COLUMNS].values

X_test = test_df[INPUT_COLUMNS].values
y_test = test_df[OUTPUT_COLUMNS].values


# ============================================================
# 5. SCALE INPUTS
# ============================================================

X_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# ============================================================
# 6. SCALE OUTPUTS
# ============================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# ============================================================
# 7. BUILD MLP
# ============================================================

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

    tf_keras.layers.Dense(3)
])


model.compile(
    optimizer=tf_keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)


print("\nTraining MLP...")


# ============================================================
# 8. TRAIN
# ============================================================

model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=500,
    batch_size=8,
    verbose=0
)


print("Training completed.")


# ============================================================
# 9. PREDICT HIDDEN CONDITIONS
# ============================================================

y_pred_scaled = model.predict(
    X_test_scaled,
    verbose=0
)

y_pred = y_scaler.inverse_transform(
    y_pred_scaled
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("INTERPOLATION RESULTS")
print("=" * 70)


results = []


for i in range(len(test_df)):

    sample_id = test_df.iloc[i]["sample"]

    material = test_df.iloc[i]["material"]

    process = test_df.iloc[i]["process"]

    print(f"\nSample {int(sample_id)}")
    print(f"Material : {material}")
    print(f"Process  : {process}")

    print("\nInput condition:")

    for j, column in enumerate(INPUT_COLUMNS):
        print(
            f"  {column:12s}: "
            f"{X_test[i, j]:.2f}"
        )

    print("\nWeld bead prediction:")

    row = {
        "sample": sample_id,
        "material": material,
        "process": process
    }

    for j, output in enumerate(OUTPUT_COLUMNS):

        actual = y_test[i, j]
        predicted = y_pred[i, j]
        error = abs(actual - predicted)

        print(
            f"  {output:15s} | "
            f"Actual: {actual:.3f} | "
            f"Predicted: {predicted:.3f} | "
            f"Error: {error:.3f}"
        )

        row[f"actual_{output}"] = actual
        row[f"predicted_{output}"] = predicted
        row[f"error_{output}"] = error

    results.append(row)


# ============================================================
# 11. OVERALL METRICS
# ============================================================

print("\n" + "=" * 70)
print("INTERPOLATION METRICS")
print("=" * 70)


for i, output in enumerate(OUTPUT_COLUMNS):

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

    print(f"\n{output.upper()}")

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_path = SAVE_DIR / "mlp_interpolation_results.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(output_path)