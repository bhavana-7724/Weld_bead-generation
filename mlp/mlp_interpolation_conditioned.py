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

print("=" * 70)
print("CONDITIONED MLP INTERPOLATION")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 3. CREATE MATERIAL + PROCESS FEATURES
# ============================================================

# One-hot encode material
df["material_IN718"] = (
    df["material"] == "IN718"
).astype(float)

df["material_SS304L"] = (
    df["material"] == "SS304L"
).astype(float)


# One-hot encode process
df["process_ColdArc"] = (
    df["process"] == "ColdArc"
).astype(float)

df["process_GMAW"] = (
    df["process"] == "GMAW"
).astype(float)


# ============================================================
# 4. INPUT / OUTPUT COLUMNS
# ============================================================

INPUT_COLUMNS = [
    "feed_rate",
    "speed",
    "HIP",
    "efficiency",
    "material_IN718",
    "material_SS304L",
    "process_ColdArc",
    "process_GMAW"
]

OUTPUT_COLUMNS = [
    "width",
    "depth",
    "reinforcement"
]


# ============================================================
# 5. HIDDEN INTERPOLATION POINTS
# ============================================================

HIDDEN_SAMPLES = [
    5,
    14,
    22,
    32
]


# ============================================================
# 6. SPLIT DATA
# ============================================================

test_df = df[
    df["sample"].isin(HIDDEN_SAMPLES)
].copy()

train_df = df[
    ~df["sample"].isin(HIDDEN_SAMPLES)
].copy()


print("\nHidden samples:", HIDDEN_SAMPLES)

print("Training samples:", len(train_df))
print("Hidden samples  :", len(test_df))


# ============================================================
# 7. SHOW HIDDEN POINTS
# ============================================================

print("\nHidden experimental points:")

print(
    test_df[
        [
            "sample",
            "material",
            "process",
            "feed_rate",
            "speed",
            "HIP",
            "efficiency",
            "width",
            "depth",
            "reinforcement"
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. PREPARE X AND Y
# ============================================================

X_train = train_df[INPUT_COLUMNS].values
X_test = test_df[INPUT_COLUMNS].values

y_train = train_df[OUTPUT_COLUMNS].values
y_test = test_df[OUTPUT_COLUMNS].values


# ============================================================
# 9. SCALE NUMERICAL INPUT FEATURES
# ============================================================

# IMPORTANT:
# We should NOT standardize one-hot categorical variables.
#
# Therefore only scale the first 4 numerical features.

X_scaler = StandardScaler()

X_train_numeric = X_scaler.fit_transform(
    X_train[:, :4]
)

X_test_numeric = X_scaler.transform(
    X_test[:, :4]
)


# Combine scaled numerical features
# with unchanged categorical features.

X_train_scaled = np.concatenate(
    [
        X_train_numeric,
        X_train[:, 4:]
    ],
    axis=1
)

X_test_scaled = np.concatenate(
    [
        X_test_numeric,
        X_test[:, 4:]
    ],
    axis=1
)


# ============================================================
# 10. SCALE OUTPUTS
# ============================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train
)

y_test_scaled = y_scaler.transform(
    y_test
)


# ============================================================
# 11. BUILD CONDITIONED MLP
# ============================================================

model = tf_keras.Sequential([

    tf_keras.layers.Input(
        shape=(8,)
    ),

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


print("\nModel input features:", len(INPUT_COLUMNS))

print("\nTraining conditioned MLP...")


# ============================================================
# 12. TRAIN
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
# 13. PREDICT HIDDEN EXPERIMENTS
# ============================================================

y_pred_scaled = model.predict(
    X_test_scaled,
    verbose=0
)

y_pred = y_scaler.inverse_transform(
    y_pred_scaled
)


# ============================================================
# 14. DISPLAY PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("CONDITIONED INTERPOLATION RESULTS")
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

    for j, column in enumerate(INPUT_COLUMNS[:4]):

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


    for j, output in enumerate(
        OUTPUT_COLUMNS
    ):

        actual = y_test[i, j]

        predicted = y_pred[i, j]

        error = abs(
            actual - predicted
        )


        print(
            f"  {output:15s} | "
            f"Actual: {actual:.3f} | "
            f"Predicted: {predicted:.3f} | "
            f"Error: {error:.3f}"
        )


        row[
            f"actual_{output}"
        ] = actual

        row[
            f"predicted_{output}"
        ] = predicted

        row[
            f"error_{output}"
        ] = error


    results.append(row)


# ============================================================
# 15. OVERALL METRICS
# ============================================================

print("\n" + "=" * 70)
print("CONDITIONED INTERPOLATION METRICS")
print("=" * 70)


for i, output in enumerate(
    OUTPUT_COLUMNS
):

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

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


output_path = (
    SAVE_DIR /
    "mlp_conditioned_interpolation_results.csv"
)


results_df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(output_path)