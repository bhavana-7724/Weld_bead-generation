import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
import tf_keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clean_dataset_36.csv"
)

SAVE_DIR = os.path.join(
    BASE_DIR,
    "mlp",
    "saved"
)

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 60)
print("DATASET")
print("=" * 60)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. DEFINE INPUTS AND OUTPUTS
# ============================================================

# Numerical inputs
NUMERIC_COLUMNS = [
    "feed_rate",
    "speed",
    "HIP",
    "efficiency"
]

# Categorical inputs
CATEGORICAL_COLUMNS = [
    "material",
    "process"
]

# Prediction targets
OUTPUT_COLUMNS = [
    "width",
    "depth",
    "reinforcement"
]


# ============================================================
# 4. ENCODE MATERIAL AND PROCESS
# ============================================================

material_encoder = LabelEncoder()
process_encoder = LabelEncoder()

df["material_encoded"] = material_encoder.fit_transform(
    df["material"]
)

df["process_encoded"] = process_encoder.fit_transform(
    df["process"]
)

print("\nMaterial encoding:")
for value, encoded in zip(
    material_encoder.classes_,
    range(len(material_encoder.classes_))
):
    print(f"  {value} -> {encoded}")

print("\nProcess encoding:")
for value, encoded in zip(
    process_encoder.classes_,
    range(len(process_encoder.classes_))
):
    print(f"  {value} -> {encoded}")


# ============================================================
# 5. CREATE INPUT MATRIX X
# ============================================================

INPUT_COLUMNS = [
    "feed_rate",
    "speed",
    "HIP",
    "efficiency",
    "material_encoded",
    "process_encoded"
]

X = df[INPUT_COLUMNS].values.astype(np.float32)

y = df[OUTPUT_COLUMNS].values.astype(np.float32)


print("\n" + "=" * 60)
print("INPUT / OUTPUT")
print("=" * 60)

print("Input columns:")
print(INPUT_COLUMNS)

print("\nOutput columns:")
print(OUTPUT_COLUMNS)

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=8,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 7. SCALE INPUTS AND OUTPUTS
# ============================================================

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(
    X_train
)

X_test_scaled = X_scaler.transform(
    X_test
)

y_train_scaled = y_scaler.fit_transform(
    y_train
)

y_test_scaled = y_scaler.transform(
    y_test
)


# ============================================================
# 8. BUILD MLP MODEL
# ============================================================

model = tf_keras.Sequential([

    tf_keras.layers.Input(
        shape=(6,)
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


# ============================================================
# 9. COMPILE
# ============================================================

model.compile(

    optimizer=tf_keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]
)


# ============================================================
# 10. MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)

model.summary()


# ============================================================
# 11. TRAIN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

history = model.fit(

    X_train_scaled,
    y_train_scaled,

    validation_data=(
        X_test_scaled,
        y_test_scaled
    ),

    epochs=500,

    batch_size=8,

    verbose=1
)


# ============================================================
# 12. PREDICT TEST DATA
# ============================================================

y_pred_scaled = model.predict(
    X_test_scaled,
    verbose=0
)

# Convert predictions back to original units
y_pred = y_scaler.inverse_transform(
    y_pred_scaled
)


# ============================================================
# 13. EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MLP TEST RESULTS")
print("=" * 60)

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

    print(f"\n{target}")

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
# 14. SHOW ACTUAL VS PREDICTED
# ============================================================

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED")
print("=" * 60)

for i in range(len(y_test)):

    print(f"\nTest sample {i + 1}")

    print(
        f"Width         : "
        f"Actual = {y_test[i, 0]:.3f}, "
        f"Predicted = {y_pred[i, 0]:.3f}"
    )

    print(
        f"Depth         : "
        f"Actual = {y_test[i, 1]:.3f}, "
        f"Predicted = {y_pred[i, 1]:.3f}"
    )

    print(
        f"Reinforcement : "
        f"Actual = {y_test[i, 2]:.3f}, "
        f"Predicted = {y_pred[i, 2]:.3f}"
    )


# ============================================================
# 15. SAVE MODEL
# ============================================================

model_path = os.path.join(
    SAVE_DIR,
    "mlp_weld_model.keras"
)

model.save(model_path)


# ============================================================
# 16. SAVE X SCALER
# ============================================================

with open(
    os.path.join(
        SAVE_DIR,
        "X_scaler.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        X_scaler,
        f
    )


# ============================================================
# 17. SAVE Y SCALER
# ============================================================

with open(
    os.path.join(
        SAVE_DIR,
        "y_scaler.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        y_scaler,
        f
    )


# ============================================================
# 18. SAVE MATERIAL ENCODER
# ============================================================

with open(
    os.path.join(
        SAVE_DIR,
        "material_encoder.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        material_encoder,
        f
    )


# ============================================================
# 19. SAVE PROCESS ENCODER
# ============================================================

with open(
    os.path.join(
        SAVE_DIR,
        "process_encoder.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        process_encoder,
        f
    )


# ============================================================
# 20. SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    SAVE_DIR,
    "mlp_training_history.csv"
)

pd.DataFrame(
    history.history
).to_csv(
    history_path,
    index=False
)


# ============================================================
# 21. SAVE TEST PREDICTIONS
# ============================================================

results = pd.DataFrame({

    "actual_width":
        y_test[:, 0],

    "predicted_width":
        y_pred[:, 0],

    "actual_depth":
        y_test[:, 1],

    "predicted_depth":
        y_pred[:, 1],

    "actual_reinforcement":
        y_test[:, 2],

    "predicted_reinforcement":
        y_pred[:, 2]
})

results.to_csv(
    os.path.join(
        SAVE_DIR,
        "mlp_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 22. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("\nSaved files:")

print(
    "Model:",
    model_path
)

print(
    "X scaler:",
    os.path.join(
        SAVE_DIR,
        "X_scaler.pkl"
    )
)

print(
    "Y scaler:",
    os.path.join(
        SAVE_DIR,
        "y_scaler.pkl"
    )
)

print(
    "Material encoder:",
    os.path.join(
        SAVE_DIR,
        "material_encoder.pkl"
    )
)

print(
    "Process encoder:",
    os.path.join(
        SAVE_DIR,
        "process_encoder.pkl"
    )
)

print(
    "Training history:",
    history_path
)

print("\nYou can now run:")
print("python mlp\\mlp_inference.py")