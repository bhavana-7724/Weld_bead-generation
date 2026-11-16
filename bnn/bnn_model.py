import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


tfd = tfp.distributions


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
    "bnn",
    "saved"
)

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 60)
print("DATASET")
print("=" * 60)

print("Dataset shape:", df.shape)


# ============================================================
# 3. INPUT / OUTPUT COLUMNS
# ============================================================

INPUT_COLUMNS = [
    "feed_rate",
    "speed",
    "HIP",
    "efficiency",
    "material_encoded",
    "process_encoded"
]

OUTPUT_COLUMNS = [
    "width",
    "depth",
    "reinforcement"
]


# ============================================================
# 4. ENCODE CATEGORICAL FEATURES
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
# 5. CREATE X AND Y
# ============================================================

X = df[INPUT_COLUMNS].values.astype(
    np.float32
)

y = df[OUTPUT_COLUMNS].values.astype(
    np.float32
)

print("\nInput shape :", X.shape)
print("Output shape:", y.shape)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=8,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 7. SCALE DATA
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
# 8. BNN KL DIVERGENCE
# ============================================================

N_TRAIN = len(X_train)

KL_WEIGHT = 1.0 / N_TRAIN


def scaled_kl_divergence(q, p, _):

    kl = tfd.kl_divergence(
        q,
        p
    )

    return kl * KL_WEIGHT


# ============================================================
# 9. BUILD BNN
# ============================================================

model = tf_keras.Sequential([

    tf_keras.layers.Input(
        shape=(6,)
    ),

    tfp.layers.DenseFlipout(
        16,
        activation="relu",
        kernel_divergence_fn=scaled_kl_divergence,
        bias_divergence_fn=scaled_kl_divergence
    ),

    tfp.layers.DenseFlipout(
        8,
        activation="relu",
        kernel_divergence_fn=scaled_kl_divergence,
        bias_divergence_fn=scaled_kl_divergence
    ),

    tfp.layers.DenseFlipout(
        3,
        kernel_divergence_fn=scaled_kl_divergence,
        bias_divergence_fn=scaled_kl_divergence
    )
])


# ============================================================
# 10. NEGATIVE LOG LIKELIHOOD
# ============================================================

def negative_log_likelihood(
    y_true,
    y_pred
):

    distribution = tfd.Normal(
        loc=y_pred,
        scale=1.0
    )

    return -tf.reduce_mean(
        distribution.log_prob(y_true)
    )


# ============================================================
# 11. COMPILE
# ============================================================

model.compile(

    optimizer=tf_keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=negative_log_likelihood,

    metrics=["mae"]
)


# ============================================================
# 12. MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BNN MODEL")
print("=" * 60)

model.summary()


# ============================================================
# 13. TRAIN
# ============================================================

print("\n" + "=" * 60)
print("BNN TRAINING")
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
# 14. DETERMINISTIC TEST PREDICTION
# ============================================================

y_pred_scaled = model(
    X_test_scaled,
    training=False
).numpy()

y_pred = y_scaler.inverse_transform(
    y_pred_scaled
)


# ============================================================
# 15. EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("BNN TEST RESULTS")
print("=" * 60)

for i, target in enumerate(
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
# 16. MC STOCHASTIC PREDICTIONS
# ============================================================

N_MC = 200

mc_predictions = []

print("\nGenerating Monte Carlo predictions...")

for _ in range(N_MC):

    prediction = model(
        X_test_scaled,
        training=True
    ).numpy()

    prediction = y_scaler.inverse_transform(
        prediction
    )

    mc_predictions.append(
        prediction
    )


mc_predictions = np.array(
    mc_predictions
)

print(
    "MC prediction shape:",
    mc_predictions.shape
)


# ============================================================
# 17. MEAN / STD / 95% CI
# ============================================================

mc_mean = np.mean(
    mc_predictions,
    axis=0
)

mc_std = np.std(
    mc_predictions,
    axis=0
)

lower = mc_mean - 1.96 * mc_std

upper = mc_mean + 1.96 * mc_std


# ============================================================
# 18. DISPLAY UNCERTAINTY
# ============================================================

print("\n" + "=" * 60)
print("BNN PREDICTIONS + UNCERTAINTY")
print("=" * 60)

for i in range(len(X_test)):

    print(
        f"\nTest sample {i + 1}"
    )

    for j, target in enumerate(
        OUTPUT_COLUMNS
    ):

        print(
            f"{target:15s} "
            f"Actual={y_test[i,j]:.3f}  "
            f"Mean={mc_mean[i,j]:.3f}  "
            f"Std={mc_std[i,j]:.3f}  "
            f"95% CI=[{lower[i,j]:.3f}, "
            f"{upper[i,j]:.3f}]"
        )


# ============================================================
# 19. SAVE MODEL WEIGHTS
# ============================================================

weights_path = os.path.join(
    SAVE_DIR,
    "bnn_weld_weights.weights.h5"
)

model.save_weights(
    weights_path
)


# ============================================================
# 20. SAVE SCALERS
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
# 21. SAVE ENCODERS
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
# 22. SAVE TRAINING HISTORY
# ============================================================

pd.DataFrame(
    history.history
).to_csv(
    os.path.join(
        SAVE_DIR,
        "bnn_training_history.csv"
    ),
    index=False
)


# ============================================================
# 23. SAVE TEST PREDICTIONS
# ============================================================

results = pd.DataFrame({

    "actual_width":
        y_test[:, 0],

    "predicted_width":
        mc_mean[:, 0],

    "width_std":
        mc_std[:, 0],

    "actual_depth":
        y_test[:, 1],

    "predicted_depth":
        mc_mean[:, 1],

    "depth_std":
        mc_std[:, 1],

    "actual_reinforcement":
        y_test[:, 2],

    "predicted_reinforcement":
        mc_mean[:, 2],

    "reinforcement_std":
        mc_std[:, 2]

})

results.to_csv(
    os.path.join(
        SAVE_DIR,
        "bnn_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 24. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("BNN TRAINING COMPLETED")
print("=" * 60)

print("\nSaved:")

print(weights_path)

print(
    os.path.join(
        SAVE_DIR,
        "X_scaler.pkl"
    )
)

print(
    os.path.join(
        SAVE_DIR,
        "y_scaler.pkl"
    )
)

print(
    os.path.join(
        SAVE_DIR,
        "material_encoder.pkl"
    )
)

print(
    os.path.join(
        SAVE_DIR,
        "process_encoder.pkl"
    )
)

print("\nNext:")
print("python bnn\\bnn_inference.py")