import os
import pickle
import numpy as np

import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp


tfd = tfp.distributions


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SAVE_DIR = os.path.join(
    BASE_DIR,
    "bnn",
    "saved"
)


# ============================================================
# 2. BNN ARCHITECTURE
# ============================================================

N_TRAIN = 28
KL_WEIGHT = 1.0 / N_TRAIN


def scaled_kl_divergence(q, p, _):

    kl = tfd.kl_divergence(
        q,
        p
    )

    return kl * KL_WEIGHT


# ============================================================
# 3. BUILD SAME BNN ARCHITECTURE
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
# 4. LOAD WEIGHTS
# ============================================================

weights_path = os.path.join(
    SAVE_DIR,
    "bnn_weld_weights.weights.h5"
)

model.load_weights(
    weights_path
)


# ============================================================
# 5. LOAD SCALERS
# ============================================================

with open(
    os.path.join(
        SAVE_DIR,
        "X_scaler.pkl"
    ),
    "rb"
) as f:

    X_scaler = pickle.load(f)


with open(
    os.path.join(
        SAVE_DIR,
        "y_scaler.pkl"
    ),
    "rb"
) as f:

    y_scaler = pickle.load(f)


# ============================================================
# 6. LOAD ENCODERS
# ============================================================

with open(
    os.path.join(
        SAVE_DIR,
        "material_encoder.pkl"
    ),
    "rb"
) as f:

    material_encoder = pickle.load(f)


with open(
    os.path.join(
        SAVE_DIR,
        "process_encoder.pkl"
    ),
    "rb"
) as f:

    process_encoder = pickle.load(f)


# ============================================================
# 7. AVAILABLE OPTIONS
# ============================================================

print("AVAILABLE OPTIONS")

print(
    "Materials:",
    list(material_encoder.classes_)
)

print(
    "Processes:",
    list(process_encoder.classes_)
)


# ============================================================
# 8. GET USER INPUT
# ============================================================

print("WELD BEAD PREDICTION - BNN")

print("Enter welding parameters:")


feed_rate = float(
    input("Feed Rate: ")
)

speed = float(
    input("Speed: ")
)

HIP = float(
    input("HIP: ")
)

efficiency = float(
    input("Efficiency: ")
)

material = input(
    "Material: "
).strip()

process = input(
    "Process: "
).strip()


# ============================================================
# 9. ENCODE MATERIAL
# ============================================================

try:

    material_encoded = material_encoder.transform(
        [material]
    )[0]

except ValueError:

    print("\nERROR: Invalid material.")

    print(
        "Available materials:",
        list(material_encoder.classes_)
    )

    raise SystemExit


# ============================================================
# 10. ENCODE PROCESS
# ============================================================

try:

    process_encoded = process_encoder.transform(
        [process]
    )[0]

except ValueError:

    print("\nERROR: Invalid process.")

    print(
        "Available processes:",
        list(process_encoder.classes_)
    )

    raise SystemExit


# ============================================================
# 11. CREATE INPUT
# ============================================================

X_new = np.array(
    [[
        feed_rate,
        speed,
        HIP,
        efficiency,
        material_encoded,
        process_encoded
    ]],
    dtype=np.float32
)


# ============================================================
# 12. SCALE INPUT
# ============================================================

X_new_scaled = X_scaler.transform(
    X_new
)


# ============================================================
# 13. MONTE CARLO PREDICTION
# ============================================================

N_MC = 200

predictions = []


for _ in range(N_MC):

    prediction_scaled = model(
        X_new_scaled,
        training=True
    ).numpy()

    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )

    predictions.append(
        prediction[0]
    )


predictions = np.array(
    predictions
)


# ============================================================
# 14. CALCULATE MEAN AND STD
# ============================================================

mean_prediction = np.mean(
    predictions,
    axis=0
)

std_prediction = np.std(
    predictions,
    axis=0
)


# ============================================================
# 15. 95% INTERVAL
# ============================================================

lower = (
    mean_prediction
    - 1.96 * std_prediction
)

upper = (
    mean_prediction
    + 1.96 * std_prediction
)


# ============================================================
# 16. DISPLAY INPUT
# ============================================================

print("INPUT")

print(
    f"Feed Rate : {feed_rate}"
)

print(
    f"Speed     : {speed}"
)

print(
    f"HIP       : {HIP}"
)

print(
    f"Efficiency: {efficiency}"
)

print(
    f"Material  : {material}"
)

print(
    f"Process   : {process}"
)


# ============================================================
# 17. DISPLAY BNN PREDICTION
# ============================================================

print("BNN PREDICTION")

print(
    f"Bead Width"
)

print(
    f"  Mean : {mean_prediction[0]:.4f}"
)

print(
    f"  Std  : {std_prediction[0]:.4f}"
)

print(
    f"  95% CI : "
    f"[{lower[0]:.4f}, {upper[0]:.4f}]"
)


print(
    f"Bead Depth"
)

print(
    f"  Mean : {mean_prediction[1]:.4f}"
)

print(
    f"  Std  : {std_prediction[1]:.4f}"
)

print(
    f"  95% CI : "
    f"[{lower[1]:.4f}, {upper[1]:.4f}]"
)


print(
    f"Reinforcement"
)

print(
    f"  Mean : {mean_prediction[2]:.4f}"
)

print(
    f"  Std  : {std_prediction[2]:.4f}"
)

print(
    f"  95% CI : "
    f"[{lower[2]:.4f}, {upper[2]:.4f}]"
)


# ============================================================
# 18. FINISHED
# ============================================================

print("BNN PREDICTION COMPLETED")