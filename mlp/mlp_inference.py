import os
import pickle
import numpy as np
import tf_keras


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
    "mlp",
    "saved"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

model_path = os.path.join(
    SAVE_DIR,
    "mlp_weld_model.keras"
)

model = tf_keras.models.load_model(
    model_path
)


# ============================================================
# 3. LOAD SCALERS
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
# 4. LOAD ENCODERS
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
# 5. DISPLAY AVAILABLE OPTIONS
# ============================================================

print("AVAILABLE OPTIONS")

print(
    "\nMaterials:",
    list(material_encoder.classes_)
)

print(
    "Processes:",
    list(process_encoder.classes_)
)


# ============================================================
# 6. GET USER INPUT
# ============================================================

print("WELD BEAD PREDICTION - MLP")

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
# 7. ENCODE MATERIAL
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
# 8. ENCODE PROCESS
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
# 9. CREATE INPUT ARRAY
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
# 10. SCALE INPUT
# ============================================================

X_new_scaled = X_scaler.transform(
    X_new
)


# ============================================================
# 11. PREDICT
# ============================================================

prediction_scaled = model.predict(
    X_new_scaled,
    verbose=0
)


# ============================================================
# 12. CONVERT BACK TO ORIGINAL UNITS
# ============================================================

prediction = y_scaler.inverse_transform(
    prediction_scaled
)[0]


# ============================================================
# 13. DISPLAY PREDICTION
# ============================================================

print("PREDICTED WELD BEAD")

print(
    f"\nBead Width    : {prediction[0]:.4f}"
)

print(
    f"Bead Depth    : {prediction[1]:.4f}"
)

print(
    f"Reinforcement : {prediction[2]:.4f}"
)

print("PREDICTION COMPLETED")