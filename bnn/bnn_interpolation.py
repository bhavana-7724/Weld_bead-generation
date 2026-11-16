import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


tfd = tfp.distributions


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "clean_dataset_36.csv"
)

SAVE_DIR = (
    Path(__file__).resolve().parent
    / "saved"
)

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


# Center points of the four experimental grids
HIDDEN_SAMPLES = [
    5,
    14,
    22,
    32
]


print("=" * 70)
print("BNN INTERPOLATION EXPERIMENT")
print("=" * 70)

print("\nHidden samples:", HIDDEN_SAMPLES)


# ============================================================
# 3. HIDE CENTER EXPERIMENTS
# ============================================================

test_df = df[
    df["sample"].isin(HIDDEN_SAMPLES)
].copy()

train_df = df[
    ~df["sample"].isin(HIDDEN_SAMPLES)
].copy()


print("\nTraining samples:", len(train_df))
print("Hidden samples  :", len(test_df))


print("\nHidden experimental points:")

print(
    test_df[
        [
            "sample",
            "material",
            "process"
        ]
        + INPUT_COLUMNS
        + OUTPUT_COLUMNS
    ].to_string(index=False)
)


# ============================================================
# 4. PREPARE X AND Y
# ============================================================

X_train = train_df[
    INPUT_COLUMNS
].values

X_test = test_df[
    INPUT_COLUMNS
].values

y_train = train_df[
    OUTPUT_COLUMNS
].values

y_test = test_df[
    OUTPUT_COLUMNS
].values


# ============================================================
# 5. SCALE INPUTS
# ============================================================

X_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(
    X_train
)

X_test_scaled = X_scaler.transform(
    X_test
)


# ============================================================
# 6. SCALE OUTPUTS
# ============================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train
)

y_test_scaled = y_scaler.transform(
    y_test
)


# ============================================================
# 7. KL DIVERGENCE
# ============================================================

N_TRAIN = len(X_train)

KL_WEIGHT = 1.0 / N_TRAIN


def scaled_kl_divergence(q, p, _):

    kl = tfd.kl_divergence(q, p)

    return kl * KL_WEIGHT


# ============================================================
# 8. BUILD BNN
# ============================================================

model = tf_keras.Sequential([

    tf_keras.layers.Input(
        shape=(4,)
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
# 9. NEGATIVE LOG LIKELIHOOD
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
        distribution.log_prob(
            y_true
        )
    )


model.compile(
    optimizer=tf_keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=negative_log_likelihood
)


print("\nBNN architecture:")

model.summary()


# ============================================================
# 10. TRAIN
# ============================================================

print("\nTraining BNN...")

model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=500,
    batch_size=8,
    verbose=0
)

print("Training completed.")


# ============================================================
# 11. MONTE CARLO PREDICTIONS
# ============================================================

N_MC = 200

mc_predictions = []


print(
    f"\nGenerating {N_MC} stochastic predictions..."
)


for _ in range(N_MC):

    prediction = model(
        X_test_scaled,
        training=True
    ).numpy()

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
# 12. CONVERT EACH MC PREDICTION
#     BACK TO ORIGINAL UNITS
# ============================================================

mc_predictions_original = np.array([

    y_scaler.inverse_transform(
        prediction
    )

    for prediction
    in mc_predictions

])


# Shape:
# (MC samples, hidden experiments, outputs)


# ============================================================
# 13. MEAN + STD
# ============================================================

prediction_mean = (
    mc_predictions_original.mean(
        axis=0
    )
)


prediction_std = (
    mc_predictions_original.std(
        axis=0
    )
)


# ============================================================
# 14. 95% INTERVAL
# ============================================================

lower_bound = (
    prediction_mean
    - 1.96 * prediction_std
)

upper_bound = (
    prediction_mean
    + 1.96 * prediction_std
)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BNN INTERPOLATION RESULTS")
print("=" * 70)


results = []


for i in range(len(test_df)):

    sample_id = test_df.iloc[i]["sample"]

    material = test_df.iloc[i]["material"]

    process = test_df.iloc[i]["process"]


    print(
        f"\nSample {int(sample_id)}"
    )

    print(
        f"Material : {material}"
    )

    print(
        f"Process  : {process}"
    )


    print("\nPredictions:")


    row = {
        "sample": sample_id,
        "material": material,
        "process": process
    }


    for j, output in enumerate(
        OUTPUT_COLUMNS
    ):

        actual = y_test[i, j]

        mean = prediction_mean[i, j]

        std = prediction_std[i, j]

        lower = lower_bound[i, j]

        upper = upper_bound[i, j]

        error = abs(
            actual - mean
        )


        print(
            f"\n  {output}"
        )

        print(
            f"    Actual     : {actual:.4f}"
        )

        print(
            f"    Prediction : {mean:.4f}"
        )

        print(
            f"    Std        : {std:.4f}"
        )

        print(
            f"    95% CI     : "
            f"[{lower:.4f}, {upper:.4f}]"
        )

        print(
            f"    Error      : {error:.4f}"
        )


        row[
            f"actual_{output}"
        ] = actual

        row[
            f"predicted_{output}"
        ] = mean

        row[
            f"std_{output}"
        ] = std

        row[
            f"lower_{output}"
        ] = lower

        row[
            f"upper_{output}"
        ] = upper

        row[
            f"error_{output}"
        ] = error


    results.append(row)


# ============================================================
# 16. METRICS
# ============================================================

print("\n" + "=" * 70)
print("BNN INTERPOLATION METRICS")
print("=" * 70)


for i, output in enumerate(
    OUTPUT_COLUMNS
):

    mae = mean_absolute_error(
        y_test[:, i],
        prediction_mean[:, i]
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test[:, i],
            prediction_mean[:, i]
        )
    )


    r2 = r2_score(
        y_test[:, i],
        prediction_mean[:, i]
    )


    print(
        f"\n{output.upper()}"
    )

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
# 17. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


output_path = (
    SAVE_DIR
    / "bnn_interpolation_results.csv"
)


results_df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(output_path)