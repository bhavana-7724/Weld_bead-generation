"""
build_clean_dataset.py

Builds the single source-of-truth CSV for the 36-sample welding dataset.
Values below are taken from the lab-authoritative PDF (parameters.pdf),
which was used to resolve 3 known conflicts against the raw Excel file:

    Sample 27 Efficiency: Excel had 6.35   -> corrected to 60.35
    Sample 30 Efficiency: Excel had 90.8   -> corrected to 95.80
    Sample 32 Reinforcement: Excel had 3.74 -> corrected to 2.74

Do not hand-edit this file. If a future conflict appears, resolve it
against the lab source first, then update the ROWS list below with a
comment explaining the change.
"""

import pandas as pd

# Columns: sample, material, process, feed_rate, speed, HIP, efficiency,
#          width, depth, reinforcement
ROWS = [
    # IN718 - Cold Arc
    (1,  "IN718", "ColdArc", 5.0, 20, 139.08, 83.45, 4.52, 1.46, 3.79),
    (2,  "IN718", "ColdArc", 5.5, 20, 153.44, 92.06, 10.76, 1.77, 4.09),
    (3,  "IN718", "ColdArc", 6.0, 20, 169.29, 101.57, 12.45, 1.87, 3.93),
    (4,  "IN718", "ColdArc", 5.0, 25, 124.64, 74.78, 9.74, 1.51, 3.08),
    (5,  "IN718", "ColdArc", 5.5, 25, 137.17, 82.30, 10.31, 1.41, 3.36),
    (6,  "IN718", "ColdArc", 6.0, 25, 154.00, 92.40, 10.37, 3.42, 1.70),
    (7,  "IN718", "ColdArc", 5.0, 30, 103.87, 62.32, 8.89, 3.17, 1.54),
    (8,  "IN718", "ColdArc", 5.5, 30, 116.44, 69.86, 9.84, 3.08, 1.38),
    (9,  "IN718", "ColdArc", 6.0, 30, 124.67, 74.80, 10.79, 3.10, 1.74),

    # IN718 - GMAW
    (10, "IN718", "GMAW", 4.5, 20, 116.76, 70.06, 9.08, 0.91, 3.86),
    (11, "IN718", "GMAW", 5.0, 20, 126.42, 75.85, 9.09, 1.29, 3.88),
    (12, "IN718", "GMAW", 5.5, 20, 151.36, 90.82, 10.28, 1.78, 4.04),
    (13, "IN718", "GMAW", 4.5, 25, 92.74, 55.64, 8.47, 0.89, 1.03),
    (14, "IN718", "GMAW", 5.0, 25, 108.07, 64.84, 9.79, 0.40, 3.07),
    (15, "IN718", "GMAW", 5.5, 25, 121.09, 72.65, 9.79, 1.33, 3.25),
    (16, "IN718", "GMAW", 4.5, 30, 77.28, 46.37, 7.42, 0.51, 3.26),
    (17, "IN718", "GMAW", 5.0, 30, 87.21, 52.33, 9.47, 0.79, 3.17),
    (18, "IN718", "GMAW", 5.5, 30, 98.04, 58.82, 8.81, 1.07, 3.12),

    # SS304L - Cold Arc
    (19, "SS304L", "ColdArc", 5.5, 22, 155.88, 93.53, 9.21, 2.70, 3.83),
    (20, "SS304L", "ColdArc", 6.0, 22, 151.20, 90.72, 10.83, 3.13, 6.87),
    (21, "SS304L", "ColdArc", 5.0, 22, 128.07, 76.84, 9.77, 2.31, 3.68),
    (22, "SS304L", "ColdArc", 5.5, 27, 127.01, 76.21, 9.00, 2.54, 3.24),
    (23, "SS304L", "ColdArc", 6.0, 27, 138.52, 83.11, 9.39, 2.13, 3.73),
    (24, "SS304L", "ColdArc", 5.0, 27, 106.00, 63.60, 8.38, 2.43, 3.42),
    (25, "SS304L", "ColdArc", 5.5, 32, 111.16, 66.70, 7.46, 1.92, 3.21),
    (26, "SS304L", "ColdArc", 6.0, 32, 128.56, 77.14, 8.54, 2.01, 3.50),
    (27, "SS304L", "ColdArc", 5.0, 32, 100.58, 60.35, 8.69, 2.23, 2.94),  # corrected efficiency

    # SS304L - GMAW
    (28, "SS304L", "GMAW", 4.5, 22, 114.36, 68.62, 8.40, 1.60, 2.95),
    (29, "SS304L", "GMAW", 5.0, 22, 133.53, 80.12, 9.63, 2.38, 3.45),
    (30, "SS304L", "GMAW", 5.5, 22, 159.66, 95.80, 12.36, 4.47, 3.20),  # corrected efficiency
    (31, "SS304L", "GMAW", 4.5, 27, 97.87, 58.72, 8.61, 1.62, 3.12),
    (32, "SS304L", "GMAW", 5.0, 27, 115.73, 69.44, 10.43, 5.25, 2.74),  # corrected reinforcement
    (33, "SS304L", "GMAW", 5.5, 27, 130.81, 78.49, 10.87, 3.14, 3.18),
    (34, "SS304L", "GMAW", 4.5, 32, 86.83, 52.10, 9.96, 1.93, 3.07),
    (35, "SS304L", "GMAW", 5.0, 32, 98.76, 59.26, 10.14, 2.08, 2.88),
    (36, "SS304L", "GMAW", 5.5, 32, 107.96, 64.78, 8.97, 2.13, 3.30),
]

COLUMNS = [
    "sample", "material", "process", "feed_rate", "speed",
    "HIP", "efficiency", "width", "depth", "reinforcement",
]

def main():
    df = pd.DataFrame(ROWS, columns=COLUMNS)

    assert df["sample"].tolist() == list(range(1, 37)), "Sample numbers not 1..36 in order"
    assert df.isnull().sum().sum() == 0, "Found unexpected missing values"

    out_path = "clean_dataset_36.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")
    print(df.groupby(["material", "process"]).size())
    print("\nPreview:")
    print(df.head())

if __name__ == "__main__":
    main()