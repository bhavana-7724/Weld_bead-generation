"""
Predict bead geometry.
Interactive:  python model/predict.py
Command line: python model/predict.py --feed 5.5 --speed 32 --material SS304L --process GMAW
"""
import os, sys, pickle, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weld_model import TARGETS, MATERIALS, PROCESSES, build_features

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE, "saved", "weld_model.pkl"), "rb") as f:
    bundle = pickle.load(f)
model, ranges = bundle["model"], bundle["ranges"]

ap = argparse.ArgumentParser()
ap.add_argument("--feed", type=float)
ap.add_argument("--speed", type=float)
ap.add_argument("--material", choices=MATERIALS)
ap.add_argument("--process", choices=PROCESSES)
a = ap.parse_args()

print("=" * 56 + "\nWELD BEAD PREDICTION\n" + "=" * 56)
print("Materials:", MATERIALS, "| Processes:", PROCESSES)
feed = a.feed if a.feed is not None else float(input("Feed rate: "))
speed = a.speed if a.speed is not None else float(input("Speed: "))
material = a.material or input(f"Material {MATERIALS}: ").strip()
process = a.process or input(f"Process {PROCESSES}: ").strip()

for name, v in (("feed_rate", feed), ("speed", speed)):
    lo, hi = ranges[name]
    if not lo <= v <= hi:
        print(f"\nWARNING: {name}={v} is outside the tested range "
              f"[{lo}, {hi}]. This is an extrapolation - do not trust it.")

X = build_features(feed, speed, material, process)
mu, sd = model.predict(X)

print("\n" + "-" * 56)
for j, t in enumerate(TARGETS):
    print(f"{t.capitalize():14s} {mu[0, j]:6.2f} mm   (95% range "
          f"{mu[0, j] - 1.96 * sd[0, j]:.2f} to {mu[0, j] + 1.96 * sd[0, j]:.2f})")
print("-" * 56)
print("Use as a starting point; confirm with a trial weld before production.")
