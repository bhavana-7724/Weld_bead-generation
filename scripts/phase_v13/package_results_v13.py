"""
package_results_v13.py

Phase V13 Stage 5: Packaging & Integrity Verification
Target System: Single NVIDIA A100 GPU / Linux Server / Local Machine

Verifies and bundles all Phase V13 artifacts into a portable tarball:
- phase_v13_results_bundle.tar.gz
"""

import os
import sys
import time
import tarfile
import hashlib
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase_v13_a100")
BUNDLE_PATH = os.path.join(BASE_DIR, "phase_v13_results_bundle.tar.gz")


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_packaging():
    print("=" * 80)
    print("PHASE V13 [STAGE 5]: RESULTS PACKAGING & INTEGRITY VERIFICATION")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Source:    {RESULTS_DIR}")
    print(f"Target:    {BUNDLE_PATH}")
    print("=" * 80)

    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory {RESULTS_DIR} does not exist!")
        return 1

    # Inventory
    checkpoints = glob.glob(os.path.join(RESULTS_DIR, "checkpoints", "**", "*.pt"), recursive=True)
    metrics_csvs = glob.glob(os.path.join(RESULTS_DIR, "metrics", "**", "*.csv"), recursive=True)
    figures = glob.glob(os.path.join(RESULTS_DIR, "figures", "*.png"))
    panels = glob.glob(os.path.join(RESULTS_DIR, "panels", "*.png"))
    reports = glob.glob(os.path.join(RESULTS_DIR, "reports", "*.md"))

    print("\n[ARTIFACT INVENTORY]")
    print(f"  PyTorch Checkpoints:   {len(checkpoints)}")
    print(f"  Metrics CSV Tables:    {len(metrics_csvs)}")
    print(f"  Publication Figures:   {len(figures)}")
    print(f"  Academic Panels:       {len(panels)}")
    print(f"  Scientific Reports:    {len(reports)}")

    print(f"\nCompressing {RESULTS_DIR} into {BUNDLE_PATH}...")
    t0 = time.time()
    with tarfile.open(BUNDLE_PATH, "w:gz") as tar:
        tar.add(RESULTS_DIR, arcname=os.path.relpath(RESULTS_DIR, BASE_DIR))
    duration = time.time() - t0

    size_mb = os.path.getsize(BUNDLE_PATH) / (1024 * 1024)
    sha256_hash = compute_sha256(BUNDLE_PATH)

    print("\n" + "=" * 80)
    print("RESULTS BUNDLE CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Archive Path:    {BUNDLE_PATH}")
    print(f"  Archive Size:    {size_mb:.2f} MB")
    print(f"  SHA-256 Hash:    {sha256_hash}")
    print(f"  Compression Time: {duration:.2f} s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_packaging())
