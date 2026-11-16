"""Interactive launcher for the EXISTING Phase V14 weld interpolation pipeline.

Place this file anywhere inside ``weld_project`` (preferably at its root).
It does not retrain or replace the Phase V14 model. It passes four user-provided
welding conditions to ``scripts/phase_v14/generate_single_condition.py``, verifies
that the saved metadata matches the requested conditions, and opens the sketches.

Usage:
    python phase_v14_input_to_sketch.py
    python phase_v14_input_to_sketch.py --material IN718 --process ColdArc --feed 5.2 --speed 26
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


def project_root() -> Path:
    here = Path(__file__).resolve().parent
    for base in (here, *here.parents):
        if (base / "scripts" / "phase_v14" / "generate_single_condition.py").is_file():
            return base
    raise FileNotFoundError(
        "Cannot find scripts/phase_v14/generate_single_condition.py.\n"
        "Copy this launcher into your weld_project folder, then run it again."
    )


def ask_choice(label: str, choices: dict[str, str], preselected: str | None) -> str:
    if preselected:
        key = re.sub(r"[\s_-]+", "", preselected).lower()
        if key in choices:
            return choices[key]
        raise ValueError(f"Invalid {label}: {preselected!r}. Valid options: {sorted(set(choices.values()))}")
    while True:
        answer = input(f"{label} ({'/'.join(dict.fromkeys(choices.values()))}): ").strip()
        key = re.sub(r"[\s_-]+", "", answer).lower()
        if key in choices:
            return choices[key]
        print(f"  Invalid choice. Enter {' or '.join(dict.fromkeys(choices.values()))}.")


def ask_number(label: str, passed: float | None) -> float:
    while True:
        raw = str(passed) if passed is not None else input(f"{label}: ").strip()
        try:
            val = float(raw)
            if not math.isfinite(val) or val <= 0:
                raise ValueError
            return val
        except ValueError:
            if passed is not None:
                raise ValueError(f"{label} must be a positive number, got {passed!r}")
            print("  Enter a positive number (for example 5.2).")


def choose_option(help_text: str, names: tuple[str, ...], label: str) -> str:
    for name in names:
        if re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", help_text):
            return name
    raise RuntimeError(
        f"Your Phase V14 script does not show a recognized {label} option in --help.\n"
        "Run: python scripts\\phase_v14\\generate_single_condition.py --help\n"
        f"Expected one of: {', '.join(names)}.\n"
        "The launcher deliberately will not run with defaults instead of your inputs."
    )


def check_cli(generator: Path) -> dict[str, str]:
    run = subprocess.run(
        [sys.executable, str(generator), "--help"],
        cwd=generator.parents[2], text=True, capture_output=True, errors="replace"
    )
    if run.returncode:
        raise RuntimeError(
            "Could not inspect the Phase V14 generator's command-line options.\n"
            + (run.stderr or run.stdout)[-1500:]
        )
    help_text = run.stdout + "\n" + run.stderr
    return {
        "material": choose_option(help_text, ("--material", "--alloy"), "material"),
        "process": choose_option(help_text, ("--process", "--welding-process"), "process"),
        "feed": choose_option(help_text, ("--feed", "--feed-rate", "--wire-feed", "--wire-feed-rate"), "feed rate"),
        "speed": choose_option(help_text, ("--speed", "--travel-speed", "--weld-speed"), "travel speed"),
    }


def close_number(a: Any, b: float) -> bool:
    try:
        return math.isclose(float(a), b, rel_tol=0, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def find_outputs(root: Path, material: str, process: str, feed: float, speed: float, started_at: float):
    base = root / "results" / "phase_v14_sharp_interpolation"
    out = {}
    if not base.is_dir():
        return out
    for path in base.glob("generation_*/view_*/metadata.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            condition = doc.get("target_condition", doc.get("condition", {}))
            if not isinstance(condition, dict):
                continue
            if str(condition.get("material", "")).lower() != material.lower():
                continue
            if str(condition.get("process", "")).lower() != process.lower():
                continue
            if not close_number(condition.get("feed_rate"), feed):
                continue
            if not close_number(condition.get("speed"), speed):
                continue
            view = str(condition.get("view", path.parent.name.split("_")[-1])).upper()
            if view not in ("A", "B"):
                continue
            # Old outputs must not be confused with results of this fresh request.
            if path.stat().st_mtime + 5 < started_at:
                continue
            sketch = path.parent / "annotated_weld_bead_sketch.png"
            if not sketch.is_file():
                continue
            out[view] = (sketch, doc, path.parent)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Enter four welding parameters to generate Phase V14 W/D/R sketches.")
    parser.add_argument("--material", help="IN718 or SS304L")
    parser.add_argument("--process", help="ColdArc or GMAW")
    parser.add_argument("--feed", type=float, help="Wire feed rate, m/min")
    parser.add_argument("--speed", type=float, help="Travel speed, cm/min")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open PNG sketches")
    args = parser.parse_args()

    try:
        root = project_root()
        print("\n=== PHASE V14: WELD PROFILE / SKETCH GENERATOR ===")
        material = ask_choice("1. Material", {"in718": "IN718", "ss304l": "SS304L"}, args.material)
        process = ask_choice("2. Process", {"coldarc": "ColdArc", "gmaw": "GMAW"}, args.process)
        feed = ask_number("3. Wire feed rate (m/min)", args.feed)
        speed = ask_number("4. Travel speed (cm/min)", args.speed)
        generator = root / "scripts" / "phase_v14" / "generate_single_condition.py"
        cli = check_cli(generator)
        command = [
            sys.executable, str(generator),
            cli["material"], material, cli["process"], process,
            cli["feed"], str(feed), cli["speed"], str(speed)
        ]
        print(f"\nRunning interpolation for {material} | {process} | F={feed:g} m/min | S={speed:g} cm/min ...\n")
        started_at = time.time()
        completed = subprocess.run(command, cwd=root)
        if completed.returncode:
            print("\nPhase V14 generator failed. Your chosen condition may be outside the supported interpolation range.")
            print(f"To inspect supported flags: {sys.executable} scripts\\phase_v14\\generate_single_condition.py --help")
            return completed.returncode

        outputs = find_outputs(root, material, process, feed, speed, started_at)
        if not outputs:
            print("\nThe generator finished but no NEW matching annotated sketch was found.")
            print("Verify the generator really accepts CLI inputs rather than using hard-coded default conditions.")
            print("Look in: " + str(root / "results" / "phase_v14_sharp_interpolation"))
            return 2

        for view in sorted(outputs):
            sketch, doc, folder = outputs[view]
            dims = doc.get("svr_dimensions_mm", {})
            print(f"\nVIEW {view} — ANNOTATED SKETCH:")
            print("  " + str(sketch))
            if dims:
                print(f"  Predicted W = {dims.get('width_mm', '?')} mm")
                print(f"  Predicted D = {dims.get('depth_mm', '?')} mm")
                print(f"  Predicted R = {dims.get('reinforcement_mm', '?')} mm")
            panel = folder / "final_comparison_panel.png"
            if panel.exists():
                print("  Full result panel: " + str(panel))
            if not args.no_open and os.name == "nt":
                try:
                    os.startfile(str(sketch))  # Windows-only; opens with default PNG viewer.
                except OSError as e:
                    print(f"  Could not open image automatically: {e}")
        print("\nDone. Depth is a schematic SVR prediction; no measured fusion boundary is inferred from pixels.")
        return 0
    except (RuntimeError, ValueError, FileNotFoundError, KeyboardInterrupt) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
