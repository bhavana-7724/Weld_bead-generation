import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v2")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Color Palette
c_input = "#e3f2fd"       # Light Blue
c_regress = "#e8f5e9"     # Light Green
c_p1 = "#fff3e0"          # Light Orange
c_p2 = "#f3e5f5"          # Light Purple
c_ddpm = "#ffebee"        # Light Red
c_dash = "#eceff1"        # Light Gray

c_border_in = "#1976d2"
c_border_reg = "#388e3c"
c_border_p1 = "#f57c00"
c_border_p2 = "#7b1fa2"
c_border_ddpm = "#d32f2f"
c_border_dash = "#455a64"

# Title Header
ax.text(50, 96, "Predicting Weld Bead Profile for GMAW and Cold Arc Welding Using Deep Learning",
        ha='center', va='center', fontsize=16, fontweight='bold', color="#1a237e")
ax.text(50, 93, "System Architecture: Two-Pillar Hybrid Engineering Framework with Audited Diffusion Baseline",
        ha='center', va='center', fontsize=12, style='italic', color="#37474f")

# 1. Operational Input Block (Left)
rect_in = patches.FancyBboxPatch((3, 40), 18, 42, boxstyle="round,pad=1.5",
                                 facecolor=c_input, edgecolor=c_border_in, linewidth=2)
ax.add_patch(rect_in)
ax.text(12, 78, "OPERATIONAL\nINPUTS (WPS)", ha='center', va='center', fontsize=11, fontweight='bold', color=c_border_in)
ax.axhline(y=72, xmin=0.05, xmax=0.19, color=c_border_in, lw=1)

inputs_text = (
    "• Wire Feed Rate (F)\n"
    "  4.50 – 6.00 m/min\n\n"
    "• Travel Speed (S)\n"
    "  20.0 – 32.0 cm/min\n\n"
    "• Base Alloy\n"
    "  IN718 / SS304L\n\n"
    "• Welding Process\n"
    "  Cold Arc / GMAW\n\n"
    "─────────────────\n"
    "Pre-Weld Prospective\n"
    "(Zero Post-Weld Leakage)"
)
ax.text(12, 55, inputs_text, ha='center', va='center', fontsize=9.5, color="#212121")

# Arrow from Input to Regression
ax.annotate("", xy=(29, 68), xytext=(21.5, 68),
            arrowprops=dict(arrowstyle="->", color="#37474f", lw=2.5, mutation_scale=18))
ax.text(25.2, 70.5, "Input\nVector", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color="#37474f")

# 2. Regression Engine Block (Center-Left)
rect_reg = patches.FancyBboxPatch((29, 52), 22, 32, boxstyle="round,pad=1.5",
                                  facecolor=c_regress, edgecolor=c_border_reg, linewidth=2)
ax.add_patch(rect_reg)
ax.text(40, 80, "REGRESSION ENGINE\n(Continuous Predictor)", ha='center', va='center', fontsize=11, fontweight='bold', color=c_border_reg)
ax.axhline(y=73, xmin=0.31, xmax=0.49, color=c_border_reg, lw=1)

reg_text = (
    "• Proposed DNN (PyTorch)\n"
    "  Dense(128)→Drop(0.2)→\n"
    "  Dense(64)→Drop(0.1)→\n"
    "  Dense(32)→Output(3)\n\n"
    "• Bayesian Neural Net (BNN)\n"
    "  Monte Carlo Dropout (N=50)\n"
    "  Epistemic Uncertainty (±σ)\n\n"
    "• 5-Fold CV Avg MAE: 0.66 mm\n"
    "• Inference Speed: < 1 ms"
)
ax.text(40, 62, reg_text, ha='center', va='center', fontsize=9, color="#1b5e20")

# Arrow from Regression to Predicted Dimensions
ax.annotate("", xy=(56.5, 68), xytext=(51.5, 68),
            arrowprops=dict(arrowstyle="->", color="#37474f", lw=2.5, mutation_scale=18))

# Predicted Geometry Mini-Box
rect_geom = patches.FancyBboxPatch((56.5, 59), 11, 18, boxstyle="round,pad=0.8",
                                  facecolor="#ffffff", edgecolor="#37474f", linewidth=1.5)
ax.add_patch(rect_geom)
ax.text(62, 73, "PREDICTED\nGEOMETRY", ha='center', va='center', fontsize=9, fontweight='bold', color="#263238")
geom_text = "Width (W)\nDepth (D)\nHeight (R)\n[± σ in mm]"
ax.text(62, 64, geom_text, ha='center', va='center', fontsize=8.5, color="#d84315", fontweight='bold')

# Arrow from Predicted Geometry to Pillar 1
ax.annotate("", xy=(73, 76), xytext=(68, 71),
            arrowprops=dict(arrowstyle="->", color=c_border_p1, lw=2.5, mutation_scale=18))

# 3. Pillar 1: Geometry-First Parametric Profiler (Top-Right)
rect_p1 = patches.FancyBboxPatch((73, 56), 24, 32, boxstyle="round,pad=1.5",
                                 facecolor=c_p1, edgecolor=c_border_p1, linewidth=2)
ax.add_patch(rect_p1)
ax.text(85, 84, "PILLAR 1: GEOMETRIC PROFILER\n(Physics-Informed Model E)", ha='center', va='center', fontsize=10.5, fontweight='bold', color=c_border_p1)
ax.axhline(y=78, xmin=0.75, xmax=0.95, color=c_border_p1, lw=1)

p1_text = (
    "• Analytical Cross-Section Formulation:\n"
    "  Crown: yr(x) = R(1 - (2x/W)²)\n"
    "  Root:  yp(x) = -D(1 - (2x/W)²)\n"
    "• Structural Edge MAE: 0.0223 (Best)\n"
    "• Strict Convexity & Toe Angles\n"
    "• 100% Calibrated Scale (μm/pixel)\n"
    "• Zero Microstructural Hallucination\n"
    "• Direct AWS D1.1 / ASME Code Audit"
)
ax.text(85, 66.5, p1_text, ha='center', va='center', fontsize=8.5, color="#bf360c")

# Arrow from Operational Inputs to Pillar 2 (Bottom path)
ax.annotate("", xy=(73, 35), xytext=(21.5, 48),
            arrowprops=dict(arrowstyle="->", color=c_border_p2, lw=2.5, connectionstyle="arc3,rad=-0.15", mutation_scale=18))
ax.text(45, 36, "Operational Feature Query [Alloy, Proc, F, S]", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=c_border_p2)

# 4. Pillar 2: Condition-Matched Metallographic Retrieval (Middle-Right)
rect_p2 = patches.FancyBboxPatch((73, 20), 24, 28, boxstyle="round,pad=1.5",
                                 facecolor=c_p2, edgecolor=c_border_p2, linewidth=2)
ax.add_patch(rect_p2)
ax.text(85, 44, "PILLAR 2: EMPIRICAL RETRIEVAL\n(Metallographic Grounding Model F)", ha='center', va='center', fontsize=10.5, fontweight='bold', color=c_border_p2)
ax.axhline(y=38, xmin=0.75, xmax=0.95, color=c_border_p2, lw=1)

p2_text = (
    "• Normalized Parameter Distance Query:\n"
    "  d(x, xi) over identical Alloy & Process\n"
    "• Genuine Micrograph Retrieval:\n"
    "  Real dendrites, etching & HAZ boundaries\n"
    "• Mean PSNR: 21.03 dB, SSIM: 0.7129\n"
    "• Ground-Truth Optical Scale\n"
    "• Proximity Transparency Metric (ΔF, ΔS)"
)
ax.text(85, 29, p2_text, ha='center', va='center', fontsize=8.5, color="#4a148c")

# 5. Audited Baseline: Conditional DDPM (Bottom-Left)
rect_ddpm = patches.FancyBboxPatch((3, 5), 48, 26, boxstyle="round,pad=1.5",
                                   facecolor=c_ddpm, edgecolor=c_border_ddpm, linewidth=2, linestyle="--")
ax.add_patch(rect_ddpm)
ax.text(27, 27, "AUDITED RESEARCH BASELINE: CONDITIONAL DDPM (100 Epochs)", ha='center', va='center', fontsize=11, fontweight='bold', color=c_border_ddpm)
ax.axhline(y=23, xmin=0.05, xmax=0.49, color=c_border_ddpm, lw=1)

ddpm_text = (
    "• Architecture: U-Net with Bilinear Upsampling + FiLM Conditioning (V3-C, Cosine Schedule, T=200, CFG=3.0)\n"
    "• Empirical Assessment: Pixel MAE: 0.2223 | PSNR: 11.75 dB | SSIM: 0.2637 | Edge MAE: 0.0635\n"
    "• Root Cause Analysis of Generative Failure:\n"
    "  1. Extreme Data Scarcity (N=36 physical runs vs 10⁴ required for reverse score diffusion)\n"
    "  2. Area Imbalance (75-85% uniform background dominates unweighted L2 noise loss)\n"
    "  3. Additive Condition Collapse (Identical outputs across varying wire feed rates; seed-dominated)\n"
    "  4. Absence of Optical Calibration Header (Generates uncalibrated pixel arrays; mm cannot be inferred)\n"
    "• Role: Preserved as an audited negative result to prove mathematical limitations of diffusion on small datasets."
)
ax.text(27, 13.5, ddpm_text, ha='center', va='center', fontsize=8.5, color="#b71c1c")

# 6. Unified Engineering Synthesis Dashboard (Bottom-Right)
rect_dash = patches.FancyBboxPatch((56, 5), 41, 10, boxstyle="round,pad=1.0",
                                   facecolor=c_dash, edgecolor=c_border_dash, linewidth=2)
ax.add_patch(rect_dash)
ax.text(76.5, 12, "UNIFIED GUIDE DEMONSTRATION & WPS DASHBOARD", ha='center', va='center', fontsize=10.5, fontweight='bold', color=c_border_dash)
dash_text = "Outputs Side-by-Side: [Parametric Profile + Millimeter Dimensions] + [Real Micrograph Grounding] + [Audited DDPM Comparison]"
ax.text(76.5, 7.5, dash_text, ha='center', va='center', fontsize=8.5, color="#263238")

plt.tight_layout()
fig8_path = os.path.join(OUT_DIR, "fig8_system_architecture.png")
fig.savefig(fig8_path, dpi=300)
plt.close(fig)
print(f"Saved: {fig8_path}")
