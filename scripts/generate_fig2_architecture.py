"""
generate_fig2_architecture.py

Generates Figure 2: Publication Schematic of the Edge-Aware Conditional Weld
Profile Generator (EC-WPG) Architecture and Composite Loss Formulation.

Features:
- Parameter Conditioning Vector c in R^6
- Condition Projection MLP
- Shared Feature Trunk with 4 Transposed Conv / ResUpBlock Stages
- Dual Output Heads:
  1. High-Fidelity RGB Weld Cross-Section Synthesis Head
  2. Auxiliary Weld Boundary / Edge Prediction Head
- Multi-Component Supervisory Losses:
  - L_img (Global L1 + MSE)
  - L_roi (Weld ROI Focused Loss)
  - L_grad (Differentiable 2D Sobel Gradient Loss)
  - L_edge (Auxiliary Canny Edge BCE + Dice Loss)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v6")
os.makedirs(OUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")

# Title and Subtitle
ax.text(7.5, 8.15, "Figure 2: Edge-Aware Conditional Weld Profile Generator (EC-WPG) Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1a252f")
ax.text(7.5, 7.82, "Dually Supervised Prospective Conditioning with Auxiliary Boundary Prediction and Differentiable Spatial Gradient Regularization",
        ha="center", va="center", fontsize=10, fontstyle="italic", color="#555555")

# Helper to draw stylish rounded boxes
def draw_block(ax, x, y, w, h, title, subtitle="", color="#e8f4f8", border="#2980b9", radius=0.15):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.05,rounding_size={radius}",
                                 facecolor=color, edgecolor=border, linewidth=1.5, zorder=2)
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=9, fontweight="bold", color="#2c3e50", zorder=3)
        ax.text(x + w/2, y + h*0.32, subtitle, ha="center", va="center", fontsize=7.5, color="#555555", zorder=3)
    else:
        ax.text(x + w/2, y + h*0.5, title, ha="center", va="center", fontsize=9, fontweight="bold", color="#2c3e50", zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color="#34495e", lw=1.5, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=3, shrinkB=3), zorder=4)
    if label:
        ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.15, label, ha="center", va="center", fontsize=7.5, color="#34495e", fontweight="bold")

# 1. Conditioning Inputs
draw_block(ax, 0.4, 4.4, 2.0, 1.8, "Prospective Input\nCondition c in R^6",
           "F: Feed rate [m/min]\nS: Travel speed [cm/min]\nMat: [IN718, SS304L]\nProc: [ColdArc, GMAW]",
           color="#e8f8f5", border="#1abc9c")

# Latent noise input
draw_block(ax, 0.4, 2.6, 2.0, 1.2, "Latent Noise z in R^16", "N(0, 0.05^2)\n(Zero at inference)",
           color="#f4ecf7", border="#8e44ad")

# Concatenation node
ax.plot([2.7], [4.1], marker='o', markersize=10, color="#34495e", zorder=4)
ax.text(2.7, 4.1, "+", ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=5)
draw_arrow(ax, 2.4, 5.3, 2.65, 4.2)
draw_arrow(ax, 2.4, 3.2, 2.65, 4.0)

# 2. Condition Projection MLP
draw_block(ax, 3.1, 3.3, 1.8, 1.6, "Condition MLP",
           "Linear(22 -> 256)\nBatchNorm + LReLU\nLinear(256 -> 8192)\nReshape to (128, 8, 8)",
           color="#fef9e7", border="#f39c12")
draw_arrow(ax, 2.8, 4.1, 3.1, 4.1, label="[c; z]")

# 3. Shared Convolutional Trunk
draw_block(ax, 5.3, 4.8, 1.7, 1.2, "ResUpStage 1", "8x8 -> 16x16\nConvT + Conv2d\nChannels: 128 -> 64", color="#ebf5fb", border="#3498db")
draw_block(ax, 5.3, 3.3, 1.7, 1.2, "ResUpStage 2", "16x16 -> 32x32\nConvT + Conv2d\nChannels: 64 -> 32", color="#ebf5fb", border="#3498db")
draw_block(ax, 5.3, 1.8, 1.7, 1.2, "ResUpStages 3-4", "32x32 -> 128x128\nConvT + Conv2d\nChannels: 32 -> 16", color="#ebf5fb", border="#3498db")

draw_arrow(ax, 4.9, 4.1, 5.3, 5.4, label="h_0 (128x8x8)")
draw_arrow(ax, 6.15, 4.8, 6.15, 4.5)
draw_arrow(ax, 6.15, 3.3, 6.15, 3.0)

# Deep Feature Map Node
draw_block(ax, 7.4, 3.4, 1.5, 1.4, "Shared Feature\nRepresentation", "16 x 128 x 128\nSpatial Tensor", color="#eaeded", border="#7f8c8d")
draw_arrow(ax, 7.0, 2.4, 7.4, 3.9)

# 4. Dual Heads
# Head 1: Image Head
draw_block(ax, 9.4, 5.0, 1.9, 1.4, "Head 1: Image Synth", "Conv2d(16->16, 3x3)\nLReLU\nConv2d(16->3, 3x3)\nSigmoid Activation", color="#e8f8f5", border="#2ecc71")
draw_arrow(ax, 8.9, 4.3, 9.4, 5.6, color="#27ae60", label="Features")

# Head 2: Boundary/Edge Head
draw_block(ax, 9.4, 2.2, 1.9, 1.4, "Head 2: Edge Head", "Conv2d(16->16, 3x3)\nLReLU\nConv2d(16->1, 3x3)\nSigmoid Activation", color="#fbeee6", border="#e67e22")
draw_arrow(ax, 8.9, 3.9, 9.4, 3.0, color="#d35400", label="Features")

# 5. Generated Outputs
draw_block(ax, 11.8, 5.1, 1.4, 1.2, "Predicted\nImage I_hat", "3 x 128 x 128\nRGB Weld", color="#d4efdf", border="#27ae60")
draw_arrow(ax, 11.3, 5.7, 11.8, 5.7)

draw_block(ax, 11.8, 2.3, 1.4, 1.2, "Predicted\nEdge E_hat", "1 x 128 x 128\nBoundary Prob", color="#fcd8cc", border="#d35400")
draw_arrow(ax, 11.3, 2.9, 11.8, 2.9)

# 6. Loss Formulations Box (Right Bottom)
loss_box = patches.FancyBboxPatch((0.4, 0.2), 14.2, 1.3, boxstyle="round,pad=0.05,rounding_size=0.1",
                                  facecolor="#f8f9f9", edgecolor="#bdc3c7", linewidth=1.2, zorder=2)
ax.add_patch(loss_box)

loss_title = "Joint Differentiable Objective Function:  L_total = L_img + lambda_roi * L_roi + lambda_grad * L_grad + lambda_edge * L_edge"
ax.text(7.5, 1.22, loss_title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1a5276", zorder=3)

l1_txt = "L_img = ||I_hat - I||_1 + 0.5||I_hat - I||_2^2   (Global Photorealistic Reconstruction)"
l2_txt = "L_roi = ||M_roi * (I_hat - I)||_1 / sum(M_roi)   (Reinforcement & Fusion Zone Capacity Focusing)"
l3_txt = "L_grad = ||grad(I_hat) - grad(I)||_1   (Fixed 3x3 Sobel Kernels for Crisp Boundary Sharpness)"
l4_txt = "L_edge = BCE(E_hat, E_canny) + Dice(E_hat, E_canny)   (Supervised Auxiliary Edge Localization)"

ax.text(0.7, 0.85, l1_txt, fontsize=8, color="#2c3e50", zorder=3)
ax.text(0.7, 0.45, l2_txt, fontsize=8, color="#2c3e50", zorder=3)
ax.text(7.7, 0.85, l3_txt, fontsize=8, color="#2c3e50", zorder=3)
ax.text(7.7, 0.45, l4_txt, fontsize=8, color="#2c3e50", zorder=3)

# Loss links
draw_arrow(ax, 13.2, 5.7, 13.8, 5.7, color="#2980b9")
ax.text(14.3, 5.7, "Supervised by\nMacrograph I", ha="center", va="center", fontsize=7.5, color="#2980b9", fontweight="bold")

draw_arrow(ax, 13.2, 2.9, 13.8, 2.9, color="#d35400")
ax.text(14.3, 2.9, "Supervised by\nCanny E_canny", ha="center", va="center", fontsize=7.5, color="#d35400", fontweight="bold")

fig_path = os.path.join(OUT_DIR, "fig2_architecture.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Generated Figure 2 Architecture Diagram: {fig_path}", flush=True)
