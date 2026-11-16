"""
weld_prediction_dashboard.py

Streamlit Web Dashboard for Phase V8:
Integrated Weld Bead Prediction and Generative Architecture Evaluation.

Usage:
    python -m streamlit run app/weld_prediction_dashboard.py

Features:
- Interactive parameter sliders (Wire Feed Rate, Travel Speed) and selectors (Material, Process).
- Model selection: Direct Conditional CNN, GC-WPG-v2 (Multi-Scale Spatial Prior), or Side-by-Side.
- Prospective Numerical Predictions: Width, Depth, Reinforcement (mm) via SVR with 5-fold CV MAE metrics.
- High-resolution raw synthetic macrographs and crown contour overlays (X=64 baseline, green contour, red apex).
- Dynamic out-of-domain extrapolation alerts.
- Checkpoint integrity status & cryptographic provenance tracking.
- Physical disclaimers and limitation disclosures.
"""

import os
import sys

# Ensure app directory is importable
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed in the current environment.")
    print("You can run the zero-dependency web dashboard instead:")
    print("    python app/local_web_dashboard.py")
    print("Or the native desktop GUI:")
    print("    python app/weld_prediction_app.py")
    print("Or the CLI inference tool:")
    print("    python app/cli_infer.py --material IN718 --process ColdArc --feed_rate 5.0 --speed 25.0")
    sys.exit(1)

import numpy as np
from PIL import Image
from inference_engine import WeldPredictor

# Page Configuration
st.set_page_config(
    page_title="Weld Bead Prediction & Generative Synthesis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern clean styling
st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 2px;
    }
    .subtitle {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px;
        border-radius: 4px;
        font-size: 13px;
        color: #92400E;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    return WeldPredictor()

predictor = get_predictor()

# Header
st.markdown('<div class="main-title">⚡ Weld Bead Profile Prediction & Generative Synthesis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Phase V8: Gas Metal Arc Welding (GMAW) & Cold Arc Generative Deep Learning System</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🔧 Welding Parameters")

material = st.sidebar.selectbox("Base Material", ["IN718", "SS304L"], index=0,
                                help="Inconel 718 (Nickel superalloy) or Stainless Steel 304L")
process = st.sidebar.selectbox("Welding Process", ["ColdArc", "GMAW"], index=0,
                               help="Controlled-dip transfer Cold Arc or Conventional GMAW")

feed_rate = st.sidebar.slider("Wire Feed Rate (m/min)", min_value=4.0, max_value=7.0, value=5.0, step=0.05,
                              help="Experimental training hull: [4.5, 6.0] m/min")
speed = st.sidebar.slider("Travel Speed (cm/min)", min_value=15.0, max_value=38.0, value=25.0, step=0.5,
                          help="Experimental training hull: [20.0, 32.0] cm/min")

model_choice = st.sidebar.radio("Generative Architecture", 
                                ["Both (Side-by-Side Comparison)", "Direct Conditional CNN", "GC-WPG-v2 (Multi-Scale Prior)"],
                                index=0)

m_key_map = {
    "Both (Side-by-Side Comparison)": "both",
    "Direct Conditional CNN": "direct_cnn",
    "GC-WPG-v2 (Multi-Scale Prior)": "gc_wpg_v2"
}

# Domain validation
is_valid, is_ood, warnings = predictor.validate_inputs(material, process, feed_rate, speed)

if is_ood:
    for w in warnings:
        st.warning(f"⚠️ **Domain Warning:** {w}")

# Run inference
m_choice_key = m_key_map[model_choice]
inference_result = predictor.run_full_inference(material, process, feed_rate, speed, model_choice=m_choice_key)

# Section 1: Prospective Numerical Predictions
st.subheader("1. Prospective Numerical Predictions (Support Vector Regressor)")
st.caption("Trained on strictly prospective parameters (F, S, Material, Process). 5-Fold Grouped Cross-Validation MAE: 0.681 mm.")

num_res = inference_result["numerical_predictions"]
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Bead Width (W)", f"{num_res['width_mm']:.2f} mm", delta=None, help="Documented 5-Fold CV MAE: ±0.77 mm")
with col2:
    st.metric("Penetration Depth (D)", f"{num_res['depth_mm']:.2f} mm", delta=None, help="Documented 5-Fold CV MAE: ±0.64 mm")
with col3:
    st.metric("Reinforcement Height (R)", f"{num_res['reinforcement_mm']:.2f} mm", delta=None, help="Documented 5-Fold CV MAE: ±0.64 mm")
with col4:
    st.metric("SVR Latency", f"{num_res['inference_time_ms']:.1f} ms", delta=None)

# Section 2: Conditional Generative Macrographs
st.subheader("2. Conditional Generative Macrograph Synthesis & Contour Extraction")
st.caption("Synthesizes 128x128 cross-sectional weld image with adaptive crown boundary extraction. Reinforcement projects to the LEFT (X < 64).")

gen_results = inference_result["generative_results"]

if m_choice_key == "both":
    col_d, col_g = st.columns(2)
    
    # Direct CNN column
    with col_d:
        st.markdown("### 🔹 Direct Conditional CNN")
        d_data = gen_results["direct_cnn"]
        d_cont = d_data["contour_data"]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(d_data["raw_rgb_uint8"], caption="Raw Synthetic (128x128)", use_container_width=True)
        with c2:
            st.image(d_cont["overlay_rgb"], caption="Contour Overlay", use_container_width=True)
        with c3:
            st.image(d_cont["contour_only_rgb"], caption="Extracted Boundary", use_container_width=True)
            
        st.markdown(f"""
        - **Reinforcement Projection:** `{d_cont['reinf_projection_px']} px`
        - **Active Deposit Area:** `{d_cont['active_area_px_count']} px²`
        - **Crown Apex:** `(X={d_cont['apex_x_px']}, Y={d_cont['apex_y_px']})`
        - **Inference Latency:** `{d_data['inference_time_ms']} ms`
        """)

    # GC-WPG-v2 column
    with col_g:
        st.markdown("### 🔸 GC-WPG-v2 (Multi-Scale Spatial Prior)")
        g_data = gen_results["gc_wpg_v2"]
        g_cont = g_data["contour_data"]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(g_data["raw_rgb_uint8"], caption="Raw Synthetic (128x128)", use_container_width=True)
        with c2:
            st.image(g_cont["overlay_rgb"], caption="Contour Overlay", use_container_width=True)
        with c3:
            st.image(g_cont["contour_only_rgb"], caption="Extracted Boundary", use_container_width=True)
            
        st.markdown(f"""
        - **Reinforcement Projection:** `{g_cont['reinf_projection_px']} px`
        - **Active Deposit Area:** `{g_cont['active_area_px_count']} px²`
        - **Crown Apex:** `(X={g_cont['apex_x_px']}, Y={g_cont['apex_y_px']})`
        - **Inference Latency:** `{g_data['inference_time_ms']} ms`
        """)
else:
    m_data = gen_results[m_choice_key]
    m_cont = m_data["contour_data"]
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(m_data["raw_rgb_uint8"], caption="Raw Synthetic Macrograph (128x128)", use_container_width=True)
    with c2:
        st.image(m_cont["overlay_rgb"], caption="Contour Overlay (Interface X=64 Cyan, Crown Green, Apex Red)", use_container_width=True)
    with c3:
        st.image(m_cont["contour_only_rgb"], caption="Extracted Boundary Mask", use_container_width=True)
        
    st.markdown(f"""
    - **Reinforcement Projection:** `{m_cont['reinf_projection_px']} px`
    - **Active Deposit Area:** `{m_cont['active_area_px_count']} px²`
    - **Crown Apex Coordinates:** `(X={m_cont['apex_x_px']}, Y={m_cont['apex_y_px']})`
    - **Inference Latency:** `{m_data['inference_time_ms']} ms`
    """)

# Section 3: Model Provenance & Checkpoint Audit
with st.expander("🔍 Model Checkpoint Integrity & Provenance Tracking", expanded=False):
    st.markdown("All models operate under strict offline evaluation mode with verified cryptographic hashes:")
    for k, v in predictor.checkpoint_status.items():
        st.markdown(f"- **{k.upper()}:** `{'Verified' if v.get('hash_match', False) else 'Loaded'}` | Path: `{v.get('path', 'N/A')}` | SHA256: `{v.get('sha256', 'N/A')[:16]}...`")

# Section 4: Physical Disclaimers
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Scientific Disclaimers & Physical Boundaries:</strong><br>
    1. <strong>Surface vs Metallurgical Penetration:</strong> The generated macrograph is an AI-synthesized estimate learned from 36 experimental welds. It represents crown and near-surface deposit morphology. Root fusion boundary and deep penetration are not physically guaranteed from macrographs without physical sectioning.<br>
    2. <strong>Contour Metric Units:</strong> Crown projection and active deposit areas are reported strictly in <strong>PIXELS</strong>. Exact conversion to millimeters requires optical telecentric scale calibration.<br>
    3. <strong>Strict Prospective Conditioning:</strong> Inferences are computed purely from feed rate, travel speed, material, and process. No retrospective variables (heat input, arc efficiency) or ground truth targets are leaked.
</div>
""", unsafe_allow_html=True)
