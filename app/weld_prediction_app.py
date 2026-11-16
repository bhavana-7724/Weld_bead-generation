"""
weld_prediction_app.py

Native Desktop Dashboard for:
AI-Based Weld Bead Geometry Prediction and Cross-Section Generation.
Built with standard library tkinter and Pillow for 100% offline local execution on Windows CPU.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.inference_engine import WeldPredictor

class WeldApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI-Based Weld Bead Geometry Prediction and Cross-Section Generation")
        self.geometry("1280x880")
        self.minsize(1100, 750)
        self.configure(bg="#1e1e2f")

        self.predictor = WeldPredictor()
        self.current_results = None
        self.cached_tk_images = []

        self.setup_styles()
        self.create_widgets()
        self.run_inference() # Initial default prediction

    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background="#1e1e2f", foreground="#ffffff", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#00d2d3")
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#feca57")
        style.configure("Card.TFrame", background="#2d2d44", relief="flat")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), background="#00d2d3", foreground="#1e1e2f")

    def create_widgets(self):
        # Header banner
        header_frame = tk.Frame(self, bg="#27273f", height=60)
        header_frame.pack(fill="x", side="top", padx=10, pady=10)
        lbl_title = tk.Label(header_frame, text="AI-Based Weld Bead Geometry Prediction and Cross-Section Generation",
                             font=("Segoe UI", 16, "bold"), bg="#27273f", fg="#00d2d3")
        lbl_title.pack(side="left", padx=20, pady=10)

        lbl_sub = tk.Label(header_frame, text="Dual-Branch Framework: Prospective SVR Regression & Conditional Generation",
                           font=("Segoe UI", 10), bg="#27273f", fg="#a4b0be")
        lbl_sub.pack(side="right", padx=20, pady=10)

        # Main Layout: Sidebar (Left) and Results (Right)
        main_container = tk.Frame(self, bg="#1e1e2f")
        main_container.pack(fill="both", expand=True, padx=10, pady=5)

        # ----------------- SIDEBAR CONTROLS -----------------
        sidebar = tk.LabelFrame(main_container, text=" Welding Conditions & Model Controls ",
                               bg="#27273f", fg="#feca57", font=("Segoe UI", 11, "bold"), width=340)
        sidebar.pack(side="left", fill="y", padx=10, pady=5)
        sidebar.pack_propagate(False)

        # Material
        tk.Label(sidebar, text="Base Metal Material:", bg="#27273f", fg="#ffffff", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=15, pady=(15, 2))
        self.var_material = tk.StringVar(value="IN718")
        cb_mat = ttk.Combobox(sidebar, textvariable=self.var_material, values=["IN718", "SS304L"], state="readonly", font=("Segoe UI", 10))
        cb_mat.pack(fill="x", padx=15, pady=2)

        # Process
        tk.Label(sidebar, text="Welding Process:", bg="#27273f", fg="#ffffff", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=15, pady=(12, 2))
        self.var_process = tk.StringVar(value="ColdArc")
        cb_proc = ttk.Combobox(sidebar, textvariable=self.var_process, values=["ColdArc", "GMAW"], state="readonly", font=("Segoe UI", 10))
        cb_proc.pack(fill="x", padx=15, pady=2)

        # Feed Rate
        tk.Label(sidebar, text="Wire Feed Rate, F (m/min):", bg="#27273f", fg="#ffffff", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=15, pady=(12, 2))
        self.var_feed = tk.DoubleVar(value=5.0)
        lbl_feed_val = tk.Label(sidebar, text="5.00 m/min", bg="#27273f", fg="#00d2d3", font=("Segoe UI", 10, "bold"))
        lbl_feed_val.pack(anchor="e", padx=15)
        scale_feed = ttk.Scale(sidebar, from_=4.5, to=6.0, variable=self.var_feed, command=lambda v: lbl_feed_val.config(text=f"{float(v):.2f} m/min"))
        scale_feed.pack(fill="x", padx=15, pady=2)
        tk.Label(sidebar, text="Domain: 4.5 – 6.0 m/min", bg="#27273f", fg="#8395a7", font=("Segoe UI", 8)).pack(anchor="w", padx=15)

        # Speed
        tk.Label(sidebar, text="Travel Speed, S (cm/min):", bg="#27273f", fg="#ffffff", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=15, pady=(12, 2))
        self.var_speed = tk.DoubleVar(value=25.0)
        lbl_speed_val = tk.Label(sidebar, text="25.00 cm/min", bg="#27273f", fg="#00d2d3", font=("Segoe UI", 10, "bold"))
        lbl_speed_val.pack(anchor="e", padx=15)
        scale_speed = ttk.Scale(sidebar, from_=20.0, to=32.0, variable=self.var_speed, command=lambda v: lbl_speed_val.config(text=f"{float(v):.2f} cm/min"))
        scale_speed.pack(fill="x", padx=15, pady=2)
        tk.Label(sidebar, text="Domain: 20.0 – 32.0 cm/min", bg="#27273f", fg="#8395a7", font=("Segoe UI", 8)).pack(anchor="w", padx=15)

        # Model Selection
        tk.Label(sidebar, text="Image Generator Architecture:", bg="#27273f", fg="#ffffff", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=15, pady=(15, 2))
        self.var_model = tk.StringVar(value="both")
        rb1 = tk.Radiobutton(sidebar, text="Both Models (Comparative)", variable=self.var_model, value="both", bg="#27273f", fg="#ffffff", selectcolor="#1e1e2f", activebackground="#27273f")
        rb2 = tk.Radiobutton(sidebar, text="Direct Conditional CNN", variable=self.var_model, value="direct_cnn", bg="#27273f", fg="#ffffff", selectcolor="#1e1e2f", activebackground="#27273f")
        rb3 = tk.Radiobutton(sidebar, text="GC-WPG-v2 (Multi-Scale Prior)", variable=self.var_model, value="gc_wpg_v2", bg="#27273f", fg="#ffffff", selectcolor="#1e1e2f", activebackground="#27273f")
        rb1.pack(anchor="w", padx=15, pady=1)
        rb2.pack(anchor="w", padx=15, pady=1)
        rb3.pack(anchor="w", padx=15, pady=1)

        # Action Buttons
        btn_run = tk.Button(sidebar, text="Run Prediction & Generation", bg="#00d2d3", fg="#1e1e2f", font=("Segoe UI", 11, "bold"),
                            relief="raised", command=self.run_inference, cursor="hand2")
        btn_run.pack(fill="x", padx=15, pady=(25, 10))

        btn_save = tk.Button(sidebar, text="Save Results to Disk", bg="#576574", fg="#ffffff", font=("Segoe UI", 10),
                             relief="flat", command=self.save_results, cursor="hand2")
        btn_save.pack(fill="x", padx=15, pady=5)

        # Checkpoint Status Indicator
        status_box = tk.LabelFrame(sidebar, text=" Model Provenance & Hashes ", bg="#27273f", fg="#8395a7", font=("Segoe UI", 9))
        status_box.pack(fill="x", side="bottom", padx=10, pady=10)
        cnn_ok = self.predictor.checkpoint_status.get("direct_cnn", {}).get("loaded", False)
        v2_ok = self.predictor.checkpoint_status.get("gc_wpg_v2", {}).get("loaded", False)
        reg_ok = self.predictor.checkpoint_status.get("regressor", {}).get("loaded", False)
        
        tk.Label(status_box, text=f"Direct CNN: {'OK (SHA256 Match)' if cnn_ok else 'Missing'}", bg="#27273f", fg="#2ecc71" if cnn_ok else "#e74c3c", font=("Segoe UI", 8)).pack(anchor="w", padx=5)
        tk.Label(status_box, text=f"GC-WPG-v2: {'OK (SHA256 Match)' if v2_ok else 'Missing'}", bg="#27273f", fg="#2ecc71" if v2_ok else "#e74c3c", font=("Segoe UI", 8)).pack(anchor="w", padx=5)
        tk.Label(status_box, text=f"Numerical SVR: {'OK (5-Fold CV)' if reg_ok else 'Missing'}", bg="#27273f", fg="#2ecc71" if reg_ok else "#e74c3c", font=("Segoe UI", 8)).pack(anchor="w", padx=5)

        # ----------------- RESULTS PANEL -----------------
        results_panel = tk.Frame(main_container, bg="#1e1e2f")
        results_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Canvas + Scrollbar for vertical scrolling
        canvas = tk.Canvas(results_panel, bg="#1e1e2f", highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_panel, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg="#1e1e2f")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Sections inside scrollable frame
        self.frame_sec_a = tk.LabelFrame(self.scrollable_frame, text=" Section A: Selected Welding Conditions & Domain Status ", bg="#27273f", fg="#feca57", font=("Segoe UI", 11, "bold"))
        self.frame_sec_a.pack(fill="x", padx=10, pady=5)
        self.lbl_cond_summary = tk.Label(self.frame_sec_a, text="", bg="#27273f", fg="#ffffff", font=("Segoe UI", 10), justify="left")
        self.lbl_cond_summary.pack(anchor="w", padx=15, pady=8)

        self.frame_sec_b = tk.LabelFrame(self.scrollable_frame, text=" Section B: Prospective Numerical Geometry Prediction (SVR) ", bg="#27273f", fg="#feca57", font=("Segoe UI", 11, "bold"))
        self.frame_sec_b.pack(fill="x", padx=10, pady=5)
        self.lbl_num_summary = tk.Label(self.frame_sec_b, text="", bg="#27273f", fg="#ffffff", font=("Segoe UI", 10), justify="left")
        self.lbl_num_summary.pack(anchor="w", padx=15, pady=8)

        self.frame_sec_cd = tk.LabelFrame(self.scrollable_frame, text=" Sections C & D: Generated Macrograph Cross-Sections & External Contour Extraction ", bg="#27273f", fg="#feca57", font=("Segoe UI", 11, "bold"))
        self.frame_sec_cd.pack(fill="x", padx=10, pady=5)
        self.img_display_container = tk.Frame(self.frame_sec_cd, bg="#27273f")
        self.img_display_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.frame_sec_ef = tk.LabelFrame(self.scrollable_frame, text=" Section E & F: Model Provenance, Calibration & Scientific Limitations ", bg="#27273f", fg="#feca57", font=("Segoe UI", 11, "bold"))
        self.frame_sec_ef.pack(fill="x", padx=10, pady=5)
        self.lbl_limitations = tk.Label(self.frame_sec_ef, text="", bg="#27273f", fg="#dcdde1", font=("Segoe UI", 9), justify="left", wraplength=850)
        self.lbl_limitations.pack(anchor="w", padx=15, pady=8)

    def run_inference(self):
        mat = self.var_material.get()
        proc = self.var_process.get()
        feed = self.var_feed.get()
        spd = self.var_speed.get()
        m_choice = self.var_model.get()

        res = self.predictor.run_full_inference(mat, proc, feed, spd, model_choice=m_choice)
        self.current_results = res

        if not res["success"]:
            messagebox.showerror("Validation Error", "\n".join(res["errors"]))
            return

        # Update Section A
        ood_text = " [STATUS: WITHIN EXPERIMENTAL TRAINING DOMAIN]" if not res["is_out_of_domain"] else " [WARNING: OUT-OF-DOMAIN / EXTRAPOLATION]"
        ood_color = "#2ecc71" if not res["is_out_of_domain"] else "#ff6b6b"
        self.frame_sec_a.config(fg=ood_color)
        
        warn_str = ""
        if res["warnings"]:
            warn_str = "\nWarnings:\n" + "\n".join([f"  • {w}" for w in res["warnings"]])
            
        self.lbl_cond_summary.config(
            text=f"Material: {mat}  |  Process: {proc}  |  Wire Feed Rate: {feed:.2f} m/min  |  Travel Speed: {spd:.2f} cm/min{ood_text}{warn_str}"
        )

        # Update Section B
        num = res["numerical_predictions"]
        if num["available"]:
            self.lbl_num_summary.config(
                text=f"Model: {num['model_name']} (Trained on prospective inputs with nested 5-fold cross-validation)\n"
                     f"  • Predicted Bead Width, W:        {num['width_mm']:.2f} mm   (Documented 5-Fold CV MAE: {num['documented_cv_mae_mm']['width_mae']:.3f} mm)\n"
                     f"  • Predicted Penetration Depth, D:  {num['depth_mm']:.2f} mm   (Documented 5-Fold CV MAE: {num['documented_cv_mae_mm']['depth_mae']:.3f} mm)\n"
                     f"  • Predicted Reinforcement, R:     {num['reinforcement_mm']:.2f} mm   (Documented 5-Fold CV MAE: {num['documented_cv_mae_mm']['reinf_mae']:.3f} mm)\n"
                     f"  • Overall Validation MAE:          {num['documented_cv_mae_mm']['avg_mae']:.3f} mm  |  Inference Execution Time: {num['inference_time_ms']} ms"
            )
        else:
            self.lbl_num_summary.config(text=num["message"])

        # Update Section C & D
        for widget in self.img_display_container.winfo_children():
            widget.destroy()
        self.cached_tk_images = []

        models_res = res["generative_results"]
        for col_idx, (m_key, g) in enumerate(models_res.items()):
            m_card = tk.LabelFrame(self.img_display_container, text=f" {g['model_name']} ", bg="#1e1e2f", fg="#00d2d3", font=("Segoe UI", 10, "bold"))
            m_card.grid(row=0, column=col_idx, padx=15, pady=5, sticky="nsew")

            # Create side-by-side images: Raw Generated and Extracted Overlay
            img_row = tk.Frame(m_card, bg="#1e1e2f")
            img_row.pack(padx=5, pady=5)

            # Raw (upscaled nearest-neighbor from 128x128 to 220x220 for clear inspection)
            im_raw = Image.fromarray(g["raw_rgb_uint8"]).resize((220, 220), Image.Resampling.NEAREST)
            tk_raw = ImageTk.PhotoImage(im_raw)
            self.cached_tk_images.append(tk_raw)

            f_raw = tk.Frame(img_row, bg="#1e1e2f")
            f_raw.pack(side="left", padx=5)
            tk.Label(f_raw, text="1. Raw Generated Image\n(Direct Neural Output)", bg="#1e1e2f", fg="#a4b0be", font=("Segoe UI", 8, "bold")).pack()
            tk.Label(f_raw, image=tk_raw, bg="#000000").pack(pady=2)

            # Overlay
            cdata = g["contour_data"]
            im_ovr = Image.fromarray(cdata["overlay_rgb"]).resize((220, 220), Image.Resampling.NEAREST)
            tk_ovr = ImageTk.PhotoImage(im_ovr)
            self.cached_tk_images.append(tk_ovr)

            f_ovr = tk.Frame(img_row, bg="#1e1e2f")
            f_ovr.pack(side="left", padx=5)
            tk.Label(f_ovr, text="2. Extracted Crown Overlay\n(Cyan: Interface, Green: Crown)", bg="#1e1e2f", fg="#a4b0be", font=("Segoe UI", 8, "bold")).pack()
            tk.Label(f_ovr, image=tk_ovr, bg="#000000").pack(pady=2)

            # Contour Mask
            im_msk = Image.fromarray(cdata["contour_only_rgb"]).resize((220, 220), Image.Resampling.NEAREST)
            tk_msk = ImageTk.PhotoImage(im_msk)
            self.cached_tk_images.append(tk_msk)

            f_msk = tk.Frame(img_row, bg="#1e1e2f")
            f_msk.pack(side="left", padx=5)
            tk.Label(f_msk, text="3. Extracted Boundary Only\n(Isolated Contour)", bg="#1e1e2f", fg="#a4b0be", font=("Segoe UI", 8, "bold")).pack()
            tk.Label(f_msk, image=tk_msk, bg="#000000").pack(pady=2)

            # Metrics Label
            if cdata["is_failed"]:
                status_str = "EXTRACTION FAILED (Insufficient contrast)"
            else:
                status_str = f"Apex: ({cdata['apex_x_px']}, {cdata['apex_y_px']}) px  |  Projection: {cdata['reinf_projection_px']} pixels (Reference X=64)\n" \
                             f"Active Deposit Area: {cdata['active_area_px_count']} px^2  |  Continuity: {cdata['extracted_rows']}/91 rows detected\n" \
                             f"Adaptive Threshold: T = {cdata['adaptive_threshold']:.1f}  |  Inference Time: {g['inference_time_ms']} ms"

            lbl_info = tk.Label(m_card, text=status_str, bg="#1e1e2f", fg="#ffffff", font=("Segoe UI", 9), justify="center")
            lbl_info.pack(padx=10, pady=5)

        # Update Section E & F
        prov = res["provenance"]
        self.lbl_limitations.config(
            text=f"Provenance & Architecture Details:\n"
                 f"  • Execution Device: {prov['framework']}  |  Native Macrograph Resolution: {prov['image_resolution']}\n"
                 f"  • Optical Calibration: {prov['calibration_note']}\n"
                 f"  • Scientific Limitations Disclaimer: {prov['disclaimer']}"
        )

    def save_results(self):
        if not self.current_results or not self.current_results["success"]:
            messagebox.showwarning("No Results", "Please run a successful prediction before saving.")
            return

        save_dir = filedialog.askdirectory(title="Select Folder to Save Results")
        if not save_dir:
            return

        mat = self.current_results["inputs"]["material"]
        proc = self.current_results["inputs"]["process"]
        f = self.current_results["inputs"]["feed_rate_m_min"]
        s = self.current_results["inputs"]["speed_cm_min"]
        prefix = f"{mat}_{proc}_F{f:.1f}_S{s:.1f}"

        for m_key, g in self.current_results["generative_results"].items():
            raw_p = os.path.join(save_dir, f"{prefix}_{m_key}_raw.png")
            ovr_p = os.path.join(save_dir, f"{prefix}_{m_key}_overlay.png")
            msk_p = os.path.join(save_dir, f"{prefix}_{m_key}_contour.png")
            cv2.imwrite(raw_p, cv2.cvtColor(g["raw_rgb_uint8"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(ovr_p, cv2.cvtColor(g["contour_data"]["overlay_rgb"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(msk_p, cv2.cvtColor(g["contour_data"]["contour_only_rgb"], cv2.COLOR_RGB2BGR))

        json_p = os.path.join(save_dir, f"{prefix}_summary.json")
        import json
        save_dict = {
            "inputs": self.current_results["inputs"],
            "numerical_predictions": self.current_results["numerical_predictions"],
            "generative_summary": {
                m: {
                    "model_name": g["model_name"],
                    "apex_x_px": g["contour_data"]["apex_x_px"],
                    "reinf_projection_px": g["contour_data"]["reinf_projection_px"],
                    "active_area_px_count": g["contour_data"]["active_area_px_count"]
                } for m, g in self.current_results["generative_results"].items()
            },
            "provenance": self.current_results["provenance"]
        }
        with open(json_p, "w") as fp:
            json.dump(save_dict, fp, indent=2)

        messagebox.showinfo("Saved Successfully", f"All raw outputs, overlays, and JSON metadata were saved to:\n{save_dir}")

def main():
    app = WeldApp()
    app.mainloop()

if __name__ == "__main__":
    main()
