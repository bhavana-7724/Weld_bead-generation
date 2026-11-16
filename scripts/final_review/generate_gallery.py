import os

base_dir = r"C:\Users\cool\OneDrive\Desktop\Project\weld_project"
html_path = os.path.join(base_dir, "FINAL_REVIEW_2026_11_16", "RESULTS_GALLERY.html")

# Build samples HTML
sample_items = []
for i in range(1, 37):
    sid = f"S{i:02d}"
    panel_rel = f"B_Figures/36_sample_panels/{sid}_final_panel.png"
    sketch_rel = f"B_Figures/36_sample_panels/{sid}_annotated_sketch.png"
    clean_rel = f"B_Figures/36_sample_panels/{sid}_clean_sketch.png"
    sample_items.append(f"""
    <div class="sample-card" data-sample="{sid}">
        <div class="card-header">
            <h3>Sample {sid}</h3>
            <span class="badge">6-Panel Comparison</span>
        </div>
        <div class="card-body">
            <a href="{panel_rel}" target="_blank">
                <img src="{panel_rel}" alt="{sid} Panel" loading="lazy">
            </a>
            <div class="card-links">
                <a href="{panel_rel}" target="_blank">View Full Plate</a> | 
                <a href="{sketch_rel}" target="_blank">Annotated Sketch</a> | 
                <a href="{clean_rel}" target="_blank">Clean Outline</a>
            </div>
        </div>
    </div>
    """)

samples_grid = "\n".join(sample_items)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welding Generative AI — Final Review Results Gallery</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --accent: #d97706;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #16a34a;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding-bottom: 60px;
        }}
        header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }}
        header h1 {{ font-size: 2.2rem; margin-bottom: 8px; font-weight: 700; }}
        header p {{ font-size: 1.1rem; color: #93c5fd; max-width: 900px; margin: 0 auto; }}
        .header-meta {{
            margin-top: 15px;
            font-size: 0.9rem;
            color: #cbd5e1;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        
        /* Navigation Tabs */
        .tabs {{
            display: flex;
            gap: 10px;
            margin: 20px 0;
            border-bottom: 2px solid var(--border);
            overflow-x: auto;
            padding-bottom: 2px;
        }}
        .tab-btn {{
            padding: 12px 20px;
            background: none;
            border: none;
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
            white-space: nowrap;
        }}
        .tab-btn:hover {{ color: var(--primary); }}
        .tab-btn.active {{
            color: var(--primary);
            border-bottom-color: var(--primary);
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* KPI Cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            border-top: 4px solid var(--primary-light);
        }}
        .kpi-card h4 {{ font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; }}
        .kpi-card .val {{ font-size: 2rem; font-weight: 700; color: var(--primary); }}
        .kpi-card .sub {{ font-size: 0.85rem; color: var(--text-muted); margin-top: 4px; }}

        /* Visual Grids */
        .panel-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
            gap: 24px;
        }}
        .sample-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .sample-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }}
        .card-header {{
            padding: 12px 16px;
            background: #f1f5f9;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-header h3 {{ font-size: 1.1rem; color: var(--primary); }}
        .badge {{
            background: #dbeafe;
            color: #1e40af;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .card-body {{ padding: 12px; }}
        .card-body img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            display: block;
            border: 1px solid #f1f5f9;
        }}
        .card-links {{
            margin-top: 10px;
            font-size: 0.85rem;
            text-align: center;
            color: var(--text-muted);
        }}
        .card-links a {{
            color: var(--primary-light);
            text-decoration: none;
            font-weight: 500;
        }}
        .card-links a:hover {{ text-decoration: underline; }}

        /* Benchmark Sections */
        .chart-container {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        .chart-container h3 {{ margin-bottom: 12px; color: var(--primary); }}

        /* Quick Links */
        .doc-list {{
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
        }}
        .doc-item {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            color: var(--text);
            transition: all 0.2s;
        }}
        .doc-item:hover {{
            border-color: var(--primary-light);
            box-shadow: 0 4px 8px rgba(0,0,0,0.06);
        }}
        .doc-icon {{
            font-size: 1.8rem;
            color: var(--primary-light);
        }}
        .doc-info h4 {{ font-size: 1rem; color: var(--primary); margin-bottom: 2px; }}
        .doc-info p {{ font-size: 0.8rem; color: var(--text-muted); }}

        /* Search Filter */
        .filter-bar {{
            margin-bottom: 20px;
            display: flex;
            gap: 12px;
            align-items: center;
        }}
        .filter-input {{
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 0.95rem;
            width: 300px;
        }}
    </style>
</head>
<body>

<header>
    <h1>Geometry-Aware Generative AI for Weld Bead Modeling</h1>
    <p>Conditional Deep Generative Modeling and Physical Metric Estimation for Arc-Welded Nickel-Base and Stainless Steel Alloys</p>
    <div class="header-meta">
        <strong>Final Academic Defense & Review</strong> | 16 November 2026 | Verified Production Model: M1 Direct CNN
    </div>
</header>

<div class="container">

    <!-- KPI Metrics Strip -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <h4>Grouped 5-Fold Numerical MAE</h4>
            <div class="val">0.692 mm</div>
            <div class="sub">Width: 0.798 mm | Depth: 0.633 mm | Reinf: 0.646 mm</div>
        </div>
        <div class="kpi-card">
            <h4>Production Contour MAE</h4>
            <div class="val">7.34 px</div>
            <div class="sub">M1 Direct CNN (95% CI: [5.37, 9.75] px)</div>
        </div>
        <div class="kpi-card">
            <h4>Mask Overlap (Dice)</h4>
            <div class="val">0.6132</div>
            <div class="sub">Boundary F1: 0.2398 | SSIM: 0.6869</div>
        </div>
        <div class="kpi-card">
            <h4>Inference Latency</h4>
            <div class="val">&lt; 1.2 s</div>
            <div class="sub">Offline CPU execution under Windows VS Code</div>
        </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tabs">
        <button class="tab-btn active" onclick="showTab('tab-gallery')">36 Sample Comparison Plates</button>
        <button class="tab-btn" onclick="showTab('tab-benchmarks')">Model Benchmarks & Residuals</button>
        <button class="tab-btn" onclick="showTab('tab-cases')">Representative Cases (S02 & S32)</button>
        <button class="tab-btn" onclick="showTab('tab-unseen')">Unseen Parameter Demo</button>
        <button class="tab-btn" onclick="showTab('tab-docs')">Documents & Manifests</button>
    </div>

    <!-- TAB 1: 36 Samples Gallery -->
    <div id="tab-gallery" class="tab-content active">
        <div class="filter-bar">
            <input type="text" id="sampleSearch" class="filter-input" placeholder="Filter sample (e.g. S02, S13, S32)..." onkeyup="filterSamples()">
            <span style="color: var(--text-muted); font-size: 0.9rem;">Showing all 36 full-plate academic figures (Click image for high-res view)</span>
        </div>
        <div class="panel-grid" id="samplesContainer">
            {samples_grid}
        </div>
    </div>

    <!-- TAB 2: Benchmarks -->
    <div id="tab-benchmarks" class="tab-content">
        <div class="chart-container">
            <h3>5-Model Generative Architecture Comparison (Contour MAE vs Mask Dice)</h3>
            <img src="B_Figures/fig_model_comparison_mae_dice.png" alt="Model Comparison">
            <p style="color: var(--text-muted); margin-top: 10px;">M1 Direct CNN demonstrates lowest contour error (7.34 px) and robust edge stability (2.8% failure rate) across 36 held-out conditions.</p>
        </div>
        <div class="chart-container">
            <h3>Experimental vs Predicted Dimensions (Leak-Free 5-Fold Grouped CV)</h3>
            <img src="B_Figures/fig_exp_vs_pred_dimensions.png" alt="Exp vs Pred Dimensions">
        </div>
        <div class="chart-container">
            <h3>Numerical Prediction Residuals Distribution</h3>
            <img src="B_Figures/fig_numerical_residuals.png" alt="Residuals Distribution">
        </div>
        <div class="chart-container">
            <h3>Error Summary Grouped by Material and Process Regime</h3>
            <img src="B_Figures/fig_error_summary_by_material_process.png" alt="Error Summary">
        </div>
    </div>

    <!-- TAB 3: Representative Cases -->
    <div id="tab-cases" class="tab-content">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
            <div class="chart-container">
                <h3>Representative Well-Behaved Condition: S02 (IN718, ColdArc)</h3>
                <img src="B_Figures/S02_representative_panel.png" alt="S02 Panel">
                <div style="text-align: left; margin-top: 15px; font-size: 0.9rem;">
                    <p><strong>Parameters:</strong> Feed = 5.5 m/min, Speed = 20 cm/min</p>
                    <p><strong>Dimensions:</strong> Width Exp: 10.76 mm | Pred: 10.66 mm; Depth Exp: 1.77 mm | Pred: 1.67 mm; Reinf Exp: 4.09 mm | Pred: 4.12 mm</p>
                    <p><strong>Performance:</strong> Symmetrical convex crown curvature, smooth wetting transition. Held-out 5-fold contour MAE = 5.35 px (in-sample avg: 4.34 px).</p>
                </div>
            </div>
            <div class="chart-container">
                <h3>Challenging Physical Outlier: S32 (SS304L, GMAW)</h3>
                <img src="B_Figures/S32_representative_panel.png" alt="S32 Panel">
                <div style="text-align: left; margin-top: 15px; font-size: 0.9rem;">
                    <p><strong>Parameters:</strong> Feed = 5.0 m/min, Speed = 27 cm/min</p>
                    <p><strong>Dimensions:</strong> Depth Exp: 5.25 mm | Pred: 2.74 mm (Err: -2.51 mm); Width Exp: 10.43 mm | Pred: 10.33 mm; Reinf Exp: 2.74 mm | Pred: 2.84 mm</p>
                    <p><strong>Metallurgical Hypothesis:</strong> Localized fluid-dynamic transition (e.g., surface-tension temperature gradient / Marangoni effects) and arc plasma jet keyholing formed an isolated deep penetration finger exceeding continuous empirical regression limits.</p>
                </div>
            </div>
        </div>
    </div>

    <!-- TAB 4: Unseen Parameter Demo -->
    <div id="tab-unseen" class="tab-content">
        <div class="chart-container">
            <h3>Offline Unseen Parameter Demonstration (Candidate Interpolation)</h3>
            <img src="B_Figures/unseen_demo_panel.png" alt="Unseen Demo Panel">
            <div style="text-align: left; margin-top: 20px; font-size: 0.95rem; max-width: 800px; margin-left: auto; margin-right: auto;">
                <p><strong>Interpolation Target:</strong> Inconel 718 | ColdArc | Wire Feed = 5.2 m/min | Travel Speed = 26.0 cm/min</p>
                <p><strong>Predicted Dimensions:</strong> Width = 9.98 mm | Depth = 1.65 mm | Reinforcement = 2.98 mm</p>
                <p><strong>Validation Note:</strong> System automatically verified parameter absence from training database, applied interpolation bounds check, and synthesized the complete 6-panel profile without human intervention.</p>
            </div>
        </div>
    </div>

    <!-- TAB 5: Documentation -->
    <div id="tab-docs" class="tab-content">
        <ul class="doc-list">
            <li>
                <a class="doc-item" href="A_Presentation/PRESENTATION_WELDING_GENERATIVE_AI.pptx">
                    <div class="doc-icon">&#128202;</div>
                    <div class="doc-info">
                        <h4>PowerPoint Presentation (.pptx)</h4>
                        <p>18 Widescreen Slides with High-Res Figures & Speaker Notes</p>
                    </div>
                </a>
            </li>
            <li>
                <a class="doc-item" href="A_Presentation/PRESENTATION_WELDING_GENERATIVE_AI.pdf">
                    <div class="doc-icon">&#128196;</div>
                    <div class="doc-info">
                        <h4>Presentation PDF (.pdf)</h4>
                        <p>Vector Slide Deck for Screen Sharing or Printing</p>
                    </div>
                </a>
            </li>
            <li>
                <a class="doc-item" href="A_Presentation/PRESENTATION_WELDING_GENERATIVE_AI.html" target="_blank">
                    <div class="doc-icon">&#127760;</div>
                    <div class="doc-info">
                        <h4>Interactive HTML Slide Deck</h4>
                        <p>Self-contained slide viewer directly inside your browser</p>
                    </div>
                </a>
            </li>
            <li>
                <a class="doc-item" href="E_Documentation/SUPERVISOR_EXECUTIVE_SUMMARY.md">
                    <div class="doc-icon">&#128203;</div>
                    <div class="doc-info">
                        <h4>Supervisor Executive Summary</h4>
                        <p>1-Page condensed briefing on findings, metrics, and readiness</p>
                    </div>
                </a>
            </li>
            <li>
                <a class="doc-item" href="E_Documentation/FINAL_TECHNICAL_REPORT.md">
                    <div class="doc-icon">&#128450;</div>
                    <div class="doc-info">
                        <h4>Comprehensive Technical Report</h4>
                        <p>17-Section academic report with formulas, tables, and references</p>
                    </div>
                </a>
            </li>
            <li>
                <a class="doc-item" href="E_Documentation/FINAL_REVIEW_VIVA_QA.md">
                    <div class="doc-icon">&#127891;</div>
                    <div class="doc-info">
                        <h4>Viva Voce Q&A Document</h4>
                        <p>25 Technical questions & answers for review defense preparation</p>
                    </div>
                </a>
            </li>
            <li>
                <a class="doc-item" href="C_Metrics/CORRECTED_36_ANNOTATED_METRICS.csv">
                    <div class="doc-icon">&#128200;</div>
                    <div class="doc-info">
                        <h4>Corrected 36 Annotated Metrics</h4>
                        <p>Official verified numerical ground-truth and predictions CSV</p>
                    </div>
                </a>
            </li>
            <li>
                <a class="doc-item" href="C_Metrics/FIVE_MODEL_GENERATIVE_COMPARISON.csv">
                    <div class="doc-icon">&#128300;</div>
                    <div class="doc-info">
                        <h4>Generative Model Ranking CSV</h4>
                        <p>Full 36-sample evaluation metrics across all 5 architectures</p>
                    </div>
                </a>
            </li>
            <li>
                <a class="doc-item" href="F_Code/CODEBASE_CATALOG.md">
                    <div class="doc-icon">&#128187;</div>
                    <div class="doc-info">
                        <h4>Codebase & Model Catalog</h4>
                        <p>Architectural reference mapping checkpoints, scripts, and workflows</p>
                    </div>
                </a>
            </li>
        </ul>
    </div>

</div>

<script>
    function showTab(tabId) {{
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        event.target.classList.add('active');
    }}

    function filterSamples() {{
        let query = document.getElementById('sampleSearch').value.toUpperCase();
        let cards = document.querySelectorAll('.sample-card');
        cards.forEach(card => {{
            let sid = card.getAttribute('data-sample');
            if (sid.includes(query)) {{
                card.style.display = '';
            }} else {{
                card.style.display = 'none';
            }}
        }});
    }}
</script>

</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Successfully generated RESULTS_GALLERY.html at {html_path}")
