# Dataset Structure & Technical Specification

This directory documents the experimental data schema and metallographic coordinate system for the **Weld Bead Geometry Prediction** framework.

---

## 1. Experimental Overview

The empirical study is grounded in physical robotic welding experiments conducted under strictly controlled laboratory conditions:
- **Total Physical Specimens:** 36 independent weld trials.
- **Metallographic Views:** 72 calibrated transverse macrographs (Dual views per specimen: View A and View B).
- **Coordinate Space:** Standardized $128 \times 128 \times 3$ RGB spatial coordinate frame.
- **Reference Baseline:** Flat substrate boundary fixed at $X = 64$ ($Y \in [0, 127]$). Weld crown protrudes outward to the left ($X < 64$), and weld root penetration extends to the right ($X > 64$).

---

## 2. Parameter Matrix & Feature Schema

Each experimental record contains 4 primary process parameters (inputs) and 3 physical transverse dimensions (ground-truth targets):

| Feature Name | Type | Units | Valid Domain | Physical Description |
|---|---|---|---|---|
| `sample` | Integer | - | 1 to 36 | Experimental specimen index ($S01$ to $S36$) |
| `material` | Categorical | - | `IN718`, `SS304L` | Aerospace superalloy (Inconel 718) or stainless steel (SS304L) |
| `process` | Categorical | - | `ColdArc`, `GMAW` | Controlled dip transfer (ColdArc) or continuous gas metal arc (GMAW) |
| `feed_rate` | Float | m/min | 4.0 to 6.5 | Wire feeder delivery speed ($F$) |
| `speed` | Float | cm/min | 20.0 to 40.0 | Robotic welding torch travel speed ($S$) |
| `HIP` | Float | J/mm | 90.0 to 260.0 | Arc Heat Input Parameter: $HIP = \eta \cdot \frac{U \cdot I}{v}$ |
| `width` | Float | mm | 4.2 to 12.5 | Transverse weld bead width ($W$) |
| `depth` | Float | mm | 0.8 to 2.8 | Root penetration depth below substrate surface ($D$) |
| `reinforcement`| Float | mm | 1.8 to 4.5 | Crown reinforcement height above substrate surface ($R$) |

---

## 3. Sample Format Template

A sample template is provided in [`sample_parameters_template.csv`](sample_parameters_template.csv):

```csv
sample,material,process,feed_rate,speed,HIP,efficiency,width,depth,reinforcement
1,IN718,ColdArc,5.0,20.0,139.08,83.45,4.52,1.46,3.79
2,IN718,ColdArc,5.5,20.0,154.20,83.45,10.76,1.77,4.09
3,IN718,ColdArc,5.0,25.0,111.26,83.45,4.88,1.22,3.31
```

---

## 4. Data Confidentiality & Access Protocol

> **Institutional Notice:**  
> In compliance with institutional research agreements and proprietary manufacturing guidelines, raw experimental macrograph image arrays and complete industrial metallurgical matrices are excluded from the public git repository. 
> 
> Researchers seeking full access to raw high-resolution metallographic images for academic benchmarking may request authorized access by contacting the laboratory repository maintainers.
