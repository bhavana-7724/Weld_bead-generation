"""
build_reference_index.py

Phase V14: Stage 3 - Parameter-Space Reference Selection and Weighting.
Builds the parameter-space spatial index for each (Material, Process) welding domain.
Enforces that queries NEVER interpolate across materials or processes.
For any continuous (Feed Rate, Travel Speed) query:
- Detects if inside convex hull (legitimate interpolation) or outside (extrapolation).
- Finds exact bracketing anchor points (4-point bilinear grid cell or 3-point Delaunay simplex).
- Calculates exact normalized barycentric/bilinear weights summing to 1.000.
- Resolves anchor paths to registered macrographs (Views A and B), ground-truth masks, and physical SVR dimensions.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_CSV = os.path.join(DATA_DIR, "clean_dataset_36.csv")
OUT_BASE = os.path.join(BASE_DIR, "results", "phase_v14_sharp_interpolation")
REG_DIR = os.path.join(OUT_BASE, "registered_images")
GT_MASK_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit", "gt_masks")
REF_OUT_DIR = os.path.join(OUT_BASE, "reference_selection")

os.makedirs(REF_OUT_DIR, exist_ok=True)


class WeldReferenceIndex:
    def __init__(self, csv_path=CLEAN_CSV, reg_dir=REG_DIR, mask_dir=GT_MASK_DIR):
        self.df = pd.read_csv(csv_path)
        self.reg_dir = reg_dir
        self.mask_dir = mask_dir
        self.groups = {}
        self._build_index()

    def _build_index(self):
        """Constructs spatial KD/Delaunay structures per (material, process) pair."""
        for (mat, proc), group_df in self.df.groupby(['material', 'process']):
            coords = group_df[['feed_rate', 'speed']].values
            sample_ids = group_df['sample'].values.astype(int)
            
            # 2D Convex Hull
            hull = ConvexHull(coords)
            
            # Delaunay triangulation for 2D simplex interpolation
            tri = Delaunay(coords)
            
            self.groups[(mat, proc)] = {
                'df': group_df,
                'coords': coords,
                'sample_ids': sample_ids,
                'hull': hull,
                'tri': tri
            }

    def query(self, material, process, feed_rate, speed):
        """
        Queries the index for target parameter.
        Returns:
            dict containing:
            - is_valid_group (bool)
            - is_interpolation (bool)
            - anchors: list of dicts:
                [{'sample_id': int, 'feed_rate': float, 'speed': float, 'weight': float,
                  'view_a_path': str, 'view_b_path': str, 'mask_a_path': str, 'mask_b_path': str,
                  'width': float, 'depth': float, 'reinforcement': float}]
            - method: 'exact_match' | 'bilinear_grid' | 'barycentric_simplex' | 'nearest_extrapolation'
        """
        key = (material, process)
        if key not in self.groups:
            raise ValueError(f"Unknown material/process combination: {material} + {process}. "
                             f"Valid options: {list(self.groups.keys())}")
        
        grp = self.groups[key]
        coords = grp['coords']
        sample_ids = grp['sample_ids']
        df = grp['df']
        
        q_pt = np.array([float(feed_rate), float(speed)])
        
        # 1. Exact match check
        dists = np.linalg.norm(coords - q_pt, axis=1)
        exact_idx = np.where(dists < 1e-4)[0]
        if len(exact_idx) > 0:
            idx = exact_idx[0]
            sid = sample_ids[idx]
            row = df[df['sample'] == sid].iloc[0]
            return {
                'material': material,
                'process': process,
                'query_feed_rate': feed_rate,
                'query_speed': speed,
                'is_interpolation': True,
                'method': 'exact_match',
                'anchors': [self._package_anchor(sid, row, 1.0)]
            }
        
        # 2. Check if inside grid cell (Bilinear 4-point interpolation)
        unique_f = np.sort(np.unique(coords[:, 0]))
        unique_s = np.sort(np.unique(coords[:, 1]))
        
        # Find bracketing F and S
        f_left = unique_f[unique_f <= feed_rate]
        f_right = unique_f[unique_f >= feed_rate]
        s_left = unique_s[unique_s <= speed]
        s_right = unique_s[unique_s >= speed]
        
        is_in_grid_interior = (len(f_left) > 0 and len(f_right) > 0 and
                               len(s_left) > 0 and len(s_right) > 0)
        
        if is_in_grid_interior:
            f0, f1 = f_left[-1], f_right[0]
            s0, s1 = s_left[-1], s_right[0]
            
            # Check if we have 4 distinct corner points in this dataset
            c00 = (f0, s0) # Bottom-Left (or Top-Left depending on axis)
            c10 = (f1, s0)
            c01 = (f0, s1)
            c11 = (f1, s1)
            
            sids = {}
            for sid, f, s in zip(sample_ids, coords[:, 0], coords[:, 1]):
                if np.isclose(f, f0) and np.isclose(s, s0): sids['00'] = sid
                elif np.isclose(f, f1) and np.isclose(s, s0): sids['10'] = sid
                elif np.isclose(f, f0) and np.isclose(s, s1): sids['01'] = sid
                elif np.isclose(f, f1) and np.isclose(s, s1): sids['11'] = sid
            
            if len(sids) == 4 and f1 > f0 and s1 > s0:
                # Perfect 4-point rectilinear cell!
                u = (feed_rate - f0) / (f1 - f0)
                v = (speed - s0) / (s1 - s0)
                
                w00 = (1.0 - u) * (1.0 - v)
                w10 = u * (1.0 - v)
                w01 = (1.0 - u) * v
                w11 = u * v
                
                weights_dict = {
                    sids['00']: float(w00),
                    sids['10']: float(w10),
                    sids['01']: float(w01),
                    sids['11']: float(w11),
                }
                
                anchors = []
                for sid, w in weights_dict.items():
                    row = df[df['sample'] == sid].iloc[0]
                    anchors.append(self._package_anchor(sid, row, w))
                
                # Sort anchors by weight descending
                anchors.sort(key=lambda x: x['weight'], reverse=True)
                
                return {
                    'material': material,
                    'process': process,
                    'query_feed_rate': feed_rate,
                    'query_speed': speed,
                    'is_interpolation': True,
                    'method': 'bilinear_grid',
                    'anchors': anchors
                }
        
        # 3. Check Delaunay simplex (3-point barycentric interpolation)
        simplex_idx = grp['tri'].find_simplex(q_pt)
        if simplex_idx >= 0:
            # Point is inside a Delaunay triangle
            vertices = grp['tri'].simplices[simplex_idx]
            v_coords = coords[vertices]
            v_sids = sample_ids[vertices]
            
            # Compute barycentric coordinates
            T = grp['tri'].transform[simplex_idx]
            # T is (ndim+1, ndim), where T[:ndim, :] is inverse matrix and T[ndim, :] is origin
            diff = q_pt - T[2, :]
            bary_uv = np.dot(T[:2, :], diff)
            bary = np.array([bary_uv[0], bary_uv[1], 1.0 - np.sum(bary_uv)])
            
            anchors = []
            for sid, w in zip(v_sids, bary):
                row = df[df['sample'] == sid].iloc[0]
                anchors.append(self._package_anchor(sid, row, float(w)))
            
            anchors.sort(key=lambda x: x['weight'], reverse=True)
            return {
                'material': material,
                'process': process,
                'query_feed_rate': feed_rate,
                'query_speed': speed,
                'is_interpolation': True,
                'method': 'barycentric_simplex',
                'anchors': anchors
            }
        
        # 4. Extrapolation - Nearest Neighbors with Inverse Distance Weighting
        k = 3
        dists = np.linalg.norm(coords - q_pt, axis=1)
        nearest_indices = np.argsort(dists)[:k]
        nearest_dists = dists[nearest_indices]
        inv_dists = 1.0 / np.maximum(nearest_dists, 1e-4)
        weights = inv_dists / np.sum(inv_dists)
        
        anchors = []
        for idx, w in zip(nearest_indices, weights):
            sid = sample_ids[idx]
            row = df[df['sample'] == sid].iloc[0]
            anchors.append(self._package_anchor(sid, row, float(w)))
            
        return {
            'material': material,
            'process': process,
            'query_feed_rate': feed_rate,
            'query_speed': speed,
            'is_interpolation': False,
            'method': 'nearest_extrapolation',
            'anchors': anchors
        }

    def _package_anchor(self, sid, row, weight):
        sid_str = f"S{sid:02d}"
        
        # Paths to registered macrographs (check multiple common naming patterns)
        def find_file(candidates):
            for c in candidates:
                if os.path.exists(c):
                    return c
            return None

        view_a = find_file([
            os.path.join(self.reg_dir, f"{sid}-A_registered.png"),
            os.path.join(self.reg_dir, f"{sid_str}_view_A_registered.png"),
            os.path.join(self.reg_dir, f"{sid_str}_A_registered.png"),
        ])
        view_b = find_file([
            os.path.join(self.reg_dir, f"{sid}-B_registered.png"),
            os.path.join(self.reg_dir, f"{sid_str}_view_B_registered.png"),
            os.path.join(self.reg_dir, f"{sid_str}_B_registered.png"),
        ])
        
        # Ground truth / registered masks
        mask_a = find_file([
            os.path.join(self.reg_dir, f"{sid}-A_registered_mask.png"),
            os.path.join(self.mask_dir, f"{sid}-A.png"),
            os.path.join(self.mask_dir, f"{sid_str}_A_gt_mask.png"),
            os.path.join(self.mask_dir, f"{sid_str}_A.png"),
        ])
        mask_b = find_file([
            os.path.join(self.reg_dir, f"{sid}-B_registered_mask.png"),
            os.path.join(self.mask_dir, f"{sid}-B.png"),
            os.path.join(self.mask_dir, f"{sid_str}_B_gt_mask.png"),
            os.path.join(self.mask_dir, f"{sid_str}_B.png"),
        ])
        
        return {
            'sample_id': int(sid),
            'sample_name': sid_str,
            'feed_rate': float(row['feed_rate']),
            'speed': float(row['speed']),
            'weight': float(weight),
            'view_a_path': view_a if os.path.exists(view_a) else None,
            'view_b_path': view_b if os.path.exists(view_b) else None,
            'mask_a_path': mask_a if os.path.exists(mask_a) else None,
            'mask_b_path': mask_b if os.path.exists(mask_b) else None,
            'width_mm': float(row['width']),
            'depth_mm': float(row['depth']),
            'reinforcement_mm': float(row['reinforcement']),
        }


def test_reference_index():
    ref_idx = WeldReferenceIndex()
    
    # Test unseen condition from Phase V14 prompt:
    # Material: IN718, Process: ColdArc, Feed rate: 5.2, Speed: 26.0
    print("\n--- Testing Target Unseen Condition (IN718, ColdArc, F=5.2, S=26.0) ---")
    res = ref_idx.query("IN718", "ColdArc", 5.2, 26.0)
    print(f"Interpolation Valid: {res['is_interpolation']}")
    print(f"Method: {res['method']}")
    print("Selected Anchors:")
    total_w = 0.0
    for a in res['anchors']:
        print(f"  {a['sample_name']}: F={a['feed_rate']:.1f}, S={a['speed']:.1f} | Weight = {a['weight']:.4f} "
              f"| W={a['width_mm']:.2f}, D={a['depth_mm']:.2f}, R={a['reinforcement_mm']:.2f}")
        total_w += a['weight']
    print(f"Total Weight Sum: {total_w:.4f}")
    
    # Save target unseen query configuration
    unseen_path = os.path.join(REF_OUT_DIR, "unseen_query_IN718_ColdArc_F5.2_S26.0.json")
    with open(unseen_path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Saved unseen query config to: {unseen_path}")


if __name__ == "__main__":
    test_reference_index()
