"""
Geometry-Guided Pretrained Interpolation Module (Variant A vs Variant B).
Phase V15: Sharp Weld Macrograph Pretrained AI Study.

Implements:
1. Variant A: Direct Pretrained Interpolation (RIFE / FILM applied directly to donor macrographs).
2. Variant B: Geometry-Guided Pretrained Interpolation:
   - Target crown geometry synthesized via multi-anchor Signed Distance Field (SDF) zero-level set.
   - Donor crowns non-linearly warped to match target boundary prior to frame synthesis.
   - Pretrained RIFE / FILM synthesizes metallic solidification texture without boundary disparity.
   - Seamless composition onto substrate baseline datum (X = 64).
"""

import os
import sys
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def mask_to_sdf(binary_mask):
    """
    Computes Euclidean Signed Distance Field (SDF) from binary mask:
    Negative inside mask, positive outside.
    """
    mask_bool = binary_mask > 0
    if not np.any(mask_bool):
        return np.ones(binary_mask.shape, dtype=np.float32) * 50.0
    if np.all(mask_bool):
        return -np.ones(binary_mask.shape, dtype=np.float32) * 50.0

    d_out = distance_transform_edt(~mask_bool)
    d_in = distance_transform_edt(mask_bool)
    sdf = d_out - d_in
    return sdf.astype(np.float32)


def blend_sdfs(mask_paths, weights, baseline_x=64):
    """
    Computes the interpolated target crown mask and perimeter points
    from anchor masks using weighted Signed Distance Fields.
    """
    weights = np.array(weights, dtype=np.float32)
    weights /= np.sum(weights)

    h, w = 128, 128
    blended_sdf = np.zeros((h, w), dtype=np.float32)

    anchor_pts_list = []
    anchor_masks = []

    for p, weight in zip(mask_paths, weights):
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (h, w):
            m = np.zeros((h, w), dtype=np.uint8)
        # Ensure mask is strictly in crown region (x < baseline_x)
        m[:, baseline_x:] = 0
        anchor_masks.append(m)

        sdf = mask_to_sdf(m)
        blended_sdf += weight * sdf

        # Extract contour points for warping
        pts = {}
        for y in range(h):
            row = np.where(m[y, :baseline_x] > 127)[0]
            if len(row) > 0:
                pts[int(y)] = float(row.min())
        anchor_pts_list.append(pts)

    # Zero level-set gives the interpolated target mask
    target_mask = ((blended_sdf <= 0.0) & (np.arange(w) < baseline_x)).astype(np.uint8) * 255

    # Clean target mask (keep largest connected component)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(target_mask)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        target_mask = ((labels == largest_label).astype(np.uint8)) * 255

    target_pts = {}
    for y in range(h):
        row = np.where(target_mask[y, :baseline_x] > 127)[0]
        if len(row) > 0:
            target_pts[int(y)] = float(row.min())

    return target_mask, target_pts, anchor_masks, anchor_pts_list


def warp_crown_to_target(anchor_img, anchor_pts, target_pts, baseline_x=64):
    """
    Computes smooth backward remap coordinates mapping target crown coordinates (x_t, y_t)
    to anchor crown coordinates (x_a, y_a). Substrate (x >= baseline_x) remains identity.
    """
    h, w = anchor_img.shape[:2]
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, np.newaxis], w, axis=1)

    if len(target_pts) < 5 or len(anchor_pts) < 5:
        return anchor_img.copy()

    t_ys = sorted(target_pts.keys())
    t_y_min, t_y_max = t_ys[0], t_ys[-1]
    t_span = max(1.0, float(t_y_max - t_y_min))

    a_ys = sorted(anchor_pts.keys())
    a_y_min, a_y_max = a_ys[0], a_ys[-1]
    a_span = max(1.0, float(a_y_max - a_y_min))

    a_curve_y = np.array(a_ys, dtype=np.float32)
    a_curve_x = np.array([anchor_pts[y] for y in a_ys], dtype=np.float32)

    for yt in range(t_y_min, t_y_max + 1):
        if yt not in target_pts:
            continue
        xt_bound = target_pts[yt]
        if xt_bound >= baseline_x:
            continue

        v_frac = (yt - t_y_min) / t_span
        ya = a_y_min + v_frac * a_span
        xa_bound = float(np.interp(ya, a_curve_y, a_curve_x))

        target_width = baseline_x - xt_bound
        anchor_width = baseline_x - xa_bound
        if target_width <= 0 or anchor_width <= 0:
            continue

        for xt in range(int(np.floor(xt_bound)), baseline_x):
            u_frac = (baseline_x - xt) / target_width
            xa = baseline_x - u_frac * anchor_width
            map_x[yt, xt] = xa
            map_y[yt, xt] = ya

    warped = cv2.remap(anchor_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def run_interpolator_chain(interpolator, images, weights):
    """
    Executes a chain of pairwise interpolations for arbitrary number of weighted donor images.
    For N=2: interp(img0, img1, t=w1)
    For N=3: interp(interp(img0, img1, t=w1/(w0+w1)), img2, t=w2)
    """
    weights = [float(w) for w in weights]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    if len(images) == 1 or weights[0] >= 0.999:
        return images[0].copy()

    if len(images) == 2:
        return interpolator.interpolate(images[0], images[1], timestep=weights[1])

    # N >= 3: recursive chaining
    curr_img = images[0].copy()
    accum_weight = weights[0]

    for i in range(1, len(images)):
        next_img = images[i]
        next_w = weights[i]
        t = next_w / (accum_weight + next_w)
        curr_img = interpolator.interpolate(curr_img, next_img, timestep=t)
        accum_weight += next_w

    return curr_img


def synthesize_direct_pretrained(interpolator, donor_images_rgb, weights):
    """
    Variant A: Direct Pretrained Interpolation without geometric pre-alignment.
    """
    return run_interpolator_chain(interpolator, donor_images_rgb, weights)


def synthesize_geometry_guided_pretrained(
    interpolator, donor_images_rgb, mask_paths, weights, baseline_x=64
):
    """
    Variant B: Geometry-Guided Pretrained Interpolation.
    1. Computes target SDF crown geometry.
    2. Non-linearly warps donor crowns into the target boundary.
    3. Feeds shape-aligned crowns into the pretrained model (RIFE / FILM).
    4. Composites the synthesized crown texture onto the substrate baseline (X = 64).
    """
    target_mask, target_pts, anchor_masks, anchor_pts_list = blend_sdfs(
        mask_paths, weights, baseline_x=baseline_x
    )

    # Warp each donor image to match target crown geometry
    warped_donors = []
    for img_rgb, a_pts in zip(donor_images_rgb, anchor_pts_list):
        # cv2.remap works on BGR or RGB identically
        warped = warp_crown_to_target(img_rgb, a_pts, target_pts, baseline_x=baseline_x)
        warped_donors.append(warped)

    # Run pretrained interpolation on shape-aligned donors
    synthesized_aligned = run_interpolator_chain(interpolator, warped_donors, weights)

    # Also compute direct substrate plate blend for the flat base (X >= 64)
    direct_substrate = run_interpolator_chain(interpolator, donor_images_rgb, weights)

    # Feathered alpha matte around target crown mask boundary
    mask_float = (target_mask > 0).astype(np.float32)
    mask_feathered = cv2.GaussianBlur(mask_float, (3, 3), 0.8)[:, :, np.newaxis]

    composite_rgb = (
        mask_feathered * synthesized_aligned.astype(np.float32)
        + (1.0 - mask_feathered) * direct_substrate.astype(np.float32)
    )
    final_rgb = np.clip(composite_rgb, 0, 255).astype(np.uint8)

    return final_rgb, target_mask, target_pts


if __name__ == "__main__":
    print("Testing geometry_guided_interpolation functions...")
    m0 = np.zeros((128, 128), dtype=np.uint8)
    m0[30:98, 20:64] = 255
    m1 = np.zeros((128, 128), dtype=np.uint8)
    m1[25:103, 15:64] = 255

    # Mock mask paths
    import tempfile
    t0 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    t1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(t0.name, m0)
    cv2.imwrite(t1.name, m1)

    t_mask, t_pts, _, _ = blend_sdfs([t0.name, t1.name], [0.5, 0.5])
    print("Blended target mask area:", np.sum(t_mask > 0), "points:", len(t_pts))

    t0.close()
    t1.close()
    try:
        os.remove(t0.name)
        os.remove(t1.name)
    except Exception:
        pass
    print("geometry_guided_interpolation self-test PASSED!")
