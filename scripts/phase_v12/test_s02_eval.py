import sys
import os
sys.path.insert(0, 'weld_project')
import cv2
import numpy as np
import pandas as pd
from scripts.phase_v12.audit_segmentation_methods import get_gt_contour_from_mask, compute_toe_errors, extract_method3_toe_bounded
from scripts.phase_v12.run_model_36_panels import evaluate_metrics, compute_clean_canny

v9_gen_dir = 'weld_project/results/phase_v9_geometry_aware/generated_36'
gt_mask_dir = 'weld_project/results/phase_v9_geometry_aware/audit/gt_masks'

real_a = cv2.cvtColor(cv2.imread(os.path.join(v9_gen_dir, 'S02', 'Real_A.png')), cv2.COLOR_BGR2RGB)
real_b = cv2.cvtColor(cv2.imread(os.path.join(v9_gen_dir, 'S02', 'Real_B.png')), cv2.COLOR_BGR2RGB)
gen_rgb = cv2.cvtColor(cv2.imread(os.path.join(v9_gen_dir, 'S02', 'M1_DirectCNN_raw.png')), cv2.COLOR_BGR2RGB)
gen_gray = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY)

gt_a = cv2.imread(os.path.join(gt_mask_dir, '2-A.png'), cv2.IMREAD_GRAYSCALE)
gt_b = cv2.imread(os.path.join(gt_mask_dir, '2-B.png'), cv2.IMREAD_GRAYSCALE)
cnt_a = get_gt_contour_from_mask(gt_a)
cnt_b = get_gt_contour_from_mask(gt_b)

canny_a = compute_clean_canny(cv2.cvtColor(real_a, cv2.COLOR_RGB2GRAY))
canny_b = compute_clean_canny(cv2.cvtColor(real_b, cv2.COLOR_RGB2GRAY))
canny_gen = compute_clean_canny(gen_gray)

cnt_pred = extract_method3_toe_bounded(gen_gray)
cnt_pred['pts'] = {int(y): int(round(x)) for y, x in cnt_pred['pts'].items()}

mA = evaluate_metrics(real_a, gen_rgb, cnt_a, cnt_pred, canny_a, canny_gen)
mB = evaluate_metrics(real_b, gen_rgb, cnt_b, cnt_pred, canny_b, canny_gen)
top_err_a, bot_err_a = compute_toe_errors(cnt_a, cnt_pred)
top_err_b, bot_err_b = compute_toe_errors(cnt_b, cnt_pred)

print('S02 Evaluation vs View A:')
print(f"  MAE: {mA['contour_mae_px']:.2f} px | F1: {mA['boundary_f1_2px']:.3f} | Dice: {mA['mask_dice']:.3f} | IoU: {mA['mask_iou']:.3f} | Apex Err: {mA['apex_projection_error_px']:.2f} px | Top Toe Err: {top_err_a:.2f} px | Bot Toe Err: {bot_err_a:.2f} px")

print('S02 Evaluation vs View B:')
print(f"  MAE: {mB['contour_mae_px']:.2f} px | F1: {mB['boundary_f1_2px']:.3f} | Dice: {mB['mask_dice']:.3f} | IoU: {mB['mask_iou']:.3f} | Apex Err: {mB['apex_projection_error_px']:.2f} px | Top Toe Err: {top_err_b:.2f} px | Bot Toe Err: {bot_err_b:.2f} px")

print('S02 Dual-View Average:')
print(f"  MAE: {0.5*(mA['contour_mae_px']+mB['contour_mae_px']):.2f} px | F1: {0.5*(mA['boundary_f1_2px']+mB['boundary_f1_2px']):.3f} | Dice: {0.5*(mA['mask_dice']+mB['mask_dice']):.3f} | IoU: {0.5*(mA['mask_iou']+mB['mask_iou']):.3f} | Apex Err: {0.5*(mA['apex_projection_error_px']+mB['apex_projection_error_px']):.2f} px")
