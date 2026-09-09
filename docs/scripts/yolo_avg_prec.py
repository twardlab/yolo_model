import numpy as np
from sklearn.metrics import auc
import time

def get_AP(gt_bboxes, pred_bboxes, conf_idx = -4, iou_thresh = 0.3, verbose = False):
    """
    Note: In the yolo_3d notebook, gt_bboxes and pred_bboxes are in their native format, so they must first be converted to lists prior to the AP calculation.
    Note: In the gt_ap notebook, gt_bboxes and pred_bboxes are already in list format
    """

    # # Convert inputs to lists
    # gt_bboxes = yolo_output_to_list(gt_bboxes)
    # gt_bboxes = gt_bbox_to_pred_bbox_format(gt_bboxes) # (cz, cy, cx, l, w, d) => (xmin, xmax, ymin, ymax, zmin, zmax)
    # pred_bboxes = yolo_output_to_list(pred_bboxes)
    
    # Sort all detections by confidence
    inds = np.argsort(pred_bboxes[..., conf_idx], axis = 0) # The sorted indeces from pred_bboxes
    pred_sorted = np.flip(pred_bboxes[inds], axis=0) # Note that inds sorts from lowest to highest, so flip is necessary to sort from highest to lowest
    
    # Compute pairwise IOU between all gt_bboxes and pred_bboxes
    pw_IOU = np.zeros((len(pred_sorted), len(gt_bboxes)))
    start = time.time()
    for i, gt_bb in enumerate(pred_sorted):
        for j, p_bb in enumerate(gt_bboxes):
            pw_IOU[i, j] = IOU_3D(gt_bb, p_bb)
        if verbose and i % 10000 == 0 and i > 0:
            print(f'Finished rows {i-9999}:{i} / {len(pred_sorted)} computation of pairwise IOU in {time.time()-start:.2f}s')
            start = time.time()

    # Assign each detection to a GT bbox + remove the GT bbox after pairing
    tpfp_out = []
    total_TP = 0
    total_FP = 0
    num_gt_remaining = len(gt_bboxes)
    for i, pred_i in enumerate(pred_sorted):
        best_gt_iou_idx = np.argmax(pw_IOU[i, :])
        best_gt_iou = pw_IOU[i, best_gt_iou_idx]
        pred_conf = pred_i[conf_idx]

        if best_gt_iou >= iou_thresh:
            total_TP += 1
            TP = 1
            FP = 0
            pw_IOU[:, best_gt_iou_idx] = 0.0 # 'Drop' the column corresponding to the gt_bbox with the greatest IOU with pred_i
            num_gt_remaining -= 1
        else:
            total_FP += 1
            TP = 0
            FP = 1

        precision = total_TP / (i+1) # Precision = num TP / all detections considered so far
        recall = total_TP / len(gt_bboxes) # Recall = num TP / all ground truths

        if verbose and i % 25000 == 0 and i > 0:
            print(f'i: {i:<5}, conf: {pred_conf:.3f}, iou: {best_gt_iou:.3f}, TP: {TP}, FP: {FP}')
       
        tpfp_out.append([i, best_gt_iou_idx, pred_conf, best_gt_iou, TP, FP, total_TP, total_FP, precision, recall])

        # All GT bboxes have been found, so all remaining bboxes are FP. Update the counts and terminate the loop.
        if num_gt_remaining == 0:
            # Note: Only the num of FP changes, so total_TP and recall remain unchanged
            total_FP_final = total_FP + (len(pred_sorted) - i - 1)
            precision_final = total_TP / len(pred_sorted)

            # Append one more row summarizing all remaining FPs
            interp = [i+1, -1, 0, 0, 0, 1, total_TP, total_FP_final, precision_final, recall]
            tpfp_out.append(interp)
            break
            
    tpfp_out = np.array(tpfp_out)
    AP = auc(tpfp_out[:,-1], tpfp_out[:,-2])
    
    return AP, tpfp_out

def gt_bbox_to_pred_bbox_format(gt_bbox, idx = [2,1,0,5,4,3]):
    """
    Converts gt_bbox from (cx, cy, cz, l, w, d) to (xmin, xmax, ymin, ymax, zmin, zmax). Note that in the simulated dataset
    """

    bb_out = []
    for bb in gt_bbox:
        # cx = bb[idx[0]]
        # cy = bb[idx[1]]
        # cz = bb[idx[2]]
        # l = bb[idx[3]]
        # w = bb[idx[4]]
        # d = bb[idx[5]]

        # xmin = cx
        # xmax = cx + l
        # ymin = cy
        # ymax = cy + w
        # zmin = cz
        # zmax = cz + d

        xmin = bb[idx[0]]
        ymin = bb[idx[1]]
        zmin = bb[idx[2]]
        xmax = xmin + bb[idx[3]]
        ymax = ymin + bb[idx[4]]
        zmax = zmin + bb[idx[5]]

        bb_out.append([xmin, xmax, ymin, ymax, zmin, zmax])

    return np.array(bb_out)

def IOU_3D(bb0, bb1):

    # Compute volume of each cube
    vol0 = abs((bb0[1] - bb0[0]) * (bb0[3] - bb0[2]) * (bb0[5] - bb0[4]))
    vol1 = abs((bb1[1] - bb1[0]) * (bb1[3] - bb1[2]) * (bb1[5] - bb1[4]))

    # Compute intersection volume between both cubes
    inter_xmin = max(bb0[0], bb1[0])
    inter_xmax = min(bb0[1], bb1[1])
    inter_ymin = max(bb0[2], bb1[2])
    inter_ymax = min(bb0[3], bb1[3])
    inter_zmin = max(bb0[4], bb1[4])
    inter_zmax = min(bb0[5], bb1[5])

    inter_dx = max(0, inter_xmax - inter_xmin)
    inter_dy = max(0, inter_ymax - inter_ymin)
    inter_dz = max(0, inter_zmax - inter_zmin)

    intersection = inter_dx * inter_dy * inter_dz
    union = vol0 + vol1 - intersection

    if union == 0.0:
        return 0.0
    
    return intersection / union

def yolo_output_to_list(bb_grid):

    if len(bb_grid.shape) == 2:
        return bb_grid
    elif len(bb_grid.shape) == 3:   
        bb_list = bb_grid.reshape(bb_grid.shape[0] * bb_grid.shape[1], bb_grid.shape[2])
    elif len(bb_grid.shape) == 4:   
        bb_list = bb_grid.reshape(bb_grid.shape[0] * bb_grid.shape[1] * bb_grid.shape[2], bb_grid.shape[3])
    else:
        raise Exception(f'bb_grid has an invalid shape of {bb_grid.shape}, should be of length 2, 3, or 4')
    
    return bb_list