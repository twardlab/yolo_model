from yolo_help import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import trapezoid



def NMS(bboxes, scores, nms_threshold = 0.5):
    """
    Perform non-maximum suppression on the outputs from the yolo model framework in order to reduce the number of candidate bounding boxes.

    Parameters:
    -----------
    bboxes : torch.Tensor of size [N, 4]
        N corresponds to the number of bounding boxes
    scores : torch.Tensor of size [N,4]
        N corresponds to the number of bboxes; The confidence scores corresponding to the
    nms_threshold : float
        The iou threshold used to remove candidate bounding boxes when comparing their spatial position with respect to the bounding box with the highest confidence during the current iteration

    Returns:
    --------
    bboxes_out : torch.Tensor of size [M,4]
        M corresponds to the number of bounding boxes remaining after NMS
    scores_out : torch .Tensor of size [M,1]
        The M scores corresponding to the bounding boxes defined in bboxes_out
    """
    # Sort the bboxes and scores in descending order based on score
    bboxes_nms = [b for _ , b in sorted(zip(scores, bboxes), reverse = True)]
    scores_nms = sorted(scores, reverse = True)

    bboxes_out = []
    scores_out = []

    # Until every bbox has been checked
    while len(bboxes_nms) > 0:

        for b_max, s_max in zip(bboxes_nms, scores_nms):
            # Append bbox w highest scores to output list + remove from input list
            bboxes_out.append(b_max)
            scores_out.append(s_max)

            # Remove bbox if iou w b_max > nms_threshold
            for b,s in zip(bboxes_nms, scores_nms):
                if iou(b_max, b) > nms_threshold:
                    scores_nms.remove(s)
                    del b
        break
    
    return bboxes_out, scores_out



def compute_pr_curves(gt_bboxes, pred_bboxes, pred_scores, verbose = False):
    """
    Given a set of ground truth bounding boxes, a set of predicted bounding boxes, and the corresponding confidences for each predicted bounding box, generate a set of pr curves quantifying the performance of the model. One PR curve will be generated at each IOU threshold from 0.5 : 0.05 : 1.0 and contain 101 points corresponding to each confidence threshold from 0.01 : 0.01 : 1.0 and 2 predefined endpoints.

    Parameters:
    -----------
    gt_bboxes : torch.Tensor of shape [N,4]
        The ground truth bounding boxes which correspond to the target objects in the original image.
    pred_bboxes : torch.Tensor of shape [M,4]
        The filtered bounding boxes which were output from the model 
    pred_scores : torch.Tensor of shape [M,1]
        The confidence scores corresponding to the bounding boxes in the parameter 'pred_boxes'
    verbose : bool
        Default - False; If True, print out the (tp, fp, fn) 3-tuple for every (iou, conf) threshold
        
    Returns:
    --------
    all_pr_curves : list of shape [10,101,3]
        Each element of this list is defined by 101 3-tuples which correspond to the (precision, recall, confidence) values at the given thresholds (iou_thresh, conf_thresh).
    """

    iou_threshold  = [round(x,2) for x in list(np.arange(0.5,1.0,0.05))]
    conf_threshold = [round(x,2) for x in list(np.arange(0.01,1.0,0.01))]
    
    all_pr_curves = []
    for iou_thresh in iou_threshold:
        
        if verbose:
            print(f'===== Starting iou {iou_thresh} =====')
            
        curr_pr_curve = [[0.0, 1.0, -1]]
        for conf_thresh in conf_threshold:
            
            if verbose:
                print(f'Starting conf {conf_thresh}')
                
            bbox = gt_bboxes.copy().tolist()        
            
            tp = 0
            fp = 0
            for bb0, s in zip(pred_bboxes, pred_scores):
                if s < conf_thresh:
                    fp += 1
                    continue
                
                for bb1 in bbox:
                    if iou(bb0,torch.FloatTensor(bb1)) > iou_thresh:
                        tp += 1
                        bbox.remove(bb1)
                        break
                    else:
                        continue
                        
            fn = len(bbox)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            curr_pr_curve.append([precision, recall, conf_thresh])
            
            if verbose:
                print(f'(tp, fp, fn) = ({tp}, {fp}, {fn})\n')
    
        curr_pr_curve.append([1.0, 0.0, -1])
        all_pr_curves.append(curr_pr_curve)

    return all_pr_curves



def plot_pr_curves(all_pr_curves, save_path = '', condense = True):
    """
    Given one or more sets of (precision, recall, confidence) 3-tuples, generate either 1 (condense=True) or 10 (condense=False) PR curves visualizing the provided data, parameterized by confidence.

    Parameters:
    -----------
    all_pr_curves : list of shape [10,101,3]
        Each element of this list is defined by 100 3-tuples which correspond to the (precision, recall, confidence) values at the given thresholds (iou_thresh, conf_thresh). This parameter is intended to be the output from the compute_pr_curves() function in this same package.
    save_path : string
        Default - ''; If provided, the figure generated by this function will be saved at the corresponding location
    condense : bool
        Default - True; If True, will plot every PR curve on the same figure and generate a legend which relates the plots with their corresponding iou_threshold value. If False, will generate a separate plot for each PR curve within the figure.

    Returns:
    --------
    fig : matplotlib.figure
        A figure containing all of the PR curves stored in the parameter 'all_pr_curves'
    ax : matplotlib.axes
        A single axis containing the plotted data of all 10 PR curves or a set of 10 axes containing the plotted data of each corresponding PR curve        
    """
    iou_threshold  = [round(x,2) for x in list(np.arange(0.5,1.0,0.05))]

    if condense:
        fig, ax = plt.subplots()
        cmap = mpl.colormaps['magma']
        colors = cmap(np.linspace(0,1,10))
        
        for idx, pr_curve in enumerate(all_pr_curves):
            p = [elem[0] for elem in pr_curve]
            r = [elem[1] for elem in pr_curve]
            ax.plot(r,p,color = colors[idx], label = f'{iou_threshold[idx]:.2f}')

        ax.scatter(r[0],p[0],color='green')
        ax.scatter(r[-1],p[-1],color='red')
        ax.legend(title="IOU")
        ax.set_title("PR Curves of YOLO model at Various IOU Thresholds")
    
    else:
        nrows, ncols = 2, 5
        fig, ax = plt.subplots(nrows, ncols, layout = 'tight')
        fig.set_size_inches(ncols*2, nrows*2)
    
        i, j = 0, 0
        for idx, pr_curve in enumerate(all_pr_curves):
        
            p = [elem[0] for elem in pr_curve]
            r = [elem[1] for elem in pr_curve]
            
            ax[i][j].scatter(r,p)
            ax[i][j].plot(r,p)
            ax[i][j].scatter(r[0],p[0],color='green')
            ax[i][j].scatter(r[-1],p[-1],color='red')
            ax[i][j].set_title(f'IOU: {iou_threshold[idx]:.2f}')
        
            j += 1
            if j >= ncols:
                i += 1
                j = 0

    plt.show()
    if save_path != '':
        plt.savefig(save_path)
    
    return fig, ax



def get_mAP(all_pr_curves):
    """
    Given a set of PR curves at various IOU thresholds, compute the mean average precision across all the provided PR curves. This is done by taking the mean of the average precision of each PR curves, which is computed using the trapezoid method from the scipy.integrate package.

    Parameters:
    -----------
    all_pr_curves : list of shape [10,101,3]
        Each element of this list is defined by 100 3-tuples which correspond to the (precision, recall, confidence) values at the given thresholds (iou_thresh, conf_thresh). This parameter is intended to be the output from the compute_pr_curves() function in this same package.

    Returns:
    --------
    mAP : float
        The mean average precision metric which quantifies the performance of a model specialized in the object segmentation task and ranges from [0,1]
    """

    all_ap = []
    for idx, pr_curve in enumerate(all_pr_curves):
        p = [elem[0] for elem in pr_curve]
        r = [elem[1] for elem in pr_curve]
        ap = trapezoid(r,p)
        all_ap.append(ap)
    
    mAP = np.mean(all_ap)

    return mAP