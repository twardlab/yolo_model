from yolo_help import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import trapezoid

def postprocess(out, B, stride, pads, ds_factor = 8, up_factor = None, n_bb = 5, verbose = False):
    """
    Converts the raw output from a YOLO network into a more meaningful data structure through the following steps:
    (1) Remove B-1 bboxes, keeping the one with highest confidence
    (2) Apply a sigmoid function to normalize the confidence values of each bbox
    (3) Convert raw output into units and positions of the (upsampled) input image
    (4) Remove padding which was added during the tile extraction step
    (5) If upsampling was performed in the preprocessing step, "downsample" the bbox scalars
    (6) Permute the output from [8,R,C] to [R,C,8] for simpler downstream analysis

    Parameters:
    -----------
    out : torch.Tensor
        The output object from the primary YOLO neural network
    B : int
        The number of bounding boxes per block that the model will output.  (Automatically defined when initializing the network)
    stride : int
        The number of pixels to pass over when convolving the input image in each layer (Automatically defined when initializing the network)
    pad : array of int
        List of the thickness of the padding along each axis in units of pixels
    ds_factor : int
        Default - 8; The factor by which the model output is shrunk relative to the original image size
    up_factor : int
        Default - None; The factor used to upsample the original image prior to padding
    n_bb : int
        Default - 5; The number of scalars which define a single bbox
    verbose: bool
        Default - False; If True, print out the time taken to run through the entire function.

    Returns:
    --------
    out_ : torch.Tensor of shape [N/4, M/4, 8]
        The processed output from an original grayscale or RGB image of size [N,M]
    """

    start = time.time()

    # Choose bbox with highest confidence from all B bboxes at each grid cell
    n_bb = 5 # The number of parameters that define a single bbox
    best_bb_idx = np.array(n_bb*np.argmax(out[(n_bb-1):n_bb*B:n_bb], axis=0)[None, ...] + np.arange(n_bb)[:, None, None], dtype=int)
    # best_bb_idx  = np.argmax(out[(n_bb-1):n_bb*B:n_bb], axis=0)[None, ...] # Chooses either bb0 or bb1 based on the bb with higher confidence
    # best_bb_idx *= n_bb # Modify the [0,1] indeces, so they now correspond to the first idx corresponding to relevant bb in 'out'
    # best_bb_idx += np.arange(n_bb)[:, None, None] # Expand the indeces to include all n_bb scalars associated with the corresponding bb in 'out'

    # NOTE: This line concats the best bbox info with ALL the remaining data from 'out'. This is currently just 3 class probabilities, but may need to be changed in the future
    out_ = np.concatenate([np.take_along_axis(out[:n_bb*B], best_bb_idx, axis=0), out[n_bb*B:]], axis=0)
    out_ = torch.tensor(out_, dtype=torch.float32)
    
    # Normalize confidence using the sigmoid function
    out_[4] = torch.sigmoid(out_[4])
    
    # Convert bbox format from [cx, cy, w, h] => [left, right, top, bottom]
    x = torch.arange(out.shape[-1])*stride + (stride-1)/2
    y = torch.arange(out.shape[-2])*stride + (stride-1)/2
    YX = torch.stack(torch.meshgrid(y,x,indexing='ij'),0)
    w = torch.exp(out_[2])*stride
    h = torch.exp(out_[3])*stride
    
    left = (torch.sigmoid(out_[0])-0.5)*stride + YX[1] - w/2
    top = (torch.sigmoid(out_[1])-0.5)*stride + YX[0] - h/2
    right = left + w
    bottom = top + h

    # Redefine bbox definition + shift by the padding
    out_[0] = left - pads[1]
    out_[1] = top - pads[0]
    out_[2] = right - pads[1]
    out_[3] = bottom - pads[0]

    # Remove padding, so that model outputs can be placed directly over the unpadded (potentially upsampled) image
    out_ = out_[:, int(pads[0]/ds_factor) : (out.shape[1]-int(pads[0]/ds_factor)), int(pads[1]/ds_factor): (out.shape[2]-int(pads[1]/ds_factor))]

    # If the original image was upsampled before being padded, convert output to the pre-upsampled units
    if up_factor != None:
        out_[:4] = out_[:4] / up_factor        

    # Reshape 'out', so that the data can be accessed via out[r,c] instead of out[:,r,c]
    out_ = torch.permute(out_, (1,2,0))

    if verbose:
        print(f'Finished postprocessing in {time.time()-start:.2f}s')

    return out_

def bb_to_rec(out, pos = [0,1,2,3], **kwargs):
    """
    This function converts the model output into a set of bounding boxes to be plotted
    
    Parameters:
    -----------
    out : torch.Tensor
        The output object from the primary YOLO neural network after the best bbox per cell has been selected and the coordinate values have been reformated in the postprocessing stage.
    pos : ndarray of int
        The indeces corresponding to the [left, top, right, bottom] points at out[i,j]

    Returns:
    --------
    out : matplotlib.collections.PolyCollection 
        Contains N rectangles to be plotted later
    
    """
    left = torch.Tensor(out[:,:,pos[0]].ravel())
    top = torch.Tensor(out[:,:,pos[1]].ravel())
    right = torch.Tensor(out[:,:,pos[2]].ravel())
    bottom = torch.Tensor(out[:,:,pos[3]].ravel())
    
    p0 = torch.stack( (left, top), dim=1)
    p1 = torch.stack( (right, top), dim=1)
    p2 = torch.stack( (right, bottom), dim=1)
    p3 = torch.stack( (left, bottom), dim=1)
    p = torch.stack( (p0,p1,p2,p3),dim=-2)
    
    return PolyCollection(p, **kwargs)

def remove_low_conf_bboxes(bboxes, scores, conf_thresh = 0.1):
    """
    Remove bounding boxes from the parameter 'bboxes' if the corresponding element in the parameter 'scores' is below conf_thresh.

    Parameters:
    -----------
    bboxes : torch.Tensor of size [N, 4]
        N corresponds to the number of bounding boxes
    scores : torch.Tensor of size [N, 1]
        N corresponds to the number of bboxes; scores[i] corresponds to the confidence that bboxes[i] encompasses a target object
    conf_thresh : float
        If scores[i] is below this threshold, then remove bboxes[i] from the input

    Returns:
    --------
    bboxes_out : torch.Tensor of size [M, 4]
        M corresponds to the number of bounding boxes remaining after this filtering step
    scores_out : torch.Tensor of size [M, 1]
        The M scores corresponding to the bounding boxes defined in bboxes_out    
    """

    bboxes_out = [b for b,s in zip(bboxes,scores) if s > conf_thresh]
    scores_out = [s for s in scores if s > conf_thresh]
    
    return torch.stack(bboxes_out,dim=0), torch.stack(scores_out,dim=0)



def NMS(bboxes, scores, nms_threshold = 0.8):
    """
    Perform non-maximum suppression (NMS) on the outputs from the yolo model framework in order to reduce the number of candidate bounding boxes.

    Parameters:
    -----------
    bboxes : torch.Tensor of size [N, 4]
        N corresponds to the number of bounding boxes
    scores : torch.Tensor of size [N, 1]
        N corresponds to the number of bboxes; scores[i] corresponds to the confidence that bboxes[i] encompasses a target object
    nms_threshold : float
        The iou threshold used to remove candidate bounding boxes when comparing their spatial position with respect to the bounding box with the highest confidence during the current iteration

    Returns:
    --------
    bboxes_out : torch.Tensor of size [M, 4]
        M corresponds to the number of bounding boxes remaining after NMS
    scores_out : torch.Tensor of size [M, 1]
        The M scores corresponding to the bounding boxes defined in bboxes_out
    """
    # Sort the bboxes and scores in descending order based on score
    def bb_sort(bbox):
        return bbox[0]
    
    bboxes_nms = [b for _ , b in sorted(list(zip(scores, bboxes)), key=bb_sort, reverse = True)]
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

    bboxes_out = torch.stack(bboxes_out, dim=0)
    scores_out = torch.stack(scores_out, dim=0)
    
    return bboxes_out, scores_out



def compute_pr_curves(all_gt_bboxes, all_pred_bboxes, all_pred_scores, outdir, verbose = False):
    """
    Given a set of ground truth bounding boxes, a set of predicted bounding boxes, and the corresponding confidence scores for each predicted bounding box, generate a set of pr curves quantifying the performance of the model. One PR curve will be generated at each IOU threshold from 0.5 : 0.05 : 1.0 and contain 101 points corresponding to each confidence threshold from 0.01 : 0.01 : 1.0 and 2 predefined endpoints. This function will automatically resume the computations at the last (iou, conf) tuple if

    Parameters:
    -----------
    gt_bboxes : torch.Tensor of shape [X,N,4]
        The ground truth bounding boxes which correspond to the target objects in the original image. X is the number of images in the original gt dataset and N is the number of gt bounding boxes for each image. Note that N may vary from image to image.
    pred_bboxes : torch.Tensor of shape [X,M,4]
        The filtered bounding boxes which were output from the model. X is the number of images in the original gt dataset and M is the number of predicted bounding boxes for image 'x' in the gt dataset.
    pred_scores : torch.Tensor of shape [X,M,1]
        The confidence scores corresponding to the bounding boxes in the parameter 'pred_boxes'. X is the number of images in the original gt dataset and M is the number of predicted bounding boxes for image 'x' in the gt dataset.
    verbose : bool
        Default - False; If True, print out the (tp, fp, fn) 3-tuple for every (iou, conf) threshold
        
    Returns:
    --------
    all_pr_curves : list of shape [10,101,3]
        Each element of this list is defined by 101 3-tuples which correspond to the (precision, recall, confidence) values at the given thresholds (iou_thresh, conf_thresh).
    """

    iou_threshold  = [round(x,2) for x in list(np.arange(0.5,1.0,0.05))]
    conf_threshold = [round(x,2) for x in list(np.arange(0.01,1.0,0.01))]

    # Resume computation if the file exists
    fname = 'all_pr_curves.npz'
    if os.path.exists(os.path.join(outdir, fname)):
        pr_data = np.load(os.path.join(outdir, fname))
        last_iou = pr_data['last_iou'].item()
        all_pr_curves = list(pr_data['data'])
        resume = True
        
        if verbose:
            print(f'Loaded data stored at {os.path.join(outdir, fname)}. Last iou was {last_iou}')
        
        # If all computations have already been completed, return the pr curve data stored at os.path.join(outdir, fname)
        if last_iou == iou_threshold[-1]:
            return all_pr_curves
    
    else:
        all_pr_curves = []
        resume = False

    start_iou = time.time()
    for iou_thresh in iou_threshold:

        # Either 'continue' current loop if data has already been generated, or define the current pr_curve
        if (not resume) or (resume and (iou_thresh > last_iou)):
            curr_pr_curve = [[0.0, 1.0, -1]]
        else: # resume AND iou_thresh <= last_iou
            print(f'Skipping iou {iou_thresh}')
            continue
        
        if verbose:
            print(f'===== Starting iou {iou_thresh} =====')
            
        for conf_thresh in conf_threshold:

            if verbose:
                print(f'Starting conf {conf_thresh}')
                start = time.time()
                
            tp = 0
            fp = 0
            fn = 0
            start0 = time.time()
            for idx, pred_bboxes in enumerate(all_pred_bboxes):
                pred_scores = all_pred_scores[idx]
                gt_bboxes = all_gt_bboxes[idx].copy().tolist()
                n_boxes_remaining = len(pred_bboxes)
                for bb0, s in zip(pred_bboxes, pred_scores):
                    if s < conf_thresh:
                        fp += 1
                        n_boxes_remaining -= 1
                        continue
                    
                    for bb1 in gt_bboxes:
                        if iou(bb0,torch.FloatTensor(bb1)) > iou_thresh:
                            tp += 1
                            gt_bboxes.remove(bb1)
                            n_boxes_remaining -= 1
                            break
                        else:
                            continue       
                fn += len(gt_bboxes) # All gt_bboxes that have no corresponding pred_bbox
                fp += n_boxes_remaining # All pred_bboxes that are above conf_thresh, but have no corresponding gt bbox
                if idx % 2 == 0 and idx != 0:
                    print(f'Finished images {idx-1}:{idx} / {len(all_pred_bboxes)} in {time.time() - start0:.2f}s')
                    start0 = time.time()

            precision = 0 if (tp+fp) == 0 else tp / (tp + fp)
            recall = 0 if (tp+fn) == 0 else tp / (tp + fn)
            curr_pr_curve.append([precision, recall, conf_thresh])
            print(f'{tp} + {fn} = {tp+fn} ?= 2125')
                

            if verbose:
                print(f'(tp, fp, fn) = ({tp}, {fp}, {fn}) across {len(all_pred_scores)} images')
                print(f'Finished iter in {time.time() - start:.2f}s\n')
                start = time.time()
    
        curr_pr_curve.append([1.0, 0.0, -1])
        all_pr_curves.append(curr_pr_curve)
        np.savez(os.path.join(outdir, fname), data = all_pr_curves, last_iou = iou_thresh)

        if verbose:
            print(f'Finished iou {iou_thresh} in {time.time() - start_iou:.2f}s')
            start_iou = time.time()

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