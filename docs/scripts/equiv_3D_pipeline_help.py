import torch
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import random
from matplotlib.patches import Rectangle
import tifffile
import h5py
from torch.utils.data import DataLoader
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter
from glob import glob

from stardist.models import StarDist3D
import nibabel as nib

import sys
sys.path.append(os.path.join(os.path.split(os.getcwd())[0], 'scripts'))

import yolo_tiles
from yolo_tiles import img_to_tiles, apply_model_to_tiles, load_test_image, preprocess, tileDataset, remove_bbox_in_overlap

import yolo_help
from yolo_help import bbox_to_rectangles, imshow, convert_data, Net, get_best_bounding_box_per_cell

import yolo_post_help
from yolo_post_help import remove_low_conf_bboxes, postprocess, bb_to_rec

import yolo_help_3D
from yolo_help_3D import apply_model_to_orthogonal_slices, gen_3D_GT, recon_down, vol_to_mip_rgb, orthogonal_to_3D, vol_to_mip

import yolo_avg_prec
from yolo_avg_prec import get_AP, gt_bbox_to_pred_bbox_format, IOU_3D, yolo_output_to_list

sys.path.append('/home/abenneck/equivariance_model')
import moment_kernels as mk

def apply_2D_equiv_to_slices(img, model_path, outdir, verbose=False):

    # Load the equivariant model
    net = mk.EqNet()
    net.load_state_dict(torch.load(model_path))
    B = net.B
    stride = net.stride

    # If outdir DNE, make it
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    start_total = time.time()
    next_idx = 0
    # Apply the model to orthognal views of the input image
    for curr_ax in [0,1,2]:
    
        outdir_ = os.path.join(outdir, f'ax{curr_ax}')
        if not os.path.exists(outdir_):
            os.mkdir(outdir_)
        
        for idx in np.arange(img.shape[curr_ax]):
        
            if idx < next_idx:
                continue
        
            # Extract the slice 
            if curr_ax == 0:
                img_slice = img[idx,:,:]
                out_fname = 'gt_test_out_' + f'idx_{idx:05d}_:_:.npy'
            elif curr_ax == 1:
                img_slice = img[:,idx,:]
                out_fname = 'gt_test_out_' + f'idx_:_{idx:05d}_:.npy'
            elif curr_ax == 2:
                img_slice = img[:,:,idx]
                out_fname = 'gt_test_out_' + f'idx_:_:_{idx:05d}.npy'
            else:
                raise Exception(f'Invalid ax ({curr_ax}) supplied, only ({[k for k in range(2)]}) allowed')
        
            # Preprocess using gamma correction + upsampling
            start = time.time()
            img_up = preprocess(img_slice)
            
            # Extract tiles from the preprocessed input image
            padded_img, tiles = img_to_tiles(img_up, lower_threshold_bg = 0.04, verbose=False)
            
            # Apply model to tiles + apply bbox edge filtering
            out = apply_equiv_model_to_tiles(tiles, model_path, padded_img.shape[0], padded_img.shape[1], verbose=False)

            # Convert ellipse parameters to bbox parameters
            out = equiv_to_yolo(out)
        
            # Convert the raw model output into a more useful data structure
            pads = (np.array(padded_img.shape) - np.array(img_up.shape))/2
            out = torch.tensor(out.clone().detach(), dtype=torch.float32)
            out = postprocess(out, B, stride, pads, up_factor=2, verbose=False)
        
            # Save the processed output
            out_path = os.path.join(outdir_, out_fname)
            if True:
                np.save(out_path, out)

            if verbose:
                print(f'Saved the outputs for slice {idx}/{img.shape[curr_ax]} in {time.time()-start_total:.2f}s')
                start_total = time.time()

    return

def apply_equiv_model_to_tiles(tiles, model_path, img_dim0, img_dim1, out_path='', batch_mode = False, verbose=False):
    """
    Apply a pretrained YOLO model to each element of 'tiles'. Stitch together each of these model outputs in order to create a processed version of the original (padded) input image. Save this reconstructed output at 'out_path'.

    Parameters:
    -----------
    tiles : list(dict)
        A list containing all the tiles which comprise 'padded_img'. Each element is a dictionary containing 5 values: (1) The pixel values of the tile, (2) The anchor point for the tile used in reconstruction, (3) A boolean which defines the tile as background or foreground, (4) The row idx of the tile in the whole image, and (5) The col idx of the tile in the whole iamge
    model_path : str
        The location of the weights for the pretrained model
    img_dim0 : int
        The size of the input image along axis 0; Used for reconstruction
    img_dim1 : int
        The size of the input image along axis 1; Used for reconstruction  
    out_path : str
        Default - ''; If no argument is passed, the output will only exist in memory and will not be saved. The location and fname of where the reconstructed output should be saved
    batch_mode : bool
        Default - False; If True, apply the model to a batch of tiles instead of a single tile
    verbose : bool
        Default - False; If True, print out intermediate and final progress notes

    Returns:
    --------
    recon : torch.Tensor()
        The reconstructed model output where each element corresponds to a specific pixel in the original image. The 13 values in each element of the reconstruction ([cx0, cy0, w0, h0, conf0, cx1, cy1, w1, h1, conf1, cl0, cl1, cl2]) can be interpreted as follows. There are 2 predicted bounding boxes centered at (cx0, cy0) and (cx1,cy1) with their corresponding width, height, and confidence values defined by (w0, h0, conf0) and (w1, h1, conf1), respectively. cl0, cl1, and cl2 are the class probabilities and define the likelihood that a cell can be categorized as having a smooth, sharp, or bumpy boundary, respectively.
    """
    start = time.time()
    start_total = time.time()

    # Load pretrained model weights
    dtype = torch.float32
    net = mk.EqNet()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    B = net.B
    C = net.C
    stride = net.stride
    conf_idx = -4
    
    # Define dimensions of the input image
    if len(np.shape(tiles[0]['img'])) == 3:
        num_ch, tile_dim, _ = np.shape(tiles[0]['img'])
        ndim = 3
    elif len(np.shape(tiles[0]['img'])) == 2:
        tile_dim, _ = np.shape(tiles[0]['img'])
        ndim = 2
    else:
        ex_tile = tiles[0]['img']
        raise Exception(f'Input tiles should be 2 or 3 dimensions, but {len(ex_tile.shape)} were found')
    
    boundary_cond = [16,16]
    # TODO: when implementing UI using argparse, Boundary_cond = min_overlap / 2
    ds_factor = 8
    bbox_dim = 5
    num_classes = 3
    # recon = np.zeros((bbox_dim*B+num_classes, int(img_dim0/ds_factor), int(img_dim1/ds_factor)))
    # (10/17/25) Now conf of background bboxes will be 0 after postprocessing
    recon = np.ones((17, int(img_dim0/ds_factor), int(img_dim1/ds_factor)))*(-np.inf) # (11/10/26) 17 param per cell in equivariant model output

    non_bg_tile = 0
    last_idx = 0
    for idx, tile in enumerate(tiles):
        I = tile['img']
        if ndim == 2: # Model expects input shape of length 4, so 
            I = I[None]
        p = tile['p']
        bg = tile['bg']
        r_idx = tile['r_idx']
        c_idx = tile['c_idx']
        
        if bg:
            continue
            if verbose and idx % 100 == 0:
                print(f'Skipping tile {idx}/{len(tiles)} (r,c):({r_idx},{c_idx})')
        else:
            # TODO: Apply trained model to the image + define some key model outputs
            out = net((torch.tensor(I[None],dtype=dtype)))
            out = remove_bbox_in_overlap(out, B, stride, tile_dim, boundary_cond = boundary_cond)

            # Append model output to reconstruction
            tile_out = out[0].detach().numpy()
            t0_start = int(p[0]/ds_factor)
            t0_end = int((p[0]+tile_dim)/ds_factor)
            t1_start = int(p[1]/ds_factor)
            t1_end = int((p[1]+tile_dim)/ds_factor)
            recon[:, t0_start:t0_end, t1_start:t1_end] = np.where(tile_out[-4,:,:] > recon[-4, t0_start:t0_end, t1_start:t1_end], tile_out, recon[:, t0_start:t0_end, t1_start:t1_end])
            # recon[:,int(p[0]/ds_factor):int((p[0]+tile_dim)/ds_factor), int(p[1]/ds_factor):int((p[1]+tile_dim)/ds_factor)] = out[0].detach().numpy()

            non_bg_tile += 1
            if verbose and idx % 100 == 0:
                print(f'Finished tiles {last_idx}:{idx}/{len(tiles)} in {time.time()-start:.2f}s')
                start = time.time()
                last_idx = idx

    if verbose:
        print(f'Finished applying model to entire image in {time.time()-start_total:.2f}s with {non_bg_tile}/{len(tiles)} ({non_bg_tile/len(tiles):.3f}) tiles marked as foreground')

    # Save the reconstructed model output as an .npy file
    if out_path != '':
        np.save(out_path, recon)

    return torch.tensor(recon, dtype=torch.float32)


def equiv_to_yolo(out, B=2, C=3, eps=1e-6):
    """
    Convert 17-channel equivariant ellipse predictions into 13-channel YOLO-style bounding-box predictions.

    Input channels:
    [cl0, cl1, cl2, conf0, conf1, cy0, cx0, cy1, cx1, 
    Lyy0, Lyx0, Lxy0, Lxx0, Lyy1, Lyx1, Lxy1, Lxx1]

    Output channels:
    [cx0, cy0, w0, h0, conf0, cx1, cy1, w1, h1, conf1, cl0, cl1, cl2]
    """

    # Get class probabilities
    classes = [out[c] for c in range(C)]

    bboxes = []
    for b in range(B):

        # Raw equivariant center offsets
        cy = out[C+B+2*b]
        cx = out[C+B+2*b+1]

        # Convert equivariant offsets to YOLO logits
        cx_yolo = torch.logit((cx + 0.5).clamp(eps, 1-eps))
        cy_yolo = torch.logit((cy + 0.5).clamp(eps, 1-eps))

        # Equivariant ellipse matrix
        Lyy = out[C+3*B+4*b] + 0.5
        Lyx = out[C+3*B+4*b+1]
        Lxy = out[C+3*B+4*b+2]
        Lxx = out[C+3*B+4*b+3] + 0.5

        # Axis-aligned bounding rectangle
        width = 2 * torch.sqrt(Lxx**2 + Lxy**2)
        height = 2 * torch.sqrt(Lyx**2 + Lyy**2)

        # Encode dimensions in YOLO's logarithmic format
        log_w = torch.log(width.clamp_min(eps))
        log_h = torch.log(height.clamp_min(eps))

        # Preserve raw confidence logits
        confidence = out[C+b]

        bboxes.extend([
            cx_yolo,
            cy_yolo,
            log_w,
            log_h,
            confidence
        ])

    return torch.stack(bboxes + classes, dim=0)