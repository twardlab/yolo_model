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

def apply_2D_yolo_to_slices(img_path, model_path, outdir, verbose=False):

    # Load the target volume
    if 'nii.gz' in img_path:
        img = nib.load(img_path)
        img = img.get_fdata()
    else:
        raise Exception(f'Invalid file provided, only Nifti files are accepted')

    # Load the YOLO model
    net = Net()
    net.load_state_dict(torch.load(model_path))
    B = net.B
    stride = net.stride

    start_total = time.time()
    next_idx = 0
    # Apply the model to orthognal views of the input image
    for curr_ax in [0,1,2]:
    
        outdir = os.path.join(outdir, f'ax{curr_ax}')
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
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
            out = apply_model_to_tiles(tiles, model_path, padded_img.shape[0], padded_img.shape[1], verbose=False)
        
            # Convert the raw model output into a more useful data structure
            pads = (np.array(padded_img.shape) - np.array(img_up.shape))/2
            out = torch.tensor(out.clone().detach(), dtype=torch.float32)
            out = postprocess(out, B, stride, pads, up_factor=2, verbose=False)
        
            # Save the processed output
            out_path = os.path.join(outdir, out_fname)
            if True:
                np.save(out_path, out)

            if verbose:
                print(f'Saved the outputs for slice {idx}/{img.shape[curr_ax]} in {time.time()-start_total:.2f}s')
                start_total = time.time()

    return

def stitch_slices_into_cube(indir0, indir1, indir2, outdir, d=256, down_factor=4):
    """
    Stitch together 3 sets of orthogonal slices into a single cube

    Parameters:
    ===========
    indir0 : str
        The directory containing the slice outputs from the previous step in the pipeline. Note: This will be the same as the \'outdir\' argument from the previous stage, but with the folder \'ax0\' appended to the end of the path.
    indir1 : str
        The directory containing the slice outputs from the previous step in the pipeline. Note: This will be the same as the \'outdir\' argument from the previous stage, but with the folder \'ax1\' appended to the end of the path.
    indir2 : str
        The directory containing the slice outputs from the previous step in the pipeline. Note: This will be the same as the \'outdir\' argument from the previous stage, but with the folder \'ax2\' appended to the end of the path.
    outdir : str
        The output directory
    d : int
        Default - 256; The desired output shape for each cube
    down_factor : int
        Default - 4; The factor by which the size of the whole dataset shrunk during processing (2x upsampling AND 8x 'downsampling' => 4x downsampling)
        
    """
    # If outdir DNE, make one
    if not os.path.exists(outdir):
        os.mkdir(outdir)    
    
    # Transverse / Axial / Horizontal slices - [DV, AP, LR]
    data0 = np.load(os.path.join(indir0,os.listdir(indir0)[0]))
    shape0 = data0.shape
    
    # Coronal slices - [AP, DV, LR]
    data1 = np.load(os.path.join(indir1,os.listdir(indir1)[0]))
    shape1 = data1.shape
    
    # Sagittal slices - [LR, DV, AP]
    data2 = np.load(os.path.join(indir2,os.listdir(indir2)[0]))
    shape2 = data2.shape
    
    final_shape = np.array([shape1[0], shape0[0], shape0[1]]) # [DV, AP, LR]
    
    n_cube_i = int(np.ceil(final_shape[0] / d)) # Num complete cubes along AP axis (Transverse)
    n_cube_j = int(np.ceil(final_shape[1] / d)) # Num complete cubes along DV axis (Coronal)
    n_cube_k = int(np.ceil(final_shape[2] / d)) # Num complete cubes along LR axis (Sagittal)
    print(f'Total Cubes Shape: ({n_cube_i}, {n_cube_j}, {n_cube_k})')

    # If there is only 1 cube to process, edge cases are dealt with in a slightly different way
    small_data = False
    if np.sum([n_cube_i, n_cube_j, n_cube_k]) == 3:
        small_data = True
        
    next_i, next_j, next_k = 0,0,0
    
    for i in np.arange(n_cube_i): # NOTE: Originally, included a (+1) to each np.arange(...) to include edge case cubes
        
        if i < next_i:
            continue
            
        for j in np.arange(n_cube_j):
            
            if j < next_j and i <= next_i:
                continue
                
            for k in np.arange(n_cube_k):
    
                if k < next_k and i <= next_i and j <= next_j:
                    continue
    
                print(f'Starting cube ({i},{j},{k}) . . .')
                start_time = time.time()
    
                # Determine if cube is an edge case
                isEdge_i = True if i == n_cube_i - 1 else False
                isEdge_j = True if j == n_cube_j - 1 else False
                isEdge_k = True if k == n_cube_k - 1 else False
                print(f'Edge? - {isEdge_i} {isEdge_j} {isEdge_k}')
    
                # The spatial coordinates of the datacube
                ndatacube = np.array((d,d,d)) 
                
                # x0 is the location of the first pixel (within the high res volume)
                x0datacube = np.array([0,0,0]) # initialize 
                x0datacube = np.array([1.5,1.5,1.5]) + np.array([i,j,k])*d*down_factor # ([]*256*4 - up by 2 down by 8)
                
                # get the coordinates of each voxel
                xdatacube = [np.arange(n)*down_factor+o for n,o in zip(ndatacube,x0datacube)]
                
                # get the voxel size
                ddatacube = np.array([down_factor,down_factor,down_factor])
                
                # Get start and end indeces for coordinate systems
                start0 = xdatacube[0][0]
                start1 = xdatacube[1][0]
                start2 = xdatacube[2][0]
                end0 = xdatacube[0][-1] if isEdge_i else xdatacube[0][-1]
                end1 = xdatacube[1][-1] if isEdge_j else xdatacube[1][-1]
                end2 = xdatacube[2][-1] if isEdge_k else xdatacube[2][-1]
    
                # Load the detections from each directory
                # NOTE: Order is xmin,ymin,xmax,ymax
                
                # Generate a list of fnames for all the slices in each dir (Each slice contains bbox info at every pixel)
                ax0files = glob(os.path.join(indir0,'*.npy'))
                ax0files.sort()
                ax0detects = np.load(ax0files[0]) # Load the first slice for shape info
                
                ax1files = glob(os.path.join(indir1,'*.npy'))
                ax1files.sort()
                ax1detects = np.load(ax1files[0]) # Load the first slice for shape info
                
                ax2files = glob(os.path.join(indir2,'*.npy'))
                ax2files.sort()
                ax2detects = np.load(ax2files[0]) # Load the first slice for shape info
    
                # (07/16/26) Added if/else logic to address edge case
                start0i = np.floor(start0).astype(int)-1
                start1i = np.floor(start1).astype(int)-1
                start2i = np.floor(start2).astype(int)-1
                end0i = len(ax0files)-1 if isEdge_i else np.ceil(end0).astype(int)
                end1i = len(ax1files)-1 if isEdge_j else np.ceil(end1).astype(int)
                end2i = len(ax2files)-1 if isEdge_k else np.ceil(end2).astype(int)
    
                print(f'Extent: [{start0},{end0}],[{start1},{end1}],[{start2},{end2}]')
                print(f'Extent_i:  [{start0i},{end0i}],[{start1i},{end1i}],[{start2i},{end2i}]')
    
                # get the pixel locations, and extents for display, in each of the three views
                ax0x1 = np.arange(ax0detects.shape[0])*down_factor + np.mean( np.array([0,1,2,3]) )
                ax0x2 = np.arange(ax0detects.shape[1])*down_factor + np.mean( np.array([0,1,2,3]) )
                ax0x = [ax0x1,ax0x2]
                extentax00,extentax01,extentax02 = extent_from_x([[0,1],ax0x1,ax0x2])
                
                ax1x0 = np.arange(ax1detects.shape[0])*down_factor + np.mean( np.array([0,1,2,3]) )
                ax1x2 = np.arange(ax1detects.shape[1])*down_factor + np.mean( np.array([0,1,2,3]) )
                ax1x = [ax1x0,ax1x2]
                extentax10,extentax11,extentax12 = extent_from_x([ax1x0,[0,1],ax1x2])
                
                ax2x0 = np.arange(ax2detects.shape[0])*down_factor + np.mean( np.array([0,1,2,3]) )
                ax2x1 = np.arange(ax2detects.shape[1])*down_factor + np.mean( np.array([0,1,2,3]) )
                ax2x = [ax2x0,ax2x1]
                extentax20,extentax21,extentax22 = extent_from_x([ax2x0,ax2x1,[0,1],])
                
                print('Finished initializing global variables and data structures . . .')
                
                # Start by loading the transverse slices + Compress data via downsampling
                detects = []
                detects_ = []
                for i_ in range(start0i,end0i+1):   # loop over all the slices we identified
                    ax0detects_ = np.load(ax0files[i_],mmap_mode='r') # use memory mapping for speed
                    # get a range, figure out which pixel to start and end at
                    # recall ax0x1,ax0x2 are the coordinates of each pixel in my slice output (detection)
                    rowstart = np.argmin((ax0x1 - start1)**2)
                    colstart = np.argmin((ax0x2 - start2)**2)
                    rowend = np.argmin((ax0x1 - end1)**2) - 1 if isEdge_j and not small_data else np.argmin((ax0x1 - end1)**2)
                    colend = np.argmin((ax0x2 - end2)**2) - 1 if isEdge_k and not small_data else np.argmin((ax0x2 - end2)**2)
                    ax0detects_ = np.array( ax0detects_[rowstart:rowend+1,colstart:colend+1]     )
                    detects_.append(ax0detects_)
                    if len(detects_) == 4 or i_ == end0i :        
                        detects_ = np.stack(detects_)
                        inds = np.argmax(detects_[...,4],axis=0)
                        detects__ = np.take_along_axis(detects_,inds[None,...,None],axis=0)
                        # we now expect detects__ to have a singleton dimension at the beginning
                        if detects__.shape[0] == 1: 
                            detects__ = detects__[0]
                        else:
                            raise Exception()
                        detects.append(detects__)        
                        detects_ = []        
                ax0combinedetects = np.stack(detects,axis=0)
                print('Finished combining detections along transverse view . . .')
                
                # next load the coronal slices
                detects = []
                detects_ = []
                for i_ in range(start1i,end1i+1):
                    ax1detects_ = np.load(ax1files[i_],mmap_mode='r')
                    # get a range
                    rowstart = np.argmin((ax1x0 - start0)**2)
                    colstart = np.argmin((ax1x2 - start2)**2)
                    rowend = np.argmin((ax1x0 - end0)**2) if isEdge_i else np.argmin((ax1x0 - end0)**2)
                    colend = np.argmin((ax1x2 - end2)**2) - 1 if isEdge_k and not small_data else np.argmin((ax1x2 - end2)**2)
                    ax1detects_ = np.array( ax1detects_[rowstart:rowend+1,colstart:colend+1]     )
                    detects_.append(ax1detects_)
                    if len(detects_) == 4 or i_ == end1i :        
                        detects_ = np.stack(detects_)
                        inds = np.argmax(detects_[...,4],axis=0)
                        detects__ = np.take_along_axis(detects_,inds[None,...,None],axis=0)
                        # we now expect detects__ to have a singleton dimension at the beginning
                        if detects__.shape[0] == 1: 
                            detects__ = detects__[0]
                        else:
                            raise Exception()
                        detects.append(detects__)        
                        detects_ = []        
                ax1combinedetects = np.stack(detects,axis=1)
                print('Finished combining detections along coronal view . . .')
                
                # now do the sagittal slices
                detects = []
                detects_ = []
                for i_ in range(start2i,end2i+1): 
                    ax2detects_ = np.load(ax2files[i_],mmap_mode='r')
                    # get a range
                    rowstart = np.argmin((ax2x0 - start0)**2)
                    colstart = np.argmin((ax2x1 - start1)**2)
                    rowend = np.argmin((ax2x0 - end0)**2) if isEdge_i else np.argmin((ax2x0 - end0)**2)
                    colend = np.argmin((ax2x1 - end1)**2) - 1 if isEdge_j and not small_data else np.argmin((ax2x1 - end1)**2)
                    ax2detects_ = np.array( ax2detects_[rowstart:rowend+1,colstart:colend+1]     )
                    detects_.append(ax2detects_)
                    if len(detects_) == 4 or i_ == end2i :        
                        detects_ = np.stack(detects_)
                        inds = np.argmax(detects_[...,4],axis=0)
                        detects__ = np.take_along_axis(detects_,inds[None,...,None],axis=0)
                        # we now expect detects__ to have a singleton dimension at the beginning
                        if detects__.shape[0] == 1: 
                            detects__ = detects__[0]
                        else:
                            raise Exception()
                        detects.append(detects__)        
                        detects_ = []        
                ax2combinedetects = np.stack(detects,axis=2)
                print('Finished combining detections along sagittal view . . .')
                
                # Initialize the data cube
                datacubenew = np.zeros(ax1combinedetects.shape[:3]+(10,)) # 6 for xyzminmax, 1 prob, 3 features

                print(f'{ax0combinedetects.shape},{ax1combinedetects.shape},{ax2combinedetects.shape}')
                
                # Update confidence values at every voxel using the minimum probability
                datacubenew[...,6] = np.min(np.stack((ax0combinedetects[...,4],ax1combinedetects[...,4],ax2combinedetects[...,4])),0)
                
                # Update the feature labels at every voxel using the avg value
                for i_ in range(3):
                    datacubenew[...,7+i_] = (ax0combinedetects[...,5+i_] + ax1combinedetects[...,5+i_] + ax2combinedetects[...,5+i_])/3
                
                # Update the bbox features using min for xmin/ymin/zmin, and max for xmax/ymax/zmax
                datacubenew[...,0] = np.minimum( ax1combinedetects[...,1], ax2combinedetects[...,1] )  
                datacubenew[...,3] = np.maximum( ax1combinedetects[...,3], ax2combinedetects[...,3] )  
                
                # the 1 coordinate (row)
                # this is ax0 the 1th coordinate
                # and ax2 the 0th coordinate
                datacubenew[...,1] = np.minimum( ax0combinedetects[...,1], ax2combinedetects[...,0] )  
                datacubenew[...,4] = np.maximum( ax0combinedetects[...,3], ax2combinedetects[...,2] )  
                
                # the 2 coordinate (col)
                # this is ax0 the 0th coordinate
                # and ax1 the 0th coordinate
                datacubenew[...,2] = np.minimum( ax0combinedetects[...,0], ax1combinedetects[...,0] )  
                datacubenew[...,5] = np.maximum( ax0combinedetects[...,2], ax1combinedetects[...,2] )
    
                # Save output cube
                outname = os.path.join(outdir,f'i{i:04d}_j{j:04d}_k{k:04d}_d{d}_Ex_488_Em_525.npy')
                datacubeoutput = datacubenew.copy()
                
                # change from 012012 to 001122 (xmin, ymin, zmin, xmax, ymax, zmax) to (xmin, xmax, ymin, ymax, zmin, zmax)
                permutation = [0,3,1,4,2,5]
                datacubeoutput[...,:6] = datacubenew[...,permutation]
                
                np.save(outname, datacubeoutput)
    
                print(f'Finished cube ({i},{j},{k}) in {time.time()-start_time:.2f}s\n')

    return datacubeoutput

def extent_from_x(xhighres):
    dxhighres = [x[1] - x[0] for x in xhighres]

    extenthighres0 = (xhighres[-1][0]-dxhighres[-1]/2, xhighres[-1][-1]+dxhighres[-1]/2, xhighres[-2][-1]+dxhighres[-2]/2, xhighres[-2][0]-dxhighres[-2]/2)
    extenthighres1 = (xhighres[-1][0]-dxhighres[-1]/2, xhighres[-1][-1]+dxhighres[-1]/2, xhighres[-3][-1]+dxhighres[-3]/2, xhighres[-3][0]-dxhighres[-3]/2)
    extenthighres2 = (xhighres[-2][0]-dxhighres[-2]/2, xhighres[-2][-1]+dxhighres[-2]/2, xhighres[-3][-1]+dxhighres[-3]/2, xhighres[-3][0]-dxhighres[-3]/2)

    return extenthighres0,extenthighres1,extenthighres2