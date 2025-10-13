import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import matplotlib as mpl
import time
import torch
import random
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('/home/abenneck/Desktop/yolo_model/docs/scripts')
from yolo_help import Net

class tileDataset(Dataset):
    def __init__(self, tiles):
        self.images = [tile['img'] for tile in tiles]
        self.all_p  = [tile['p'] for tile in tiles]
        self.all_bg = [tile['bg'] for tile in tiles]
        self.all_r  = [tile['r_idx'] for tile in tiles]
        self.n_row = np.max(self.all_r) + 1
        self.all_c  = [tile['c_idx'] for tile in tiles]
        self.n_col = np.max(self.all_c) + 1
        # self.transform = ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        prcb = {'p':self.all_p[idx], 'r':self.all_r[idx], 'c':self.all_c[idx], 'bg':self.all_bg[idx]}
        if len(img.shape) == 2: # If image is grayscale, add a color dimension
            img = img[None]
        return torch.tensor(img, dtype=torch.float32), prcb

def preprocess(img, gamma = True, upsample = True):
    """
    Prepare an image for the YOLO tile pipeline by normalization, a gamma correction, and upsampling by a factor of 2

    Parameters:
    -----------
    img : array of shape [N,M]
        A microscopy image of tissue stained for nuclei
    gamma : bool
        Default - True; If True, apply a gamma correction where the gamma value is 1/2
    upsample : bool
        Default - True; If True, upsampled the image by a factor of 2

    Returns:
    --------
    img_up : array of shape [N*2, M*2]
        A normalized, gamma corrected, upsampled version of the input image 
    """
    # Normalize + gamma correction on input image
    if gamma:
        img = img[None] / np.max(img,axis=(-1,-2),keepdims=True)
        img = img**0.5
        if not upsample:
            return img[0]
    
    # Upsample this image since it contains NUCLEI, not cells
    if upsample:
        img_up_ = np.zeros((img.shape[0],img.shape[1],img.shape[2]*2))
        img_up_[:,:,0::2] = img
        img_up_[:,:,1:img_up_.shape[-1]-1:2] = img[:,:,:-1]*0.5 + img[:,:,1:]*0.5
        
        img_up = np.zeros((img.shape[0],img.shape[1]*2,img.shape[2]*2))
        img_up[:,0::2,:] = img_up_
        img_up[:,1:img_up.shape[-2]-1:2,:] = img_up_[:,:-1,:]*0.5 + img_up_[:,1:,:]*0.5
        img_up = img_up[0]

    return img_up
    

def load_test_image(img_dim0, img_dim1, spacing=16, r=4):
    """
    Generate an artificial image simulating bright cells on a dark background

    Parameters:
    -----------
    img_dim0 : float
        The desired size of the test image along axis 0
    img_dim1 : float
        The desired size of the test image along axis 1
    spacing : int
        Default - 16; The distance between 2 adjacent simulated cells
    r : int
        Default - 4; The radius of a simulated cell
        

    Returns:
    --------
    img : np.array of size [imgdim0, imgdim1]
        The test image
    total_bbox : int
        The numbe of simulated cells in the test image
    """
    
    img = np.zeros((img_dim0, img_dim1, 3))
    Y, X = np.ogrid[:img_dim0, :img_dim1]
    total_bbox = 0
    for i in range(int(img_dim0/spacing)):
        for j in range(int(img_dim1/spacing)):
            if i > 0 and j > 0:
                mask = (X - spacing*i)**2 + (Y - spacing*j)**2 <= r**2
                img[mask,:]=1
                total_bbox += 1
    return img, total_bbox

def img_to_tiles(img, outdir='', min_overlap = 32, tile_dim = 256, upper_threshold_bg = np.inf, lower_threshold_bg = -np.inf,verbose = False):
    """
    Convert an image into a set of tiles. Tiles will be of the shape [tile_dim, tile_dim, :] and overlap with each adjacent tile by 'min_overlap' pixels.

    Parameters:
    -----------
    img : np.array of size [N, M]
        The image from which tiles will be extracted
    outdir : str
        Default - ''; The location where tiles should be saved. If none supplied, tiles won't be saved in a .npz file, just returned by this function
    min_overlap : int
        Default - 32; The number of overlapping pixels between 2 adjacent tiles
    tile_dim : int
        Default - 256; The length of a tile along each axis
    upper_threshold_bg : int
        Default - inf; If the 95th percentile of a tile is >= this value, mark it as background. If a value is supplied, it will be used as the padding for the image
    lower_threshold_bg : int
        Default - -inf;  If the 95th percentile of a tile is <= this value, mark it as background. If a value is supplied, it will be used as the padding for the image
    verbose : bool
        Default - False; If True, print out intermediate and final progress notes

    Returns:
    --------
    padded_img : np.array of size [N*, M*]
        The original image padded along all 4 sides with either 0 or the supplied background threshold value. This image is intended to be pushed through the rest of the YOLO tile pipeline.
    tiles : list(dict)
        A list containing all the tiles which comprise 'padded_img'. Each element is a dictionary containing 5 values: (1) The pixel values of the tile, (2) The anchor point for the tile used in reconstruction, (3) A boolean which defines the tile as background or foreground, (4) The row idx of the tile in the whole iamge, and (5) The col idx of the tile in the whole iamge
    
    """
    start = time.time()
    
    # Define dimensions of the input image
    if len(np.shape(img)) == 3:
        img_dim0, img_dim1, num_ch = img.shape
        ndim = 3
    elif len(np.shape(img)) == 2:
        img_dim0, img_dim1 = img.shape
        ndim = 2
        num_ch = 1
    else:
        raise Exception(f'Input image should be 2 or 3 dimensions, but {len(img.shape)} were found')

    # Determine value to with which to pad the borders
    if np.isfinite(upper_threshold_bg):
        pad = upper_threshold_bg
    elif np.isfinite(lower_threshold_bg):
        pad = lower_threshold_bg
    else:
        pad = 0
    
    # Define the dimensions of the tiles to be extracted + minimum desired overlap between 2 adjacent tiles
    if outdir != '':
        os.makedirs(outdir, exist_ok=True)
    
    # If the 'threshold_perc'th percentile of a tile is >= 'threshold_bg', mark it as background
    threshold_perc = 0.95 # The percentile of all pixels in the tile to check against threshold_bg
    
    # Compute number of tiles needed along each axis + necessary padding to preserve stride across tiles
    tile_n0 = int(img_dim0 / (tile_dim-min_overlap)) + 1 # Number of tiles to extract along axis 0
    pad_dim0 = ((tile_n0-1)*(tile_dim - min_overlap) + tile_dim)
    
    tile_n1 = int(img_dim1 / (tile_dim-min_overlap)) + 1 # Number of tiles to extract along axis 1
    pad_dim1 = ((tile_n1-1)*(tile_dim - min_overlap) + tile_dim)
    
    # Pad the original input image with zeros
    left_idx = int((pad_dim0-img_dim0)/2)
    right_idx = left_idx+img_dim0
    upper_idx = int((pad_dim1-img_dim1)/2)
    lower_idx = upper_idx+img_dim1
    
    if ndim == 3:
        padded_img = np.ones((pad_dim0, pad_dim1, num_ch), dtype=int)*pad
        padded_img[left_idx:right_idx, upper_idx:lower_idx, :] = img
    else:
        padded_img = np.ones((pad_dim0, pad_dim1), dtype=int)*pad
        padded_img[left_idx:right_idx, upper_idx:lower_idx] = img

    # Determine the upper-left corner of each tile
    all_upper_left_corners = []
    for i in range(tile_n0):
        coord_dim0 = i*(tile_dim - min_overlap)
        if coord_dim0 + tile_dim >= pad_dim0:
            coord_dim0 = pad_dim0 - tile_dim
        for j in range(tile_n1):
            coord_dim1 = j*(tile_dim - min_overlap)
            if coord_dim1 + tile_dim >= pad_dim1:
                coord_dim1 = pad_dim1 - tile_dim
            all_upper_left_corners.append([int(coord_dim0), int(coord_dim1), i, j])

    # Extract all tiles from the image using the anchor points computed above
    tiles = []
    for idx, p in enumerate(all_upper_left_corners):
        idx0 = p[2]
        idx1 = p[3]
        p = p[:2]
    
        # Extract tile to be saved for downstream processing
        if ndim == 3:
            sub_img = padded_img[p[0]:p[0]+tile_dim, p[1]:p[1]+tile_dim,:]
            sub_img = np.transpose(sub_img, (2,0,1)) # Transpose image, so that it is in the proper format for the YOLO pipeline
        else:
            sub_img = padded_img[p[0]:p[0]+tile_dim, p[1]:p[1]+tile_dim]
            # sub_img = np.transpose(sub_img, (2,0,1)) # Transpose image, so that it is in the proper format for the YOLO pipeline
            
        
        if np.percentile(sub_img, threshold_perc) >= upper_threshold_bg:
            bg = True
        elif np.percentile(sub_img, threshold_perc) <= lower_threshold_bg:
            bg = True
        else:
            bg = False

        tile = {'img':sub_img, 'p':p, 'bg':bg, 'r_idx':idx0, 'c_idx':idx1}
        tiles.append(tile)
        # if verbose and idx % 100 == 0:
        #     print(f'Finished tile {idx}/{len(all_upper_left_corners)}')
    
    if outdir != '':
        np.savez(os.path.join(outdir, 'tiles.npz'), tiles=tiles)

    if verbose:
        print(f'Finished extracting all tiles in {time.time()-start:.2f}s')

    # Reverse the order, so now tiles will be parsed in a reverse lexicographic order
    tiles = tiles[::-1]
    
    return padded_img, tiles

def apply_model_to_tiles(tiles, model_path, img_dim0, img_dim1, out_path='', batch_mode = False, verbose=False):
    """
    Apply a pretrained YOLO model to each element of 'tiles'. Stitch together each of these model outputs in order to create a processed version of the original (padded) input image. Save this reconstructed output at 'out_path'.

    Parameters:
    -----------
    tiles : list(dict)
        A list containing all the tiles which comprise 'padded_img'. Each element is a dictionary containing 5 values: (1) The pixel values of the tile, (2) The anchor point for the tile used in reconstruction, (3) A boolean which defines the tile as background or foreground, (4) The row idx of the tile in the whole iamge, and (5) The col idx of the tile in the whole iamge
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
        The reconstructed model output where each element corresponds to a specific point in the original image. The 13 values in each element of the reconstruction ([cx0, cy0, w0, h0, conf0, cx1, cy1, w1, h1, conf1, cl0, cl1, cl2]) can be interpreted as follows. There are 2 predicted bounding boxes centered at (cx0, cy0) and (cx1,cy1) with their corresponding width, height, and confidence values defined by (w0, h0, conf0) and (w1, h1, conf1), respectively. cl0, cl1, and cl2 are the class probabilities and define the likelihood that a cell can be categorized as having a smooth, sharp, or bumpy boundary, respectively.
    """
    start = time.time()
    start_total = time.time()

    # Load pretrained model weights
    dtype = torch.float32
    net = Net()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    B = net.B
    stride = net.stride
    
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
    recon = np.zeros((bbox_dim*B+num_classes, int(img_dim0/ds_factor), int(img_dim1/ds_factor)))

    non_bg_tile = 0

    if batch_mode:
        nrow_t = int(np.max([tile['r_idx'] for tile in tiles])) + 1
        ncol_t = int(np.max([tile['c_idx'] for tile in tiles])) + 1
        
        ds = tileDataset(tiles)
        dl = DataLoader(ds, batch_size = ncol_t, shuffle=False)

        for i_batch, batch in enumerate(dl):
            imgs, meta_data = batch
            if np.all(np.array(meta_data['bg'])):
                if verbose:
                    print(f'Skipping batch {i_batch}/{nrow_t}')
                continue
            else:
                start = time.time()
                out = net(imgs)
                for idx, elem in enumerate(out):
                    out_i = remove_bbox_in_overlap(elem[None].clone().detach(), B, stride, tile_dim, boundary_cond = boundary_cond)
                    p0 = meta_data['p'][0][idx]
                    p1 = meta_data['p'][1][idx]
                    recon[:,int(p0/ds_factor):int((p0+tile_dim)/ds_factor), int(p1/ds_factor):int((p1+tile_dim)/ds_factor)] = out_i[0].detach().numpy()
                if verbose:
                    print(f'Finished batch {i_batch}/{nrow_t} in {time.time()-start:.2f}s')
                    start = time.time()
    else:
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
                # Apply trained model to the image + define some key model outputs
                out = net((torch.tensor(I[None],dtype=dtype)))
                out = remove_bbox_in_overlap(out, B, stride, tile_dim, boundary_cond = boundary_cond)
                
                # Append model output to reconstruction
                recon[:,int(p[0]/ds_factor):int((p[0]+tile_dim)/ds_factor), int(p[1]/ds_factor):int((p[1]+tile_dim)/ds_factor)] = out[0].detach().numpy()
    
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

def count_bbox(out):
    """
    Count the numebr of bounding boxes output by the YOLO model.

    Parameters:
    -----------
    out : torch.Tensor
        The unprocessed output from the YOLO model

    Returns:
    --------
    n0 : int
        The number of bounding boxes found in the YOLO model output
    """
    n0 = 0
    for elem in out[4]:
        n0 += len(np.where(np.isfinite(elem))[0])
    for elem in out[9]:
        n0 += len(np.where(np.isfinite(elem))[0])
    return n0

# Remove invalid bboxes from 'out' by setting the corresponding conf value in 'out' to -inf
def remove_bbox_in_overlap(out, B, stride, tile_dim, boundary_cond = [16,16]):
    """
    Remove bounding boxes that lie in the exclusion zone defined by 'boundary_cond' from the model output before appending to reconstruction.

    Parameters:
    -----------
    out : torch.Tensor of shape [13, tile_dim/stride, tile_im/stride]
        The output from the YOLO model after processing a tile_dim x tile_dim image.
    B : int
        the number of target classes used in the YOLO model
    stride : int
        The nmuber of pixels the kernel moves during each step of the YOLO model
    tile_dim : int
        The dimensions of a tile extracted from the original image
    boundary_cond : np.array(int)
        The thickness of the exclusion zones along each axis

    Returns:
    --------
    out : torch.Tensor of shape [13, tile_dim/stride, tile_im/stride]
        The original output from the model, but now bboxes that don't meet the filtering criterion have a confidence of -inf.
    """

    # Get the positions of the grid cells
    x = torch.arange(out.shape[-1])*stride + (stride-1)/2
    y = torch.arange(out.shape[-2])*stride + (stride-1)/2
    YX = torch.stack(torch.meshgrid(y,x,indexing='ij'),0)

    # Convert bbox0 data to dimensions that are relative to the original input
    outB0 = out[:,:5]
    x0 = (torch.sigmoid(outB0[:,0])-0.5)*stride + YX[1] # between -0.5 and 0.5, scaled
    y0 = (torch.sigmoid(outB0[:,1])-0.5)*stride + YX[0]
    w0 = torch.exp(outB0[:,2])*stride
    h0 = torch.exp(outB0[:,3])*stride
    left0 = x0 - w0/2
    upper0 = y0 - h0/2

    # Convert bbox1 data to dimensions that are relative to the original input
    outB1 = out[:,5:10]    
    x1 = (torch.sigmoid(outB1[:,0])-0.5)*stride + YX[1] # between -0.5 and 0.5, scaled
    y1 = (torch.sigmoid(outB1[:,1])-0.5)*stride + YX[0]
    w1 = torch.exp(outB1[:,2])*stride
    h1 = torch.exp(outB1[:,3])*stride
    left1 = x1 - w1/2
    upper1 = y1 - h1/2

    # Compute upper and left bounds for all bboxes AND generate a mask of which bboxes meet the filtering criteria
    part_of_bb0 = torch.logical_or(left0 < boundary_cond[0], upper0 < boundary_cond[1])
    entire_bb0 = torch.logical_or(left0 >= (tile_dim-boundary_cond[0]), upper0 >= (tile_dim-boundary_cond[1]))
    remove_bb0 = torch.logical_or(part_of_bb0, entire_bb0).bool() # A mask of which bboxes should be removed to prevent double counting
    
    part_of_bb1 = torch.logical_or(left1 < boundary_cond[0], upper1 < boundary_cond[1])
    entire_bb1 = torch.logical_or(left1 >= (tile_dim-boundary_cond[0]), upper1 >= (tile_dim-boundary_cond[1]))
    remove_bb1 = torch.logical_or(part_of_bb1, entire_bb1).bool() # A mask of which bboxes should be removed to prevent double counting
    
    # Using the above masks, change conf of bboxes which are ar risk of being double counted to -inf
    out = out.clone()
    neg_inf0 = torch.full_like(out[:, 4], -torch.inf)
    neg_inf1 = torch.full_like(out[:, 9], -torch.inf)
    out[:, 4] = torch.where(remove_bb0, neg_inf0, out[:, 4]) # Update conf values for bbox0
    out[:, 9] = torch.where(remove_bb1, neg_inf1, out[:, 9]) # Update conf values for bbox1

    return out