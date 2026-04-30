"""
IN PROGRESS . . .
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from matplotlib.collections import PolyCollection
import os

from scipy.ndimage import gaussian_filter

class GroundTruthDataset3D(torch.utils.data.Dataset):
    """ This dataset will generate a set of random images of size N x N. And they will contain poisson distributed cells with mean M.
    
    Parameters for Initialization:
    ------------------------------
    N : int
        The number of images to generate for this simulated dataset; Default - 256
    M : int
        The mean number of cells appended to each image, sampled from a Poisson distribution; Default - 350
    nclasses : int
        The number of classes to categorize this collection of simulated cells; Default = 3
    reproducible : bool
        If True, set np.random.seed() to the integer index supplied
        
    Returns:
    --------
    out_arr : 3 x N array
        - out_arr[0] is a 1 x N x N array defining the base image
        - out_arr[1] is a X x 4 array containing the bbox info for X cells
        - out_arr[2] is a X x 1 array containing the categorical label for the cell within the corresponding bbox; 0 - smooth boundary, 1 - sharp boudnary; 2 - bumpy boundary
        
    """
    
    def __init__(self,N=256,M=64,nclasses=3,reproducible=False):
        M = 256 # denser
        M = 350
        self.N = N
        self.M = M
        self.reproducible = reproducible
        self.nclasses = nclasses
        
        
    def __len__(self):
        #return 1000
        return 100
        #return 1
        
    def __getitem__(self,i):
        if self.reproducible: np.random.seed(i)
        
        # initialize the image
        d = np.random.randint(4)+1
        bg = np.random.rand(d)
        I = np.zeros((d,self.N,self.N,self.N)) + bg[...,None,None,None]
        
        # how many cells
        # m = np.random.poisson(self.M)
        m = 1+np.random.randint(self.M)
        
        # sample conditionally uniform centers
        c = np.random.rand(m,2)
        c = c * (np.array(I.shape[1:])-1)
        
        # sample the sizes, they will all be isotropic here to start
        # s = 10*np.exp(np.random.randn(m,2)*0.2)
        # s = 5*np.exp(np.random.randn(m,2)*0.2 + 1) # try smaller, but with variance
        s = 3 + np.random.rand(m,3)*20 # 
        
        
        # note bbox is xy not row col
        # theta = (np.random.rand(m)-0.5)*np.pi/2
        theta = np.random.rand(m)*np.pi*2

        # TODO: NO ROTATION FOR NOW
        theta = theta*0

        # we will use the original s'ss below
        # but define a new variable for the bounding boxes
        sr = np.stack((
            np.sqrt(  np.cos(theta)**2*s[:,0]**2 + np.sin(theta)**2*s[:,1]**2 ),
            np.sqrt(  np.cos(theta)**2*s[:,1]**2 + np.sin(theta)**2*s[:,0]**2 ),
            s[:,2],
        ),-1)
        bbox = np.stack((c[:,0]-sr[:,0]/2, c[:,1]-sr[:,1]/2, c[:,2]-sr[:,2]/2, sr[:,0], sr[:,1], sr[:,2] ), -1)
        
        # now the image
        rows = np.arange(I.shape[1])
        cols = np.arange(I.shape[2])
        slices = np.arange(I.shape[3])
        Rows,Cols,Slices = np.meshgrid(rows,cols,slices,indexing='ij')
        
        #cl = (np.random.rand(m)>0.5).astype(int) # will be 0 or 1, later they will be 1,2 with 0 reserved for background
        cl = np.random.randint(low=0,high=self.nclasses,size=m)
        #cl = cl*0+2
        
        color0 = np.random.rand(d)
        for ci,si,cli,ti in zip(c,s,cl,theta):
            # get a window
            mxi = np.max(si)
            left = np.maximum(ci[0]-3*mxi,0).astype(int)
            right = np.minimum(ci[0]+3*mxi,I.shape[-1]-1).astype(int)
            top = np.maximum(ci[1]-3*mxi,0).astype(int)
            bottom = np.minimum(ci[1]+3*mxi,I.shape[-2]-1).astype(int)
            front = np.maximum(ci[2]-3*mxi,0).astype(int)
            back = np.minimum(ci[2]+3*mxi,I.shape[-3]-1).astype(int)
            
            # ROTATION                        
            theta = ti
            Rows_ = (Rows-ci[1])*np.cos(theta) - (Cols-ci[0])*np.sin(theta)
            Cols_ = (Rows-ci[1])*np.sin(theta) + (Cols-ci[0])*np.cos(theta)
            Slices_ = Slices - ci[2] # TODO: No rotation about z-axis for now
            
            if cli <= 1:
                mask = np.exp(-(((Rows_[...,top:bottom,left:right,front:back]/si[1])**2 + (Cols_[...,top:bottom,left:right,front:back]/si[0])**2)/2*3**2 + (Slices_[...,top:bottom,left:right,front:back]/si[2])**2)**(1 + 2*cli)) 
            else:
                mask = np.exp(-(((Rows_[...,top:bottom,left:right,front:back]/si[1])**2 + (Cols_[...,top:bottom,left:right,front:back]/si[0])**2)/2*3**2 + (Slices_[...,top:bottom,left:right,front:back]/si[2])**2)**4 )
                bumps = np.random.randn(d,mask.shape[0],mask.shape[1],mask.shape[2])
                bumps = gaussian_filter(bumps,1)
                mask = np.exp(bumps)*mask
                
            color = color0*0.95 + np.random.rand(d)*0.05
            
            I[...,top:bottom,left:right,fonrt:back] = I[...,top:bottom,left:right,front:back] + color[...,None,None,None]*mask
        # artifact?

        # # I'd like to add a line
        # theta = np.random.rand()*2.0*np.pi
        # perp = (Rows - np.random.rand()*I.shape[-2])*np.cos(theta) + (Cols - np.random.rand()*I.shape[-1])*np.sin(theta)
        # mask = np.exp(-(perp/(1+np.random.rand()*20))**2)
        # color = np.random.rand(d)*2
        
        # #I = I*(1-mask) +color*mask
        # # just add
        # I = I + color[...,None,None]*mask
        
        
        # # and add a plane
        # theta = np.random.rand()*2.0*np.pi
        # perp = (Rows - np.random.rand()*I.shape[-2])*np.cos(theta) + (Cols - np.random.rand()*I.shape[-1])*np.sin(theta)
        # mask = 1.0/(1.0 +  np.exp(-(perp/(1+np.random.rand()*10))) )
        # color = np.random.rand(d)
        # I = I + color[...,None,None]*mask
        
        
        if np.random.rand() > 0.5:
            mu = np.random.rand()*4            
            I = np.exp(-mu*I)
        # random blur
        sigma = np.random.rand()*2
        I = gaussian_filter(I,sigma)
        
        # ad noise of random magnitude
        noise = np.random.rand(*I.shape)*np.random.rand()*0.5
        sigma = np.random.rand()*2
        noise = gaussian_filter(noise,sigma)        
        I = I + noise
        
        
        return I,bbox,cl

class Net(torch.nn.Module):
    """
    A neural network using the YOLO framework, with a batch size of 1 and an input layer capable of accepting inputs of variable shape. 

    Parameters:
    -----------
    nclasses : int
        Default - 3; The number of distinct classes in the dataset

    Returns:
    --------
    Net : torch.nn.Module
        A neural network which can take input images with any number of channels
    
    """
    def __init__(self, nclasses = 3):
        """
        Init method for the yolo network class. Most hyperparameters are hard-coded (See the comments within the below function)
        
        Parameters:
        -----------
        nclasses : int
            Default - 3; The number of distinct classes in the dataset
        """
        super().__init__()
        
        self.chin = 16 # The number of channels are VariableInputConv2D() layer will map to
        self.ch0 = 32 # The number of channels our first convolution layer will map to. Every other conv layer halves the resolution and doubles the number of channels.
        
        self.padding_mode='reflect' # Padding for convolution, reflect behaves nicely at boundaries
        
        self.B = 2 # bounding boxes per block (The original yolo model uses 2)
        self.C = nclasses # number of classes, conditioned on their being an object
        self.chout = self.B*5 + self.C # 5 numbers per box, cx, cy, width, height, confidence (confidence is a predition of p(object)*IOU). AND 1 probability score per class.
        
        self.color = VariableInputConv2d(self.chin)

        self.bn_kwargs = {'track_running_stats':False, 'affine':True}
        
        # self.bn = torch.nn.BatchNorm2d # i also set track running stats to false, so eval mode will be the same as train mode
        self.bn = torch.nn.InstanceNorm3d # i also set track running stats to false, so eval mode will be the same as train mode
        #self.bn = BatchNorm2dRunningOnly
        
        self.c0 = torch.nn.Conv3d(self.chin,self.ch0,3,1,1,padding_mode=self.padding_mode) # no downsampling
        self.b0 = self.bn(self.ch0,**self.bn_kwargs)
        self.c0a = torch.nn.Conv3d(self.ch0,self.ch0,3,1,1,padding_mode=self.padding_mode)
        self.b0a = self.bn(self.ch0,**self.bn_kwargs)
        
        self.c1 = torch.nn.Conv3d(self.ch0,self.ch0*2,3,2,1,padding_mode=self.padding_mode)
        self.b1 = self.bn(self.ch0*2,**self.bn_kwargs)
        self.c1a = torch.nn.Conv3d(self.ch0*2,self.ch0*2,3,1,1,padding_mode=self.padding_mode)
        self.b1a = self.bn(self.ch0*2,**self.bn_kwargs)
        
        self.c2 = torch.nn.Conv3d(self.ch0*2,self.ch0*4,3,2,1,padding_mode=self.padding_mode)
        self.b2 = self.bn(self.ch0*4,**self.bn_kwargs)
        self.c2a = torch.nn.Conv3d(self.ch0*4,self.ch0*4,3,1,1,padding_mode=self.padding_mode)
        self.b2a = self.bn(self.ch0*4,**self.bn_kwargs)
        
        
        
        self.c3 = torch.nn.Conv3d(self.ch0*4,self.ch0*8,3,2,1,padding_mode=self.padding_mode)
        self.b3 = self.bn(self.ch0*8,**self.bn_kwargs)
        self.c3a = torch.nn.Conv3d(self.ch0*8,self.ch0*8,3,1,1,padding_mode=self.padding_mode)
        self.b3a = self.bn(self.ch0*8,**self.bn_kwargs)
        
        
        self.c4 = torch.nn.Conv3d(self.ch0*8,self.chout,1,1,padding_mode=self.padding_mode)
        

        
        # the total stride is important for interpretting the bounding box in the rpn
        self.stride = 8        

        
        # the total stride is important for interpretting the bounding box in the rpn
        self.stride = 8
        
        
    def forward(self,x):
        """
        Forward method for the yolo network.
        
        Inputs
        ======
        x : torch.Tensor 
            Object is of size 1 x CH x ROW x COL. # of channels, rows, and columns are arbitrary. This differs from the original yolo paper, which requires a fixed number of channels, rows, and columns.
            
        Outputs
        =======
        x : torch.Tensor
            Object is of size 5 * bbox_per_cell (2) + num_classes (3) x ROW x COL, where ROW and COL are equal to the input image size divided by the network's stride (8). The five numbers per cell are [cx, cy, scalex, scaley, confidence].

        """
        
        # color         
        x = self.color(x)
        
        # here x is a batch of images
        x = self.c0(x)
        x = self.b0(x)
        x = torch.relu(x)
        
        x = self.c0a(x)
        x = self.b0a(x)
        x = torch.relu(x)
        
        
        
        x = self.c1(x)
        x = self.b1(x)
        x = torch.relu(x)
        
        x = self.c1a(x)
        x = self.b1a(x)
        x = torch.relu(x)
        

        
        
        x = self.c2(x)
        x = self.b2(x)
        x = torch.relu(x)

        x = self.c2a(x)
        x = self.b2a(x)
        x = torch.relu(x)

        
        x = self.c3(x)
        x = self.b3(x)
        x = torch.relu(x)
        
        x = self.c3a(x)
        x = self.b3a(x)
        x = torch.relu(x)
        
        x = self.c4(x)
        
        
        return x
        
class VariableInputConv2d(torch.nn.Module):
    ''' Note the assumption here is that we have batch size one, so I can work with the batch dimension.
    This version adds a few extra convolutions.
    
    We only process 1 image at a time, so we can use the batch dimension.
    
    Step 1: move channel dimension to batch dimension.  Now we have N samples, with one channel each.
    
    Step 2: apply some convolutions and relus, to end up with an N x M array of images. Where M is fixed, 
            and N is variable.
    
    Step 3: Take a softmax of the result over the N channels.  Now matrix multiply, and N x M array, with a N x 1 array,
            to get an M x 1 array (where M is fixed).
    
    Step 4: Return the result which is a fixed number of channels.
    
    What's nice is it is permutation invariant.  So if we input an RGB image, or a BGR image, the result would be exactly the same.
    We don't need to have the channels in any specific order.
    
    WORKING TODO
    Change so it supports a batch dimension
    This requires every image in a batch to have the same number of channels, but I think its okay
    
    '''
    def __init__(self,M):
        super().__init__()
        self.M = M
        self.c0  = torch.nn.Conv3d(1,M,3,1,1)
        self.c1  = torch.nn.Conv3d(M,M,3,1,1)
        self.c2  = torch.nn.Conv3d(M,M,3,1,1)
        
        
    def forward(self,x):
        #print(x.shape)
        # move the batch dim, make the size N x 1 (NO! not any more)
        nbatch = x.shape[0]
        nchannels = x.shape[1] # this is variable, but will be the same for every image in a batch
        
        
        #print(x.shape)
        # reshape B x N into (BN) x 1
        x_ = x.reshape( nbatch*nchannels, 1, x.shape[-3], x.shape[-2],x.shape[-1] )
        #print(x_.shape)
        cx = torch.relu(self.c0(x_))
        # after the first layer we now have (BN) x M
        #print(cx.shape)
        cx = torch.relu(self.c1(cx))
        # after the second layer again (BN) x M
        cx = self.c2(cx)
        #print(cx.shape)
        # now reshape it to B X N X M        
        cx = cx.reshape(nbatch,nchannels,self.M, x.shape[-3], x.shape[-2],x.shape[-1])
        #print(cx.shape)
        cx = torch.softmax(cx*100,1)
        #print(cx.shape)
        
        # now matrix multiply over the N axis
        out = torch.sum(x[:,:,None]*cx,1,keepdims=False)
        self.out = out
        return out

def apply_model_to_orthogonal_slices(model_path, I):
    """
    Apply the 2D YOLO model to all the slices in each orothognal view of the volume I

    Parameters:
    ===========
    model_path : str
        The path to the saved weights of the pretrained YOLO model
    I : array
        The 3D volume to be processed by the 2D YOLO model. Expects shape [N,N,N,d] where N is the desired cube size and d is the number of channels.

    Returns:
    ========
    recon_xy: array
        The 3D reconstruction generated from applying the model to XY slices. 
    recon_xz: array
        The 3D reconstruction generated from applying the model to XZ slices. 
    recon_yz: array
        The 3D reconstruction generated from applying the model to YZ slices. 
    """
    
    # Load the pretrained model
    net = Net()
    net.load_state_dict(torch.load(model_path))
    
    # Define some hyperparameters
    tile_dim = I.shape[0]
    ds_factor = 8
    bbox_dim = 5
    num_classes = 3
    B = net.B
    stride = net.stride
    dtype = torch.float32
    pads = np.array([0,0])
    
    # Initialize reconstruction containers
    recon_xy = np.ones((int(tile_dim/ds_factor), int(tile_dim/ds_factor), tile_dim, bbox_dim+num_classes))*(-np.inf)
    recon_xz = np.ones((int(tile_dim/ds_factor), tile_dim, int(tile_dim/ds_factor), bbox_dim+num_classes))*(-np.inf)
    recon_yz = np.ones((tile_dim, int(tile_dim/ds_factor), int(tile_dim/ds_factor), bbox_dim+num_classes))*(-np.inf)
    
    start = time.time()
    for idx in np.arange(tile_dim):
        slice_xy = I[:,:,idx,:]
        slice_xz = I[:,idx,:,:]
        slice_yz = I[idx,:,:,:]
    
        # Apply model to slices parallel to the XY-plane
        in_xy = torch.tensor(slice_xy, dtype=dtype).permute(2, 0, 1).unsqueeze(0)
        out_xy = net(in_xy)
        out_xy = out_xy[0].detach().numpy()
        out_xy = postprocess(out_xy, B, stride, pads, up_factor=1)
        recon_xy[:,:,idx,:] = out_xy
    
        # Apply model to slices parallel to the XZ-plane
        in_xz = torch.tensor(slice_xz, dtype=dtype).permute(2, 0, 1).unsqueeze(0)
        out_xz = net(in_xz)
        out_xz = out_xz[0].detach().numpy()
        out_xz = postprocess(out_xz, B, stride, pads, up_factor=1)
        recon_xz[:,idx,:,:] = out_xz   
        
        # Apply model to slices parallel to the YZ-plane
        in_yz = torch.tensor(slice_yz, dtype=dtype).permute(2, 0, 1).unsqueeze(0)
        out_yz = net(in_yz)
        out_yz = out_yz[0].detach().numpy()
        out_yz = postprocess(out_yz, B, stride, pads, up_factor=1)
        recon_yz[idx,:,:,:] = out_yz    
    
        if idx % 25 == 0:
            print(f'Finished {idx}/{tile_dim}')
    
    print(f'Finished applying model to 3 orthogonal views in {time.time()-start:.2f}s')
    return recon_xy, recon_xz, recon_yz

def gen_3D_GT(N, M, nclasses, thr_rel=0.15):
    """
    Generate a synthetic 3D volume with ellipsoidal blobs ("cells") and several artifacts
    commonly found in natural microscopy data.

    Output volume is channels-last: (Z, Y, X, d).

    Bounding boxes are computed by:
      1) Synthesizing each object's mask in a local ROI
      2) Thresholding the mask at (thr_rel * max(mask))
      3) Taking the tight axis-aligned bounding box of the thresholded support

    Parameters:
    ===========
    N : int
        Spatial size for X, Y, and Z (volume is N x N x N).
    M : int
        Upper bound for number of objects (m = 1 + randint(M)).
    nclasses : int
        Number of classes (labels in [0, nclasses-1]).
    thr_rel : float
        Relative threshold for bbox extraction: keep = mask > thr_rel * mask.max().

    Returns:
    ========
    I : np.ndarray
        Image volume of shape (N, N, N, d) float32  (Z, Y, X, d)
    bbox : np.ndarray
        Bounding boxes of shape (m, 6) float32
        Each row: (x0, y0, z0, w, h, depth) with half-open extents
    cl : np.ndarray
        Class labels of shape (m,)
    """
    # number of channels
    d = 3
    bg = np.random.rand(d).astype(np.float32)

    # channels-last volume: (Z, Y, X, d)
    I = np.zeros((N, N, N, d), dtype=np.float32) + bg[None, None, None, :]

    # how many objects
    m = 1 + np.random.randint(M)
    cl = np.random.randint(0, nclasses, size=m)

    # centers in voxel coordinates: (x, y, z)
    c = (np.random.rand(m, 3) * (np.array([N, N, N]) - 1)).astype(np.float32)

    # sizes (sigmas) per axis: (sx, sy, sz)
    s = (3 + np.random.rand(m, 3) * 20).astype(np.float32)

    # Euler angles (rx, ry, rz)
    angles = (np.random.rand(m, 3) * 2 * np.pi).astype(np.float32)

    color0 = np.random.rand(d).astype(np.float32)
    bbox = np.zeros((m, 6), dtype=np.float32)

    def euler_R(rx, ry, rz):
        """Rotation matrix R = Rz @ Ry @ Rx."""
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx,  cx]], dtype=np.float32)
        Ry = np.array([[ cy, 0, sy],
                       [  0,  1,  0],
                       [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [ 0,   0, 1]], dtype=np.float32)
        return (Rz @ Ry @ Rx).astype(np.float32)

    for i in range(m):
        cx, cy, cz = c[i]
        sx_, sy_, sz_ = s[i]
        cli = int(cl[i])
        R = euler_R(*angles[i])

        # ROI for synthesis: center +/- 3*max(sigma)
        mxi = float(max(sx_, sy_, sz_))
        left   = int(np.clip(cx - 3*mxi, 0, N - 1))
        right  = int(np.clip(cx + 3*mxi, 0, N - 1))
        top    = int(np.clip(cy - 3*mxi, 0, N - 1))
        bottom = int(np.clip(cy + 3*mxi, 0, N - 1))
        front  = int(np.clip(cz - 3*mxi, 0, N - 1))
        back   = int(np.clip(cz + 3*mxi, 0, N - 1))

        # enforce non-empty half-open ranges
        right  = max(right,  left + 1)
        bottom = max(bottom, top + 1)
        back   = max(back,   front + 1)

        # local coordinate blocks (broadcasted)
        zz = (np.arange(front, back, dtype=np.float32) - cz)[:, None, None]   # (Dz,1,1)
        yy = (np.arange(top, bottom, dtype=np.float32) - cy)[None, :, None]   # (1,Dy,1)
        xx = (np.arange(left, right, dtype=np.float32) - cx)[None, None, :]   # (1,1,Dx)

        # rotate into object frame: P' = R^T P
        Xp = R[0, 0]*xx + R[1, 0]*yy + R[2, 0]*zz
        Yp = R[0, 1]*xx + R[1, 1]*yy + R[2, 1]*zz
        Zp = R[0, 2]*xx + R[1, 2]*yy + R[2, 2]*zz

        # ellipsoidal metric
        q = (Xp/(sx_ + 1e-6))**2 + (Yp/(sy_ + 1e-6))**2 + (Zp/(sz_ + 1e-6))**2
        base = (q/2.0)*(3.0**2)

        # class-dependent mask shape + optional bumps
        if cli <= 1:
            mask = np.exp(-(base)**(1 + 2*cli)).astype(np.float32)
        else:
            mask = np.exp(-(base)**4).astype(np.float32)
            bumps = gaussian_filter(np.random.randn(*mask.shape).astype(np.float32), 1)
            mask *= np.exp(bumps).astype(np.float32)

        # add object to volume
        color = (color0*0.95 + np.random.rand(d).astype(np.float32)*0.05).astype(np.float32)
        I[front:back, top:bottom, left:right, :] += mask[..., None] * color[None, None, None, :]

        # bbox from thresholded mask support within ROI
        thr = float(mask.max()) * float(thr_rel)
        keep = mask > thr

        if np.any(keep):
            zz_idx, yy_idx, xx_idx = np.where(keep)
            z0 = front + int(zz_idx.min())
            z1 = front + int(zz_idx.max()) + 1  # half-open
            y0 = top   + int(yy_idx.min())
            y1 = top   + int(yy_idx.max()) + 1
            x0 = left  + int(xx_idx.min())
            x1 = left  + int(xx_idx.max()) + 1
        else:
            # fallback: 1-voxel box at rounded center (rare)
            x0 = int(np.clip(round(cx), 0, N-1))
            x1 = min(x0 + 1, N)
            y0 = int(np.clip(round(cy), 0, N-1))
            y1 = min(y0 + 1, N)
            z0 = int(np.clip(round(cz), 0, Z-1))
            z1 = min(z0 + 1, Z)

        bbox[i] = np.array([x0, y0, z0, x1-x0, y1-y0, z1-z0], dtype=np.float32)

    # --- global artifacts (computed without allocating a full meshgrid) ---
    z = np.arange(N, dtype=np.float32)[:, None, None]
    y = np.arange(N, dtype=np.float32)[None, :, None]
    x = np.arange(N, dtype=np.float32)[None, None, :]

    # Gaussian "sheet" artifact (thin plane)
    n = np.random.randn(3).astype(np.float32)
    n /= (np.linalg.norm(n) + 1e-6)
    p0 = np.array([np.random.rand()*(N-1),
                   np.random.rand()*(N-1),
                   np.random.rand()*(N-1)], dtype=np.float32)
    dist = n[0]*(x - p0[0]) + n[1]*(y - p0[1]) + n[2]*(z - p0[2])
    width = np.float32(1 + np.random.rand()*20)
    sheet = np.exp(-(dist/width)**2).astype(np.float32)
    I += sheet[..., None] * (np.random.rand(d).astype(np.float32)*2)[None, None, None, :]

    # Logistic "slab" artifact (smooth plane transition)
    n2 = np.random.randn(3).astype(np.float32)
    n2 /= (np.linalg.norm(n2) + 1e-6)
    p1 = np.array([np.random.rand()*(N-1),
                   np.random.rand()*(N-1),
                   np.random.rand()*(N-1)], dtype=np.float32)
    dist2 = n2[0]*(x - p1[0]) + n2[1]*(y - p1[1]) + n2[2]*(z - p1[2])
    sharp = np.float32(1 + np.random.rand()*10)
    slab = (1.0/(1.0 + np.exp(-(dist2/sharp)))).astype(np.float32)
    I += slab[..., None] * np.random.rand(d).astype(np.float32)[None, None, None, :]

    # optional intensity warp
    if np.random.rand() > 0.5:
        mu = np.float32(np.random.rand()*4)
        I = np.exp(-mu * I).astype(np.float32)

    # blur (3D per channel)
    sigma = np.random.rand()*2
    for ch in range(d):
        I[..., ch] = gaussian_filter(I[..., ch], sigma)

    # additive smooth noise
    noise = (np.random.rand(*I.shape).astype(np.float32) * (np.random.rand()*0.5)).astype(np.float32)
    sigma_n = np.random.rand()*2
    for ch in range(d):
        noise[..., ch] = gaussian_filter(noise[..., ch], sigma_n)
    I += noise

    return I, bbox, cl

def orthogonal_to_3D(down_xy, down_xz, down_yz, tile_dim = 256):
    """
    Stitch together 3 volumes of orthogonal slices with 2D bbox info stored at each voxel into 1 volume with 3D bbox info stored at each voxel.

    Parameters:
    ===========
    down_xy: array
        The 3D reconstruction of XY slices, downsampled by a factor of 'ds_factor'. 
    down_xz: array
        The 3D reconstruction of XZ slices, downsampled by a factor of 'ds_factor'. 
    down_yz: array
        The 3D reconstruction of YZ slices, downsampled by a factor of 'ds_factor'. 
    tile_dim : int
        The size of the original 3D reconstruction

    Returns:
    ========
    
    """
    start = time.time()
    ds_factor = 8
    down_dim = int(tile_dim/ds_factor)
    
    bbox_dim = 7
    num_classes = 3
    recon_total = np.ones((down_dim, down_dim, down_dim, bbox_dim+num_classes))*(-np.inf)
    for i in np.arange(down_dim):
        for j in np.arange(down_dim):
            for k in np.arange(down_dim):
            
                # From meeting 02/26/26
                xmin = (down_xy[i,j,k,1] + down_xz[i,j,k,1]) / 2
                xmax = (down_xy[i,j,k,3] + down_xz[i,j,k,3]) / 2
                ymin = (down_xy[i,j,k,0] + down_yz[i,j,k,1]) / 2
                ymax = (down_xy[i,j,k,2] + down_yz[i,j,k,3]) / 2
                zmin = (down_xz[i,j,k,0] + down_yz[i,j,k,0]) / 2
                zmax = (down_xz[i,j,k,2] + down_yz[i,j,k,2]) / 2
                
                # conf = np.max([down_xy[i,j,k,4], down_xz[i,j,k,4], down_yz[i,j,k,4]])
                conf = np.min([down_xy[i,j,k,4], down_xz[i,j,k,4], down_yz[i,j,k,4]])
                # conf = down_xy[i,j,k,4] * down_xz[i,j,k,4] * down_yz[i,j,k,4]
                # conf = np.sum([down_xy[i,j,k,4], down_xz[i,j,k,4], down_yz[i,j,k,4]]) / 3
                
                cl0 = np.max([down_xy[i,j,k,5], down_xz[k,i,j,5], down_yz[j,k,i,5]])
                cl1 = np.max([down_xy[i,j,k,6], down_xz[k,i,j,6], down_yz[j,k,i,6]])
                cl2 = np.max([down_xy[i,j,k,7], down_xz[k,i,j,7], down_yz[j,k,i,7]])
                
                recon_total[i,j,k] = [xmin, xmax, ymin, ymax, zmin, zmax, conf, cl0, cl1, cl2]
    
    print(f'Finished combining outputs in {time.time()-start:.2f}s')
    return recon_total
        
def recon_down(recon_xy, recon_xz, recon_yz, tile_dim = 256):
    """
    Dowsample each volume along their respective 'slice' dimensions by taking the bbox of highest confidence at every pixel across 'ds_factor' consecutive slices.

    Parameters:
    ===========
    recon_xy: array
        The 3D reconstruction generated from applying the model to XY slices. 
    recon_xz: array
        The 3D reconstruction generated from applying the model to XZ slices. 
    recon_yz: array
        The 3D reconstruction generated from applying the model to YZ slices. 

    Returns:
    ========
    down_xy: array
        The 3D reconstruction of XY slices, downsampled by a factor of 'ds_factor'. 
    down_xz: array
        The 3D reconstruction of XZ slices, downsampled by a factor of 'ds_factor'. 
    down_yz: array
        The 3D reconstruction of YZ slices, downsampled by a factor of 'ds_factor'. 
    """
    conf_idx = 4
    ds_factor = 8
    down_dim = int(tile_dim/ds_factor)
    
    down_xy = []
    down_xz = []
    down_yz = []
    
    start = time.time()
    for idx in np.arange(down_dim):
    
        # Extract subset of 8 XY-slices to be downsampled into 1 slice
        recon_xy_down = recon_xy[:,:,(idx*ds_factor):((idx+1)*ds_factor),:]
    
        # Get idx of bbox with highest confidence at each pixel
        inds_xy = np.argmax(recon_xy_down[..., conf_idx], axis = 2)
    
        # Downsample along relevant axis using the above indeces
        recon_xy_down = np.take_along_axis(recon_xy_down, inds_xy[:, :, None, None], axis=2)
    
        # Append downsampled slice to final output for XY-slices
        down_xy.append(recon_xy_down)
    
        # Repeat for XZ-slices
        recon_xz_down = recon_xz[:,(idx*ds_factor):((idx+1)*ds_factor),:,:]
        inds_xz = np.argmax(recon_xz_down[..., conf_idx], axis = 1)
        recon_xz_down = np.take_along_axis(recon_xz_down, inds_xz[:, None, :, None], axis=1)
        down_xz.append(recon_xz_down)
    
        # Repear for YZ-slices
        recon_yz_down = recon_yz[(idx*ds_factor):((idx+1)*ds_factor),:,:,:]
        inds_yz = np.argmax(recon_yz_down[..., conf_idx], axis = 0)
        recon_yz_down = np.take_along_axis(recon_yz_down, inds_yz[None, :, :, None], axis=0)
        down_yz.append(recon_yz_down)
    
    # Concat lists so that they are now each 32x32x32 cubes
    down_xy = np.concatenate(down_xy, axis=2)
    down_xz = np.concatenate(down_xz, axis=1)
    down_yz = np.concatenate(down_yz, axis=0)
    
    print(f'Finished downsampling outputs in {time.time()-start:.2f}s')
    return down_xy, down_xz, down_yz

# Downsample 3D image along one axis. Useful for overlaying bbox outputs on MIPs
def vol_to_mip(vol, ax_idx, ds_factor = 8, slice_idx = -1):
    """
    Convert a 3D image volume (vol) into a stack of 2D MIPs along an axis (ax_idx). 
    This is useful for visualizing outputs from the YOLO_3D pipeline.

    Parameters:
    ===========
    vol : array of shape N x M x K
        The image volume to be 'downsampled'
    ax_idx : int
        The axis along which MIPs should be generated
    ds_factor : int
        The number of slices to use in each MIP
    slice_idx : int
        Default - -1; If -1, return the downsampled 3d volume of 2d slices along 'ax_idx'. Else, Return the MIP of [slice_idx:slice_idx+ds_factor] along 'ax_idx'.
        
    Returns:
    ========
    vol_mip : array 
        A stack of MIPs along the 'ax_idx' axis
    """
    if slice_idx == -1:
        vol_mip = []
        vol_dim = vol.shape[ax_idx]
        for i in np.arange(int(vol_dim/ds_factor)):
            
            if ax_idx == 0:
                vol_slice_stack = vol[(i*ds_factor):((i+1)*ds_factor), :, :]
            elif ax_idx == 1:
                vol_slice_stack = vol[:, (i*ds_factor):((i+1)*ds_factor), :]
            elif ax_idx == 2:
                vol_slice_stack = vol[:, :, (i*ds_factor):((i+1)*ds_factor)]
            else:
                raise Exception(f'vol only has {len(vol.shape)} axes, but ax_idx = {ax_idx} was passed')
                
            vol_slice_max = np.max(vol_slice_stack, axis=ax_idx)
            vol_mip.append(vol_slice_max)
                
        vol_mip = np.stack(vol_mip, axis=ax_idx)
        return vol_mip
    else:
        if ax_idx == 0:
            vol_slice_stack = vol[(slice_idx):(slice_idx+ds_factor-1), :, :]
        elif ax_idx == 1:
            vol_slice_stack = vol[:, (slice_idx):(slice_idx+ds_factor-1), :]
        elif ax_idx == 2:
            vol_slice_stack = vol[:, :, (slice_idx):(slice_idx+ds_factor-1)]
        else:
            raise Exception(f'vol only has {len(vol.shape)} axes, but ax_idx = {ax_idx} was passed')
            
        vol_slice_max = np.max(vol_slice_stack, axis=ax_idx)
        return vol_slice_max

def vol_to_mip_rgb(vol, ax_idx, ds_factor = 8):
    """
    Convert a 3D image volume (vol) into a stack of 2D MIPs along an axis (ax_idx). 
    This is useful for visualizing outputs from the YOLO_3D pipeline.

    Parameters:
    ===========
    vol : array of shape 256x256x256x3
        The image volume to be 'downsampled'
    ax_idx : int
        The axis along which MIPs should be generated
    ds_factor : int
        The number of slices to use in each MIP

    Returns:
    ========
    vol_mip : array 
        A stack of MIPs along the 'ax_idx' axis
    """
    vol_mip = []
    vol_dim = vol.shape[ax_idx]
    for i in np.arange(int(vol_dim/ds_factor)):
        
        if ax_idx == 0:
            vol_slice_stack = vol[(i*ds_factor):((i+1)*ds_factor), :, :, :]
        elif ax_idx == 1:
            vol_slice_stack = vol[:, (i*ds_factor):((i+1)*ds_factor), :, :]
        elif ax_idx == 2:
            vol_slice_stack = vol[:, :, (i*ds_factor):((i+1)*ds_factor), :]
        else:
            raise Exception(f'vol only has {len(vol.shape)} axes, but ax_idx = {ax_idx} was passed')
            
        vol_slice_max = np.max(vol_slice_stack, axis=ax_idx)
        vol_mip.append(vol_slice_max)
            
    vol_mip = np.stack(vol_mip, axis=ax_idx)
    return vol_mip