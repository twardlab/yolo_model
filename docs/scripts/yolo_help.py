import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from matplotlib.collections import PolyCollection
import os

from scipy.ndimage import gaussian_filter

class GroundTruthDataset(torch.utils.data.Dataset):
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
        I = np.zeros((d,self.N,self.N)) + bg[...,None,None]
        
        # how many cells
        # m = np.random.poisson(self.M)
        m = 1+np.random.randint(self.M)
        
        # sample conditionally uniform centers
        c = np.random.rand(m,2)
        c = c * (np.array(I.shape[1:])-1)
        
        # sample the sizes, they will all be isotropic here to start
        # s = 10*np.exp(np.random.randn(m,2)*0.2)
        # s = 5*np.exp(np.random.randn(m,2)*0.2 + 1) # try smaller, but with variance
        s = 3 + np.random.rand(m,2)*20 # 
        
        
        # note bbox is xy not row col
        # theta = (np.random.rand(m)-0.5)*np.pi/2
        theta = np.random.rand(m)*np.pi*2

        # we will use the original s'ss below
        # but define a new variable for the bounding boxes
        sr = np.stack((
            np.sqrt(  np.cos(theta)**2*s[:,0]**2 + np.sin(theta)**2*s[:,1]**2 ),
            np.sqrt(  np.cos(theta)**2*s[:,1]**2 + np.sin(theta)**2*s[:,0]**2 ),
        ),-1)
        bbox = np.stack((c[:,0]-sr[:,0]/2, c[:,1]-sr[:,1]/2, sr[:,0], sr[:,1]   ), -1)
        
        # now the image
        rows = np.arange(I.shape[1])
        cols = np.arange(I.shape[2])
        Rows,Cols = np.meshgrid(rows,cols,indexing='ij')
        
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
            
            # ROTATION                        
            theta = ti
            Rows_ = (Rows-ci[1])*np.cos(theta) - (Cols-ci[0])*np.sin(theta)
            Cols_ = (Rows-ci[1])*np.sin(theta) + (Cols-ci[0])*np.cos(theta)
            
            if cli <= 1:
                mask = np.exp(-(((Rows_[...,top:bottom,left:right]/si[1])**2 + (Cols_[...,top:bottom,left:right]/si[0])**2)/2*3**2)**(1 + 2*cli)) 
            else:
                mask = np.exp(-(((Rows_[...,top:bottom,left:right]/si[1])**2 + (Cols_[...,top:bottom,left:right]/si[0])**2)/2*3**2)**4 )
                bumps = np.random.randn(d,mask.shape[0],mask.shape[1])
                bumps = gaussian_filter(bumps,1)
                mask = np.exp(bumps)*mask
                
            color = color0*0.95 + np.random.rand(d)*0.05
            
            I[...,top:bottom,left:right] = I[...,top:bottom,left:right] + color[...,None,None]*mask
        # artifact?

        # I'd like to add a line
        theta = np.random.rand()*2.0*np.pi
        perp = (Rows - np.random.rand()*I.shape[-2])*np.cos(theta) + (Cols - np.random.rand()*I.shape[-1])*np.sin(theta)
        mask = np.exp(-(perp/(1+np.random.rand()*20))**2)
        color = np.random.rand(d)*2
        
        #I = I*(1-mask) +color*mask
        # just add
        I = I + color[...,None,None]*mask
        
        
        # and add a plane
        theta = np.random.rand()*2.0*np.pi
        perp = (Rows - np.random.rand()*I.shape[-2])*np.cos(theta) + (Cols - np.random.rand()*I.shape[-1])*np.sin(theta)
        mask = 1.0/(1.0 +  np.exp(-(perp/(1+np.random.rand()*10))) )
        color = np.random.rand(d)
        I = I + color[...,None,None]*mask
        
        
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
        self.bn = torch.nn.InstanceNorm2d # i also set track running stats to false, so eval mode will be the same as train mode
        #self.bn = BatchNorm2dRunningOnly
        
        self.c0 = torch.nn.Conv2d(self.chin,self.ch0,3,1,1,padding_mode=self.padding_mode) # no downsampling
        self.b0 = self.bn(self.ch0,**self.bn_kwargs)
        self.c0a = torch.nn.Conv2d(self.ch0,self.ch0,3,1,1,padding_mode=self.padding_mode)
        self.b0a = self.bn(self.ch0,**self.bn_kwargs)
        
        self.c1 = torch.nn.Conv2d(self.ch0,self.ch0*2,3,2,1,padding_mode=self.padding_mode)
        self.b1 = self.bn(self.ch0*2,**self.bn_kwargs)
        self.c1a = torch.nn.Conv2d(self.ch0*2,self.ch0*2,3,1,1,padding_mode=self.padding_mode)
        self.b1a = self.bn(self.ch0*2,**self.bn_kwargs)
        
        self.c2 = torch.nn.Conv2d(self.ch0*2,self.ch0*4,3,2,1,padding_mode=self.padding_mode)
        self.b2 = self.bn(self.ch0*4,**self.bn_kwargs)
        self.c2a = torch.nn.Conv2d(self.ch0*4,self.ch0*4,3,1,1,padding_mode=self.padding_mode)
        self.b2a = self.bn(self.ch0*4,**self.bn_kwargs)
        
        
        
        self.c3 = torch.nn.Conv2d(self.ch0*4,self.ch0*8,3,2,1,padding_mode=self.padding_mode)
        self.b3 = self.bn(self.ch0*8,**self.bn_kwargs)
        self.c3a = torch.nn.Conv2d(self.ch0*8,self.ch0*8,3,1,1,padding_mode=self.padding_mode)
        self.b3a = self.bn(self.ch0*8,**self.bn_kwargs)
        
        
        self.c4 = torch.nn.Conv2d(self.ch0*8,self.chout,1,1,padding_mode=self.padding_mode)
        

        
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
        self.c0  = torch.nn.Conv2d(1,M,3,1,1)
        self.c1  = torch.nn.Conv2d(M,M,3,1,1)
        self.c2  = torch.nn.Conv2d(M,M,3,1,1)
        
        
    def forward(self,x):
        #print(x.shape)
        # move the batch dim, make the size N x 1 (NO! not any more)
        nbatch = x.shape[0]
        nchannels = x.shape[1] # this is variable, but will be the same for every image in a batch
        
        
        #print(x.shape)
        # reshape B x N into (BN) x 1
        x_ = x.reshape( nbatch*nchannels, 1,x.shape[-2],x.shape[-1] )
        #print(x_.shape)
        cx = torch.relu(self.c0(x_))
        # after the first layer we now have (BN) x M
        #print(cx.shape)
        cx = torch.relu(self.c1(cx))
        # after the second layer again (BN) x M
        cx = self.c2(cx)
        #print(cx.shape)
        # now reshape it to B X N X M        
        cx = cx.reshape(nbatch,nchannels,self.M,x.shape[-2],x.shape[-1])
        #print(cx.shape)
        cx = torch.softmax(cx*100,1)
        #print(cx.shape)
        
        # now matrix multiply over the N axis
        out = torch.sum(x[:,:,None]*cx,1,keepdims=False)
        self.out = out
        return out

# class VariableInputConv2d(torch.nn.Module):
#     """The input layer to the YOLO neural network which takes an input image of size 1 x N x R x C, applies several convolutions to the image, and outputs an image of size 1 x M x R x C where N is variable and M is fixed. The assumption here is that we have a batch size of 1, so the batch dimension can be manipulated. This is accomplished via the following workflow:
    
#     (1): Move channel dimension to batch dimension, so that we now have N samples with 1 channel each
#         - 1 x N x R x C => N x 1 x R x C
#     (2) Apply some convolutions and relu operations, so that we now have an N x M array of images where M is fixed and N is variable
#         - N x 1 x R x C => N x M x R x C
#     (3) Take a softmax over all N channels and multiply
#         - N x M x R x C => 1 x M x R x C
#     (4) Return the result with a fixed number of channels
    
#     Note that this layer makes our overall network permutation invariant (i.e. if we input an RGB image, or a BGR image, the output would be exactly the same.

#     Parameters:
#     -----------
#     M : int
#         The desired number of output channels, independent of the number of input channels

#     Returns:
#     --------
#     out : torch.tensor of size 1 x M x R x C
    
#     """
#     def __init__(self,M):
#         super().__init__()
#         self.M = M
#         self.c0  = torch.nn.Conv2d(1,M,3,1,1)
#         self.c1  = torch.nn.Conv2d(M,M,3,1,1)
#         self.c2  = torch.nn.Conv2d(M,M,3,1,1)
        
        
#     def forward(self,x):
#         # move the batch dim, make the size N x 1 x R x C
#         x = x[0,:,None]
        
#         # apply the conv layer, make the size N x M x R x C
#         cx = torch.relu(self.c0(x))
#         cx = torch.relu(self.c1(cx))
#         cx = self.c2(cx)
#         cx = torch.softmax(cx*100,0)
        
#         # now matrix multiply
#         out = torch.sum(x*cx,0,keepdims=True)
#         self.out = out
#         return out
        


def bbox_to_rectangles(bbox,**kwargs):
    """
    This function converts a set of bounding boxes of the form [cx,cy,w,h] to a set of rectangular objects for visualization purposes

    Parameters:
    -----------
    bbox : A N x 4 array[numpy.float64]
        Contains the bbox info N bboxes of the form [xc,cy,w,h]

    Returns:
    --------
    out : matplotlib.collections.PolyCollection 
        Contains N rectangles to be plotted later
    
    """
    N = bbox.shape[0]
    p0 = np.stack((bbox[:,0],bbox[:,1]),-1)
    p1 = np.stack((bbox[:,0]+bbox[:,2],bbox[:,1]),-1)
    p2 = np.stack((bbox[:,0]+bbox[:,2],bbox[:,1]+bbox[:,3]),-1)
    p3 = np.stack((bbox[:,0],bbox[:,1]+bbox[:,3]),-1)
    p = np.stack((p0,p1,p2,p3),-2)
    p = p.reshape(N*4,2)
    f = np.arange(N*4)
    f = f.reshape(-1,4)
    
    return PolyCollection(p[f],**kwargs)



def convert_data(out,B,stride):
    """
    Convert the outputs from the YOLO neural network into bounding boxes for performance quantification. Note that this operation is not differentiable. It also outputs the raw data, reformmated into a list intsead of an image of grid cells. These outputs are differentiable. The last one of these outputs is the score (data[...,-1]). Note that class probabilities are NOT output by this function. 

    Parameters:
    -----------
    out : torch.Tensor
        The output object from the primary YOLO neural network
    B : int
        The number of bounding boxes per block that the model will output.  (Automatically defined when initializing the network)
    stride : int
        The number of pixels to pass over when convolving the input image in each layer (Automatically defined when initializing the network)

    Returns:
    --------
    bboxes : torch.Tensor of size N x 4
        N is the number of bounding boxes output from the network; Bounding boxes are of the form [cx,cy,scalex,scaley]. There are no gradient computations done here. These bounding boxes are for visualization or downstream analysis.
    data : torch.Tensor of size N x 5
        N is the number of bounding boxes output from the network; Bounding boxes are of the form [cx,cy,w,h,conf]. There are gradient computations done here. These are for our loss function and, potentially, other downstream analysis.
    
    """
    # Get the positions of the grid cells
    x = torch.arange(out.shape[-1])*stride + (stride-1)/2
    y = torch.arange(out.shape[-2])*stride + (stride-1)/2
    YX = torch.stack(torch.meshgrid(y,x,indexing='ij'),0)
    
    bboxes = []
    data = []
    for i in range(B):
        # Get the five numbers defining a bbox
        outB = out[:,5*i:5*(i+1)]
        
        # Get the standard bbox coordinates for drawing and computing iou; They don't go into the loss
        x = (torch.sigmoid(outB[:,0])-0.5)*stride + YX[1] # between -0.5 and 0.5, scaled
        y = (torch.sigmoid(outB[:,1])-0.5)*stride + YX[0]
        w = torch.exp(outB[:,2])*stride
        h = torch.exp(outB[:,3])*stride
        x = x - w/2
        y = y - h/2

        bboxes.append(torch.stack( (x.ravel(),y.ravel(),w.ravel(),h.ravel()) ).clone().detach())

        # now the data that I actually need for the loss
        # needs to be differentiable

        c = torch.sigmoid(outB[:,4])


        # these numbers need to be compared to the appropriate boxes
        cx = torch.sigmoid(outB[:,0])-0.5
        cy = torch.sigmoid(outB[:,1])-0.5
        ws = torch.exp(outB[:,2])
        hs = torch.exp(outB[:,3])    
        data.append(torch.stack((cx.ravel(),cy.ravel(),ws.ravel(),hs.ravel(),c.ravel(),)))


    bboxes = torch.cat(bboxes,-1).T
    data = torch.cat(data,-1).T

    return bboxes,data



def get_assignment_inds(bboxes,bbox,shape,stride,B):
    """
    Assigns each training bounding box to a specific cell, and picks the bounding box from that cell with the best iou.

    Parameters:
    -----------
    bboxes : torch.Tensor of size [N, 4]
        N is the number of bounding boxes computed for a single image and is equal to (np.shape(I)[0] / stride) * (np.shape(I)[0] / stride) * net.B
    bbox : torch.Tensor of size [X, 4]
        X is the true number of bboxes associated with image I
    shape : 4-dimensional torch.Tensor
        The shape of the data directly output from the YOLO neural network
    stride : int
        The number of pixels to pass over when convolving the input image in each layer (Automatically defined when initializing the network)
    B : int
        The number of bboxes generated by the YOLO neural network at each block

    Returns:
    --------
    assignment_inds : numpy array of size X
        The index of the bounding box from the 'bboxes' parameter cooresponding to the ith bounding box from the 'bbox' parameter
    ious : numpy array of size X
        The iou between the ith bounding box from the 'bbox' parameter, and the corresponding bbox from the 'bboxes' parameter

    """
    # this should be the shape of the outputs
    x = np.arange(shape[-1])*stride + (stride-1)/2
    y = np.arange(shape[-2])*stride + (stride-1)/2
    YX = np.stack(np.meshgrid(y,x,indexing='ij'),0).reshape(2,-1)

    cx_bbox = bbox[...,0] + bbox[...,2]/2
    cy_bbox = bbox[...,1] + bbox[...,3]/2

    # each bbox should lie in exactly one cell
    d2 = (cx_bbox[:,None] - YX[1,None,:])**2 + (cy_bbox[:,None] - YX[0,None,:])**2
    assignment_inds = np.argmin(d2,-1)        
    # the assignment ind will be used for classification


    # after I figure out the cell I will calculate IOU
    # and pick the best
    # note the diag here is not a good idea, I need to avoid pairwise and just do it pointwise
    ious = []
    for i in range(B):
        #ious.append( np.diag(iou(bboxes[assignment_inds+(bboxes.shape[0]//B)*i].numpy(),bbox) ))    
        # print()
        ious.append( iou(bboxes[assignment_inds+(bboxes.shape[0]//B)*i].numpy(),bbox,nopairwise=True) )
    ious = np.stack(ious,0)    
    B_inds = np.argmax(ious,0)    
    assignment_inds = assignment_inds + ((B_inds*bboxes.shape[0]//B)*i)

    ious = np.take_along_axis(ious,B_inds[None],axis=0)[0]
    
    return assignment_inds, ious


def get_best_bounding_box_per_cell(bboxes,scores,B):
    """
    Get the best bounding box for each cell output from the YOLO neural network. 

    Parameters:
    -----------
    bboxes : torch.Tensor of size [N, 4]
        N is the number of bounding boxes computed for a single image and is equal to (np.shape(I)[0] / stride) * (np.shape(I)[0] / stride) * net.B
    scores : torch.Tensor of size [N,1]
        The confidence for each corresponding box in the 'bboxes' parameter
    B : int
        Default - 2; The number of bboxes generated by the YOLO neural network at each cell

    Returns:
    --------
    bboxes_out : torch.Tensor of size [N/B, 4]
        A list of the best bounding boxes for each cell
    scores_out : torch.Tensor of size [N/B, 1]
        A list of the confidence for the corresponding bounding boxes in 'bboxes_out'
        
    """

    scores_out = scores.reshape(B,-1)
    inds = torch.argmax(scores_out,0)
    scores_out = torch.take_along_dim(scores_out,inds[None,:],0).squeeze()

    bboxes_out = bboxes.reshape(B,-1,4)
    bboxes_out = torch.take_along_dim(bboxes_out,inds[None,:,None],0).squeeze()

    return bboxes_out, scores_out


def get_reg_targets(assignment_inds,bbox,B,shape,stride):
    """What are the true bounding box parameters we want to predict.

    Parameters:
    -----------
    assignment_inds : numpy array of size X
        The idx at position i corresponds to the bounding box output from the YOLO framework which most accurately encompasses bounding box 'i' in the 'bbox' (ground truth labels) parameter
    bbox : torch.Tensor of size [X, 4]
        X is the true number of bboxes associated with image I
    B : int
        The number of bboxes generated by the YOLO neural network at each block
    shape : 4-dimensional torch.Tensor
        The shape of the data directly output from the YOLO neural network
    stride : int
        The number of pixels to pass over when convolving the input image in each layer (Automatically defined when initializing the network)

    Returns:
    --------
    shiftx : numpy array of shape [X,]
        Defines the shift in x needed to align bbox[i] with the corresponding best estimated bounding box
    shifty : numpy array of shape [X,]
        Defines the shift in y needed to align bbox[i] with the corresponding best estimated bounding box
    scalex : numpy array of shape [X,]
        Defines the scale in x needed to align bbox[i] with the corresponding best estimated bounding box
    scaley : numpy array of shape [X,]
        Defines the scale in y needed to align bbox[i] with the corresponding best estimated bounding box
    
    """
    # get the yx positions of each bbox
    x = np.arange(shape[-1])*stride + (stride-1)/2
    y = np.arange(shape[-2])*stride + (stride-1)/2
    YX = np.stack(np.meshgrid(y,x,indexing='ij'),0).reshape(2,-1)


    cx_assigned = YX[1].ravel()[assignment_inds%(shape[-1]*shape[-2])]
    cy_assigned = YX[0].ravel()[assignment_inds%(shape[-1]*shape[-2])]

    cx_true = bbox[:,0] + bbox[:,2]/2
    cy_true = bbox[:,1] + bbox[:,3]/2

    shiftx = (cx_true - cx_assigned)/stride
    shifty = (cy_true - cy_assigned)/stride

    scalex =(bbox[:,2]/stride)
    scaley = (bbox[:,3]/stride)
    
    return shiftx,shifty,scalex,scaley



def imshow(I,ax,**kwargs):
    """
    This function will normalize the image I and plot it on the axes object ax

    Parameters:
    -----------
    I : 1 x N x N array[numpy.float64]
        An image
    ax : matplotlib.axes._axes.Axe object
        An empty axis onto which I will be plotted


    Returns:
    --------
    None
        
    """
    I = np.array(I)
    I = I - np.min(I,axis=(1,2),keepdims=True)
    I = I / np.max(I,axis=(1,2),keepdims=True)
    if I.shape[0] == 1:
        ax.imshow(I[0],**kwargs)
    elif I.shape[0] == 2:
        ax.imshow(np.stack((I[0],I[1],I[0]*0),-1),**kwargs)
    elif I.shape[0] >= 3:
        ax.imshow(np.stack((I[0],I[1],I[2]),-1),**kwargs)


def iou(bbox0,bbox1,nopairwise=False):
    """
    Calculate pairwise iou between a set of estimated bounding boxes (bbox0) and the set of ground truth bounding boxes (bbox1).

    Parameters:
    -----------
    bbox0 : torch.Tensor of size [N, 4]
        N is the number of bounding boxes
    bbox1 : torch.Tensor of size [N, 4]
        N is the number of bounding boxes
    nopairwise : bool
        If True, compute pointwise, not pairwise IOU

    Returns:
    --------
    out : Array of length N
        An array containing the pairwise (or pointwise) IOU values between the elements bbox0 and bbox 1
    
    """
    if bbox0.ndim == 1: bbox0 = bbox0[None]
    if bbox1.ndim == 1: bbox1 = bbox1[None]
    if not nopairwise:
        bbox0 = bbox0[:,None]
        bbox1 = bbox1[None,:]
        
    # Get the coordinates for 2 opposite corners from all bboxes
    l0 = bbox0[...,0]
    l1 = bbox1[...,0]
    r0 = bbox0[...,0] + bbox0[...,2]
    r1 = bbox1[...,0] + bbox1[...,2]
    
    t0 = bbox0[...,1]
    t1 = bbox1[...,1]
    b0 = bbox0[...,1] + bbox0[...,3]
    b1 = bbox1[...,1] + bbox1[...,3]
    
    
    # Compute the coordinates for the intersection of all pairs of bounding boxes
    l = np.maximum(l0*np.ones_like(l1),l1*np.ones_like(l0))
    r = np.minimum(r0*np.ones_like(r1),r1*np.ones_like(r0))
    t = np.maximum(t0*np.ones_like(t1),t1*np.ones_like(t0))
    b = np.minimum(b0*np.ones_like(b1),b1*np.ones_like(b0))

    # Compute the area within bbox0 (vol0), within bbox1 (vol1), and their shared area
    vol0 = (r0-l0)*(b0-t0)    
    vol1 = (r1-l1)*(b1-t1)    
    vol = (r-l)*(b-t) * (r>l) * (b>t) # could be 0
    
    # Use de morgans law to get volume of union
    VOL = vol0 + vol1 - vol
    
    return vol/VOL



def train_yolo_model(nepochs, lr, cls_loss, outdir, modelname, optimizername, lossname, verbose = False, resume = False, J_path=None):
    """
    Train a neural network defined by the YOLO framework and other provided hyperparameters using the simulated dataset 'groundtruth'.

    Parameters:
    -----------
    nepochs : int
        The number of epochs used to train the model
    lr : float
        The learning rate used to train the model
    outdir : str
        The output directory where all files will be saved during training, or where all files were saved if the model has already been trained.
    modelname : str
        The file name of the model used during training
    optimizername : str
        The file name of the optimizer used during training
    lossname : str
        The file name of the losses computed during training
    resume : bool
        Default - False; If True, resume the training of model 'outdir/modelname' or load the pretrained model saved at 'outdir/modelname'
    J_path : str
        Default - None; The file path to the image to be used during the validation portion of training
    
    Returns:
    --------
    net : torch.nn.Module
        A neural network which has been trained on a simulated dataset
    
    """

    # Check to see that outdir exists and create outdir if it does not exist
    os.makedirs(outdir,exist_ok=True)

    net = Net()
    groundtruth = GroundTruthDataset()
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    nclasses = 3

    # Load the target image to be used during the evaluation step of training
    if J_path is not None:
        J = plt.imread(J_path)
        J= J[...,:3]
        if J.dtype == np.uint8:
            J = J / 255.0
        J = J.transpose((-1,0,1))

    Esave = []    
    fig,ax = plt.subplots(2,3,figsize=(9,6)) 
    ax = ax.ravel()
    fig1,ax1 = plt.subplots(3,3,figsize=(9,9))
    fig1.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.1,wspace=0.1)

    if resume:        
        net.load_state_dict(torch.load(os.path.join(outdir,modelname)))
        optimizer.load_state_dict(torch.load(os.path.join(outdir,optimizername)))
        Esave = torch.load(os.path.join(outdir,lossname))[0]
        print(f'Loaded predefined model from {outdir}/{modelname}')
    for e in range(nepochs):
        if verbose:
            print(f'Starting epoch {e}')
            start = time.time()

        if resume and e < len(Esave):
            continue
        count = 0
        Esave_ = []    
        for I,bbox,cl in groundtruth:
            optimizer.zero_grad()
            # run through the net
            out = net(torch.tensor(I[None],dtype=torch.float32))
            
            # convert the data into bbox format
            bboxes,data = convert_data(out,net.B,net.stride)
            
            # get assignments
            assignment_inds,ious = get_assignment_inds(bboxes,bbox,out.shape,net.stride,net.B)
            unassigned_inds = np.array([a for a in range(bboxes.shape[0]) if a not in assignment_inds])
            
            
            # get target parameters
            shiftx,shifty,scalex,scaley = get_reg_targets(assignment_inds,bbox,net.B,out.shape,net.stride)    
    
            
            # now build the loss function
            data_assigned = data[assignment_inds]
            targets = np.stack((shiftx,shifty,scalex,scaley,ious),-1)
            
            # this is the mean square error for assigned
            # they used a weight of 0.5 for noobj
            # note that in the paper they used a different loss for the scales (take the square root first)
            Ecoord = torch.sum((data_assigned[:,:4]-torch.tensor(targets[:,:4]))**2)*5
            # if it is assigned ot an object we want to predict the iou
            Eobj = torch.sum((data_assigned[:,-1]-torch.tensor(targets[:,-1]))**2)
            # if there is no object assigned we want to predict 0
            Enoobj = torch.sum((data[unassigned_inds,-1]-0)**2)*0.5
            # and we want to classify
            classprobs = out[:,-nclasses:].reshape(nclasses,-1)
            classprobs_assigned = classprobs[...,assignment_inds%(out.shape[-1]*out.shape[-2])]
            # note the paper uses mean square error on the probability vector
            # here I use cross entropy
            Ec = cls_loss(classprobs_assigned[None],torch.tensor(cl)[None])
            
            E = Ecoord + Eobj + Enoobj + Ec
            Esave_.append(E.item())
            E.backward()
            optimizer.step()
            count += 1
            if count >= len(groundtruth): break
        
        # draw        
        Esave.append(np.mean(Esave_))
        ax[0].cla()
        ax[0].plot(Esave,label='loss')
        ax[0].legend()
        ax[0].set_yscale('log')
        ax[0].set_title('training loss')
        
        
        ax[1].cla()
        imshow(net.color.out.clone().detach()[0],ax[1])
        ax[1].add_collection(bbox_to_rectangles(bbox,fc='none',ec=[0.5,0.5,0.5,0.5]))
        
        # get better colors
        p = torch.softmax(classprobs.clone().detach(),0)
        c0 = torch.tensor([1.0,0.0,0.0])
        c1 = torch.tensor([0.0,1.0,0.0])
        colors = ( (p[0]*c0[...,None]) + (p[1]*c1[...,None])  ).T.numpy()    
        if nclasses == 3:
            c2 = torch.tensor([0.0,0.0,1.0])
            colors = ( (p[0]*c0[...,None]) + (p[1]*c1[...,None]) + (p[2]*c2[...,None]) ).T.numpy()    

        bboxes_,scores_ = get_best_bounding_box_per_cell(bboxes,data[:,-1].clone().detach(),net.B)
        ax[1].add_collection(bbox_to_rectangles(bboxes_,fc='none',ec=colors,ls='-',alpha=scores_))
        ax[1].set_title('Annotations')
        
        classprob = torch.softmax(out.clone().detach()[0,-nclasses:],0)
        mask = torch.sigmoid(out[0,4].clone().detach())
        ax[2].cla()
        ax[2].imshow(mask,vmin=0,vmax=1,interpolation='gaussian')
        ax[2].set_title('Predicted score (pobj*iou)')

        ax[3].cla()    
        ax[3].imshow(   classprob[0]*mask  ,vmin=0,vmax=1,interpolation='gaussian')
        ax[3].set_title('Class 0 (smooth) prob map')
        
        ax[4].cla()
        ax[4].imshow(   classprob[1]*mask   ,vmin=0,vmax=1,interpolation='gaussian')
        ax[4].set_title('Class 1 (sharp) prob map')
        
        if nclasses == 3:
            ax[5].cla()
            ax[5].imshow(   classprob[2]*mask   ,vmin=0,vmax=1,interpolation='gaussian')
            ax[5].set_title('Class 2 (bumpy) prob map')
        fig.canvas.draw()
        
        
        if J_path is not None:
            with torch.no_grad():
                net.eval()
                for r in range(3):
                    for c in range(3):
                        ax1[r,c].cla()
                        sl = (slice(r*2000+2000,r*2000+2000+256),slice(c*2000+2000,c*2000+2000+256)) # Note: '2000' may need to be changed for different images. 
                        out = net(torch.tensor(J[(slice(None),)+sl][None],dtype=torch.float32))
        
                        # convert the data into bbox format
                        bboxes,data = convert_data(out,net.B,net.stride)
                        imshow(net.color.out.clone().detach()[0],ax1[r,c])
        
                        classprobs = out[:,-nclasses:].reshape(nclasses,-1)
                        p = torch.softmax(classprobs.clone().detach(),0)
                        c0 = torch.tensor([1.0,0.0,0.0])
                        c1 = torch.tensor([0.0,1.0,0.0])
                        colors = 'r'
                        bboxes_,scores_ = get_best_bounding_box_per_cell(bboxes,data[:,-1].clone().detach(),net.B)
                        alpha = scores_.clone().detach()
                        alpha = alpha * (alpha>0.5)
                        
                        ax1[r,c].add_collection(bbox_to_rectangles(bboxes_,fc='none',ec=colors,ls='-',alpha=alpha))
                        ax1[r,c].axis('off')
                net.train()
            fig1.canvas.draw()
            if not e%10:
                fig1.savefig(os.path.join(outdir,f'example_e_{e:06d}.png'))

        if verbose:
            print(f'Finished epoch {e} in {time.time() - start:.2f}s')        
        
        # save data
        torch.save(net.state_dict(),os.path.join(outdir,modelname))
        torch.save(optimizer.state_dict(),os.path.join(outdir,optimizername))
        torch.save([Esave],os.path.join(outdir,lossname))  

    return net