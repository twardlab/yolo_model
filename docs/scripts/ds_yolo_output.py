import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def downsample_yolo_output(fpath, n, ds_factor = 2, outdir = None):
    """
    Downsample the output from our custom YOLO pipeline. 
    The tensor stored in the input file (.npy) will be downsampled 'n' times by a factor of ds_factor
    
    Parameters
    ==========
    fpath : str
        The location of the file to be downsampled; Must be one of the .npy files output from the YOLO pipeline
    n : int
        The number of times to downsample the image by 'ds_factor'; For n=0, the bbox coordinate values will be replaced by their area.
    ds_factor : int
        Default - 2; The downsampling factor for 1 iteration of downsampling
    outdir : str
        Default - None; If None, don't save the output. Else, the directory where the output will be saved

    Returns
    =======
    out : ndarray of float32
        The downsampled version of the original input file        
    """

    # Load the .npy file and extract relevant subsets of pixels
    # Note: 8 numbers at each pixel - first four are bounding box (left top right bottom, i.e. xy of top left, xy of bottom right), confidence, 3 class prob
    I = np.load(fpath)

    # Compute the area of a bbox at every pixel
    A = (I[...,2]-I[...,0])*(I[...,3]-I[...,1])
    
    # Extract the confidence at every pixel
    P = I[...,4]
    
    # Extract the class probabilities at every pixel (then apply softmax)
    F = I[...,5:]
    F = np.exp(-F)/np.sum(np.exp(-F),-1,keepdims=True) 

    if n == 0:
        return ds_outputs_to_orig_format(A, P, F)

    # ==================================
    # ===== Begin the downsampling =====
    # ==================================

    # Recursively downsample the input array when n > 1
    n_remaining = n
    while n_remaining > 0:
        
        # For downsampling bbox coords, take a weighted sum of the bbox area with weights P
        nd = np.array(P.shape)//ds_factor
        AP = A*P
        Ad = np.zeros(nd)
        for i in range(ds_factor):
            for j in range(ds_factor):
                Ad += AP[i:nd[0]*ds_factor:ds_factor,j:nd[1]*ds_factor:ds_factor]
    
        # For downsampling confidence, take a sum of probability, which becomes count
        Pd = np.zeros(nd)
        for i in range(ds_factor):
            for j in range(ds_factor):
                Pd += P[i:nd[0]*ds_factor:ds_factor,j:nd[1]*ds_factor:ds_factor]
    
        # For downsampling class probabilities / features, take a weighted average with weights P
        FP = F*P[...,None]
        num_features = 3
        Fd = np.zeros(tuple(nd)+(num_features,))
        for i in range(ds_factor):
            for j in range(ds_factor):
                Fd += FP[i:nd[0]*ds_factor:ds_factor,j:nd[1]*ds_factor:ds_factor]
        Fd = Fd/Pd[...,None]

        A = Ad
        P = Pd
        F = Fd
        n_remaining -= 1

    out_arr = ds_outputs_to_orig_format(Ad, Pd, Fd)
    if outdir != None:    
        out_fname = os.path.splitext(os.path.split(fpath)[-1])[0] + f'_ds{n}.npy'
        np.save(os.path.join(outdir, out_fname), out_arr)

    return out_arr

def plot_yolo_outputs(out, n):
    """
    Generate a set of 4 plots visualizing the downsampled output.
    
    Parameters
    ==========
    out : ndarray of shape (R, C, 5)
        The downsampled YOLO output
    n : int
        The number of times 'out' was downsampled from the original 
        
    Returns
    =======
    fig : matplotlib.pyplot.figure
        The figure obhect visualizing the 4 subplots
    ax : matplotlib.pyplot.axes
        The axes object containing the 4 subplots   
    """        
    # Initialize data structures
    Ad, Pd, Fd = orig_format_to_ds_outputs(out)
    nrow = 1
    ncol = 4
    fig,ax = plt.subplots(nrow, ncol, layout='tight')

    # Plot area
    ax0_plot = ax[0].imshow(Ad)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax0_plot, cax=cax, orientation='vertical')
    ax[0].set_title('Area ($\mu m^2$)')
    ax[0].set_xticks([0,Ad.shape[1]-1])
    ax[0].set_yticks([0,Ad.shape[0]-1])

    # Plot confidence
    ax1_plot = ax[1].imshow(Pd)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax1_plot, cax=cax, orientation='vertical')
    ax[1].set_title('Confidence (P:[0,1])')
    ax[1].set_xticks([0,Pd.shape[1]-1])
    ax[1].set_yticks([])

    # Plot features
    ax2_plot = ax[2].imshow(Fd)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax2_plot, cax=cax, orientation='vertical')
    ax[2].set_title('Features (P:[0,1]) for each rgb')
    ax[2].set_xticks([0,Fd.shape[1]-1])
    ax[2].set_yticks([])

    # Plot area using confidence for the alpha value    
    ax3_plot = ax[3].imshow(Ad, alpha=Pd/(4**n))
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax3_plot, cax=cax, orientation='vertical')
    ax[3].set_title(f'Area(alpha=P/$4^n$)')
    ax[3].set_xticks([0,Ad.shape[1]-1])
    ax[3].set_yticks([])
    
    fig.set_size_inches((ncol*3,nrow*4))
    fig.suptitle(f'{n} rounds of downsampling')

    return fig, ax

def ds_outputs_to_orig_format(Ad, Pd, Fd):
    """
    A helper function for stacking the downsampled arrays back into the format to be saved as an .npy file
    """
    out = np.stack((Ad, Pd, Fd[:,:,0], Fd[:,:,1], Fd[:,:,2]), axis=-1)
    return out

def orig_format_to_ds_outputs(out):
    """
    A helper function for breaking down the stacked arrays back into the intermediate outputs from downsampling - Ad, Pd, and Fd
    """
    Ad = out[:,:,0]
    Pd = out[:,:,1]
    Fd = out[:,:,2:]
    return Ad, Pd, Fd