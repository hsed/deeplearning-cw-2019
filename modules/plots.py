import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, LogLocator
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from sklearn.metrics import confusion_matrix

import seaborn as sns

import argparse
from argparse import RawTextHelpFormatter
from os import path




def plotDualModelLearningCurves(xyLst, xzLst,
                             xTicks=None,
                             xLim=(None,None), yLim=(None,None), zLim=(None,None),
                             title='', figSize=(10, 4), xStrFmt='%0.1f',
                             tightLayout=True, showMajorGridLines=True, 
                             showMinorGridLines=False, xLbl='xAxis ->', 
                             yLbl='yAxis ->', zLbl='zAxis ->',
                             xLogBase = None, yLogBase=None, zLogBase=None,
                             tight_pad=2.0, tight_rect=[0,0,1,1], savefig=False,
                             legendsYLst = ['TrainY1', 'TrainY2'],
                             legendsZLst = ['TrainZ1', 'TrainZ2'],
                             yStrFmt='%0.1f', zStrFmt='%0.1f',
                             figfilename='plots/lcplot.pdf'):
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figSize)
    lw = 2  # line width

    ## datLst is all the y-plots i.e.
    ## 2D list 

    # plot both y and z
    for (ax, pltDatLst, legendLst, lbl, strFmt, lim) in \
        zip([ax1, ax2], [xyLst, xzLst], 
            [legendsYLst, legendsZLst],
            [yLbl, zLbl],
            [yStrFmt, zStrFmt],
            [yLim, zLim]):
        
        for (pltDat, legend) in zip(pltDatLst, legendLst):
            ax.plot(pltDat[:,0], pltDat[:,1], label=legend)
        

        ax.legend()        
        ax.set_ylabel(lbl)
        ax.set_xlabel(xLbl)

        if xLogBase is not None:
            ax.set_xscale('log', basex=xLogBase)
            ax.xaxis.set_major_locator(LogLocator(base=xLogBase, numticks=xTicks))
        else:
            # mutuall exclusive
            if xTicks is not None:
                # must supply xTicks to do this
                ax.xaxis.set_major_locator(LinearLocator(numticks=xTicks))
            pass

        ax.xaxis.set_major_formatter(FormatStrFormatter(xStrFmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(strFmt))

        ax.set_xlim(xLim)
        ax.set_ylim(lim)

        if showMajorGridLines and showMinorGridLines:
            ax.grid(which='both')
        elif showMajorGridLines:
            ax.grid(which='major')
        elif showMinorGridLines:
            ax.grid(which='minor')
        



    plt.suptitle(title)

    if tightLayout: 
        #(left, bottom, right, top),
        #(0,0,1,1)
        plt.tight_layout(rect=tight_rect, pad=tight_pad)
    
    if savefig:
        #fig.tight_layout()
        fig.savefig('plots/%s' % figfilename,
                    format='pdf',
                    dpi=300,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0.01)
    
    plt.show()




def plot_denoise_mae_dssim(denoise_baseline_model, denoise_improved_model,
                           denoise_baseline_title='Denoised Baseline', denoise_improved_title='Denoised Improved',
                           show_plt=False, save_fig=True, figdir='plots', figfilename='denoise_compare.pdf'):
    """Plots a noisy patch, denoised patch and clean patch.
    Args:
        denoise_model: keras model to predict clean patch
    """
    from read_data import tps, hpatches_sequence_folder, DenoiseHPatches
    from utils import gallery
    import matplotlib.pyplot as plt
    dataset = DenoiseHPatches(['./hpatches/v_talent'], batch_size=16, cache_overwrite=True, toy_data=True)
    #print("size: ", len(generator))
    imgs, imgs_clean = dataset[999] # get a random batch where imgs and imgs clean are DIFFERENT
    #index = np.random.randint(0, imgs.shape[0])
    imgs_den = denoise_baseline_model.predict(imgs)
    imgs_den2 = denoise_improved_model.predict(imgs)
    #print("REAL_SHAPE:", imgs.shape, "PRED_SHAPE: ", imgs_den.shape)

    fig = plt.figure(figsize=(12,3))
    plt.subplot(141)
    plt.imshow(gallery(imgs, ncols=4)[:,:,0], cmap='gray') 
    plt.title('Noisy', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.subplot(142)
    plt.imshow(gallery(imgs_den, ncols=4)[:,:,0], cmap='gray') 
    plt.title(denoise_baseline_title, fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.subplot(143)
    plt.imshow(gallery(imgs_den2, ncols=4)[:,:,0], cmap='gray') 
    plt.title(denoise_improved_title, fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.subplot(144)
    plt.imshow(gallery(imgs_clean, ncols=4)[:,:,0], cmap='gray')
    plt.title('Clean', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.tight_layout()

    
    if save_fig:
        fig = plt.gcf()
        fig.savefig(os.path.join(figdir, figfilename),
                    format='pdf',
                    dpi=300,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0.01)
    

    if show_plt:
        plt.show()




def plot_denoise_mae_dssim_mix(show_plt=False, save_fig=True, figdir='plots', figfilename='denoise_mix_loss.pdf'):
    """Plots a mix of patches for several alpha value.
       Functions hardcoded
    Args:
        denoise_model: keras model to predict clean patch
    """
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    from read_data import tps, hpatches_sequence_folder, DenoiseHPatches
    from utils import gallery
    import matplotlib.pyplot as plt
    from keras.models import load_model
    dataset = DenoiseHPatches(['./hpatches/v_talent'], batch_size=16, cache_overwrite=True, toy_data=True)
    #print("size: ", len(generator))
    imgs, imgs_clean = dataset[999] # get a random batch where imgs and imgs clean are DIFFERENT
    #index = np.random.randint(0, imgs.shape[0])
    #  from plots import plot_denoise_mae_dssim_mix; plot_denoise_mae_dssim_mix(); exit();

    model_titles = [
        'MAE (Baseline)\n(0.449, 0.057, 0.269)',
        'DSSIM+MAE α=0.005\n(0.473, 0.056, 0.252)',
        'α=0.01\n(0.449, 0.057, 0.269)',
        'α=0.1\n(0.46369, 0.0581, 0.199)',
        'α=0.5\n(0.726, 0.106, 0.368)',
        'α=0.9\n(0.517, 0.0506, 0.256)',
        'α=0.999\n(0.491, 0.0502, 0.248)',
        'DSSIM\n(0.484, 0.056, 0.247)',
    ]
    model_files = [
        '20190321_2128',
        '20190320_2349',
        '20190320_2210',
        '20190320_2215',
        '20190320_2220',
        '20190320_2237',
        '20190320_2242',
        '20190320_2049',

    ]

    fig = plt.figure(figsize=(12,6))

    for i, (mtitle, mfile) in enumerate(zip(model_titles, model_files)):
        model = load_model('cache/training/%s/denoiser_best.hdf5' % mfile,
                            custom_objects={'<lambda>': 'mean_absolute_error',
                                                      'DSSIMObjective': 'mean_absolute_error'})
        imgs_den = model.predict(imgs)
        del model

        plt.subplot(241+i)
        plt.imshow(gallery(imgs_den, ncols=4)[:,:,0], cmap='gray') 
        if i == 4:
            plt.title(mtitle, fontsize=8, weight = 'bold')
        else:
            plt.title(mtitle, fontsize=8)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])


    plt.tight_layout()

    if save_fig:
        fig = plt.gcf()
        fig.savefig(os.path.join(figdir, figfilename),
                    format='pdf',
                    dpi=300,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0.01)
    
    if show_plt:
        plt.show()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform plotting given plot_tag.',
                                 formatter_class=RawTextHelpFormatter)
    parser.add_argument('-dnt', '--denoiser_tags', metavar='TAG', type=str, nargs='+',
                        help='denoiser experiment shortnames', required=True)
    parser.add_argument('-dst', '--descriptor_tags', metavar='TAG', type=str, nargs='+',
                        help='descriptor experiment shortnames', required=False)
    parser.add_argument('-dnf', '--denoiser_files', help="Denoiser filenames", type=str,
                        metavar='FNAME', nargs='+', default=True)
    parser.add_argument('-dsf', '--descriptor_files', help="Descriptor filenames", type=str,
                        metavar='FNAME', nargs='+', default=True)
    
    args = parser.parse_args()
    
    plt_dual_val_loss = False
    plt_train_val_loss = True

    if plt_dual_val_loss:
        yFNames = args.denoiser_files
        zFNames = args.descriptor_files

        legendsYLst = args.denoiser_tags
        legendsZLst = args.descriptor_tags


        assert (len(yFNames) == len(legendsYLst))
        assert (len(zFNames) == len(legendsZLst))  

        ## denoiser files; comma delim for csv
        ## epoch id is the first col

        ## val_loss is the last columnid2
        xyLst2D = [
            np.loadtxt(filename, delimiter=',',
                    usecols=(0,2), skiprows=1) for filename in yFNames
        ]
        ## val_loss is the last columnid2
        xzLst2D = [
            np.loadtxt(filename, delimiter=',',
                    usecols=(0,2), skiprows=1) for filename in zFNames
        ]

        '''
        '''
        title = 'Learning Curves For Denoiser and Descriptor Model Experiments (Max Epochs: 20)'
        
        plotDualModelLearningCurves(xyLst=xyLst2D, xzLst=xzLst2D, 
                        xLogBase=None, yLogBase=None, zLogBase=None,
                        xStrFmt='%d', yStrFmt='%0.2f', zStrFmt='%0.2f',
                        xLim=(0,19),# xTicks=10,
                        xLbl='Epoch ID', yLbl='Denoiser Val Loss (MAE)',
                        zLbl='Descriptor Val Loss (MAE)',
                        title=title,
                        legendsYLst=legendsYLst, legendsZLst=legendsZLst,
                        tight_pad=0.5, tight_rect=[0.0, 0.0, 1.0, 0.95], tightLayout=True,
                        savefig=True, figfilename='lrcurves_dual.pdf')
    
    if plt_train_val_loss:
        yFNames = args.denoiser_files
        #zFNames = args.descriptor_files

        legendsYLst = args.denoiser_tags
        #legendsZLst = args.descriptor_tags


        assert (len(yFNames) == len(legendsYLst))
        #assert (len(zFNames) == len(legendsZLst))  

        ## denoiser files; comma delim for csv
        ## epoch id is the first col

        ## val_loss is the last columnid2
        xyLst2D = [
            np.loadtxt(filename, delimiter=',',
                    usecols=(0,1), skiprows=1) for filename in yFNames
        ]
        ## val_loss is the last columnid2
        xzLst2D = [
            np.loadtxt(filename, delimiter=',',
                    usecols=(0,2), skiprows=1) for filename in yFNames
        ]

        '''
        '''
        title = 'Descriptor Network: No Triplet Resampling vs Per Epoch Resampling'
        
        plotDualModelLearningCurves(xyLst=xyLst2D, xzLst=xzLst2D, 
                        xLogBase=None, yLogBase=None, zLogBase=None,
                        xStrFmt='%d', yStrFmt='%0.2f', zStrFmt='%0.2f',
                        xLim=(0,9),# xTicks=10,
                        xLbl='Epoch ID', yLbl='Descriptor Train Loss',
                        zLbl='Descriptor Val Loss',
                        title=title,
                        legendsYLst=legendsYLst, legendsZLst=legendsYLst,
                        tight_pad=0.5, tight_rect=[0.0, 0.0, 1.0, 0.95], tightLayout=True,
                        savefig=True, figfilename='lrcurves_last.pdf')