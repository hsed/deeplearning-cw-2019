from read_data import tps, hpatches_sequence_folder, DenoiseHPatches
import csv
import cv2
import numpy as np
import os
from tqdm import tqdm
import io
import PIL.Image as Image

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def plot_triplet(generator):
    import matplotlib.pyplot as plt
    a = next(iter(generator))
    index = np.random.randint(0, a[0]['a'].shape[0])
    plt.subplot(131)
    plt.imshow(a[0]['a'][index,:,:,0], cmap='gray') 
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title('Anchor', fontsize=20)
    plt.subplot(132)
    plt.imshow(a[0]['p'][index,:,:,0], cmap='gray') 
    plt.title('Positive', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(133)
    plt.imshow(a[0]['n'][index,:,:,0], cmap='gray') 
    plt.title('Negative', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.show()

def plot_denoise(denoise_model, show_plt=False, save_fig=True, fig_dir='plots'):
    """Plots a noisy patch, denoised patch and clean patch.
    Args:
        denoise_model: keras model to predict clean patch
    """
    import matplotlib.pyplot as plt
    generator = DenoiseHPatches(['./hpatches/v_there'])
    imgs, imgs_clean = next(iter(generator))
    index = np.random.randint(0, imgs.shape[0])
    imgs_den = denoise_model.predict(imgs)
    plt.subplot(131)
    plt.imshow(imgs[index,:,:,0], cmap='gray') 
    plt.title('Noisy', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(132)
    plt.imshow(imgs_den[index,:,:,0], cmap='gray') 
    plt.title('Denoised', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(133)
    plt.imshow(imgs_clean[index,:,:,0], cmap='gray')
    plt.title('Clean', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    
    if save_fig:
        plt.save_fig()
    if show_plt:
        plt.show()


def plot_denoise_v2(denoise_model, show_plt=False, save_fig=True, figdir='plots', figfilename='denoise_plot.png',
                    return_buf=False):
    """Plots a noisy patch, denoised patch and clean patch.
    Args:
        denoise_model: keras model to predict clean patch
    """
    import matplotlib.pyplot as plt
    dataset = DenoiseHPatches(['./hpatches/v_talent'], batch_size=16, cache_overwrite=True, toy_data=True)
    #print("size: ", len(generator))
    imgs, imgs_clean = dataset[999] # get a random batch where imgs and imgs clean are DIFFERENT
    #index = np.random.randint(0, imgs.shape[0])
    imgs_den = denoise_model.predict(imgs)
    #print("REAL_SHAPE:", imgs.shape, "PRED_SHAPE: ", imgs_den.shape)

    fig = plt.figure(figsize=(9,3))
    plt.subplot(131)
    
    plt.imshow(gallery(imgs, ncols=4)[:,:,0], cmap='gray') 
    plt.title('Noisy', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(132)
    plt.imshow(gallery(imgs_den, ncols=4)[:,:,0], cmap='gray') 
    plt.title('Denoised', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(133)
    plt.imshow(gallery(imgs_clean, ncols=4)[:,:,0], cmap='gray')
    plt.title('Clean', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.tight_layout()

    if return_buf:
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf,
            format='png',
            dpi=150, # lower res don't need too highres pic
            transparent=True,
            bbox_inches='tight',
            pad_inches=0.01)
        buf.seek(0)

        ## return h,w,c as tuple as well
        img = Image.open(buf)
        w, h = img.size
        c = len(img.mode)
        #img.close() do not close as need to read from buf again!
        buf.seek(0) # ensure buf at start
        
        return buf, (h, w, c)
    
    if save_fig:
        fig = plt.gcf()
        fig.savefig(os.path.join(figdir, figfilename),
                    format='png',
                    dpi=300,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0.01)
    

    if show_plt:
        plt.show()

def generate_desc_csv(descriptor_model, seqs_test, denoise_model=None, use_clean=False, curr_desc_name='custom'):
    """Plots a noisy patch, denoised patch and clean patch.
    Args:
        descriptor_model: keras model used to generate descriptor
        denoise_model: keras model used to predict clean patch. If None,
                       will pass noisy patch directly to the descriptor model
        seqs_test: CSVs will be generated for sequences in seq_test
    """
    w = 32
    bs = 128
    output_dir = './out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if use_clean:
        noisy_patches = 0
        denoise_model = None
    else:
        noisy_patches = 1
    for seq_path in tqdm(seqs_test):
        seq = hpatches_sequence_folder(seq_path, noise=noisy_patches)

        path = os.path.join(output_dir, os.path.join(curr_desc_name, seq.name))
        if not os.path.exists(path):
            os.makedirs(path)
        for tp in tps:
            n_patches = 0
            for i, patch in enumerate(getattr(seq, tp)):
                n_patches += 1

            patches_for_net = np.zeros((n_patches, 32, 32, 1))
            for i, patch in enumerate(getattr(seq, tp)):
                patches_for_net[i, :, :, 0] = cv2.resize(patch[0:w, 0:w], (32,32))
            ###
            outs = []
            
            n_batches = int(n_patches / bs) + 1
            for batch_idx in range(n_batches):
                st = batch_idx * bs
                if batch_idx == n_batches - 1:
                    if (batch_idx + 1) * bs > n_patches:
                        end = n_patches
                    else:
                        end = (batch_idx + 1) * bs
                else:
                    end = (batch_idx + 1) * bs
                if st >= end:
                    continue
                data_a = patches_for_net[st: end, :, :, :].astype(np.float32)
                if denoise_model:
                    data_a = np.clip(denoise_model.predict(data_a).astype(int), 0, 255).astype(np.float32)

                # compute output
                out_a = descriptor_model.predict(x=data_a)
                outs.append(out_a.reshape(-1, 128))

            res_desc = np.concatenate(outs)
            res_desc = np.reshape(res_desc, (n_patches, -1))
            out = np.reshape(res_desc, (n_patches, -1))
            np.savetxt(os.path.join(path,tp+'.csv'), out, delimiter=';', fmt='%10.5f')   # X is an array




if __name__ == '__main__':
    from keras.models import load_model
    ### test plotting
    denoiser_model = load_model('cache/training/20190222_1517/denoiser_20_20190222_1517.hdf5')
    plot_denoise_v2(denoiser_model, show_plt=True, save_fig=True)