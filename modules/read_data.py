## Modification from original HardNet implementation in 
## https://raw.githubusercontent.com/DagnyT/hardnet/master/code/dataloaders/HPatchesDatasetCreator.py
## I need to clean it a little bit and modify some things, but it works

## TODO: EXPORT SIZES OF PATCHES
import pickle

import os
import numpy as np
import cv2
import sys
import json
import keras
from tqdm import tqdm
import glob
import random
import joblib

'''
    Each single scene has P patches which greatly
    varies form image to image but should be about 3000

    All such patches are included in a single very long ong file
    one after the other vertically

    So you have one original ref png and 5 different transformed
    png for each of easy,hard, tought

    each transform is geometric and/or illumination changes
    with varying difficulty, then u have noise version of
    these as well but those are just paired like (x, x_hat)
'''

splits = ['a', 'b', 'c', 'view', 'illum']
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']




class DenoiseHPatches(keras.utils.Sequence):
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self, seqs, batch_size = 32, cache_overwrite=False, train_mode=True, toy_data=False):
        self.all_paths = []
        self.batch_size = batch_size
        self.dim = (32, 32)
        self.n_channels = 1
        self.sequences = {}
        self.sequences_n = {}
        self.train_mode = train_mode

        if cache_overwrite or (not self._load_cache()):
            for base in tqdm(seqs, disable=(toy_data)):
                name = base.split('/')
                self.name = name[-1]
                self.base = base
                for t in self.itr:
                    #test this function visually first!!
                    #im_path_base = os.path.join(base, 'ref' + '.png') #<-- learn geometric and guassian invariance
                    
                    im_path = os.path.join(base, t + '.png')
                    img_n = cv2.imread(im_path.replace('.png', '_noise.png'), 0)
                    
                    img = cv2.imread(im_path, 0)
                    N = img.shape[0] / 32
                    seq_im = np.array(np.split(img, N),
                                    dtype=np.uint8)
                    seq_im_n = np.array(np.split(img_n, N),
                                        dtype=np.uint8)
                    for i in range(int(N)):
                        path = os.path.join(base, t, str(i) + '.png')
                        self.all_paths.append(path)
                        self.sequences[path] = seq_im[i]
                        self.sequences_n[path] = seq_im_n[i]
            if not toy_data: self._save_cache()
        if not toy_data: self.on_epoch_end()
    
    def get_full_len(self):
        return len(self.all_paths)

    def get_images(self, index):
        path = self.all_paths[index]
        img = self.sequences[path].astype(np.float32)
        img_n = self.sequences_n[path].astype(np.float32)
        #print("is_arr_eq:", np.array_equal(img, img_n))
        return img, img_n

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.all_paths) / self.batch_size))

    def __getitem__(self, index):
        img_clean = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_noise = np.empty((self.batch_size,) + self.dim + (self.n_channels,))

        for i in range(self.batch_size):
            img, img_n = self.get_images(index*self.batch_size+i)    
            img_clean[i] = np.expand_dims(img, -1)
            img_noise[i] = np.expand_dims(img_n, -1)
        
        #print("is_arr_eq:", np.array_equal(img, img_n))
        return img_noise, img_clean    

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        random.shuffle(self.all_paths)
    

    def _load_cache(self):
        ## well its not really weights, but we save clutter of folders
        fname = 'cache/dataset/denoise_%s.pickle' % ('train' if self.train_mode else 'val')
        if not os.path.isfile(fname):
            print("Denoise cache %s not found will need to create a new one" % fname)
            return False
        
        else:
            with open(fname, 'rb') as f:
                print("loading denoise from cache file %s..." % fname)
                orig_batch_size = self.batch_size
                _dict = pickle.load(f)
                self.all_paths = _dict['all_paths']
                self.sequences = _dict['sequences']
                self.sequences_n = _dict['sequences_n']
                self.batch_size = _dict['batch_size']
                self.dim = _dict['dim']
                self.n_channels = _dict['n_channels']
                print('Loaded!', "Got bSz: %d, dim: %a, n_channels: %d" % (self.batch_size, self.dim, self.n_channels))

                if self.batch_size != orig_batch_size:
                    print("Requested batch size, mismatch! Cache: %d, Requested: %d, re-creating cache..." \
                            % (self.batch_size, orig_batch_size))
                    self.batch_size = orig_batch_size # bug fix this for cache!
                    return False
                else:
                    return True
    
    def _save_cache(self):
        fname = 'cache/dataset/denoise_%s.pickle' % ('train' if self.train_mode else 'val')
        with open(fname, 'wb') as f:
            _dict = {
                'all_paths': self.all_paths,
                'sequences': self.sequences,
                'sequences_n': self.sequences_n,
                'batch_size': self.batch_size,
                'dim': self.dim,
                'n_channels': self.n_channels,

            }
            pickle.dump(_dict, f)
            print('dataset cache saved to %s' % fname)


class hpatches_sequence_folder:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self, base, noise=1):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        if noise:
            noise_path = '_noise'
        else:
            noise_path = ''
        for t in self.itr:
            im_path = os.path.join(base, t+noise_path+'.png')
            im = cv2.imread(im_path,0)
            self.N = im.shape[0]/32
            setattr(self, t, np.split(im, self.N))



def generate_triplets(labels, num_triplets, batch_size):
    def create_indices(_labels):
        inds = dict()

        ## take all labels and group them into same value lists
        for idx, ind in enumerate(_labels):
            if ind not in inds:
                inds[ind] = []
            inds[ind].append(idx)
        return inds
    triplets = []
    indices = create_indices(np.asarray(labels))
    unique_labels = np.unique(np.asarray(labels))
    n_classes = unique_labels.shape[0]
    #print("n_classes: ", n_classes)
    # add only unique indices in batch
    # i.e. a single batch will only contain unique classes
    already_idxs = set()
    
    for x in tqdm(range(num_triplets), disable=True): # disable progressbar
        if len(already_idxs) >= batch_size:
            # flush out when exceed batch size
            already_idxs = set()
        # get random unique class for anchor and positive point
        c1 = np.random.randint(0, n_classes)
        while c1 in already_idxs:
            c1 = np.random.randint(0, n_classes)
        already_idxs.add(c1)

        # get random class for negative point
        c2 = np.random.randint(0, n_classes)
        while c1 == c2:
            c2 = np.random.randint(0, n_classes)
        if len(indices[c1]) == 2:  # hack to speed up process
            # for this class, set n1, n2 as first and second element in list of elements of class
            n1, n2 = 0, 1
        else:
            # get a pair of random samples from class c1 that are different
            n1 = np.random.randint(0, len(indices[c1]))
            n2 = np.random.randint(0, len(indices[c1]))
            while n1 == n2:
                n2 = np.random.randint(0, len(indices[c1]))
        # google
        n3 = np.random.randint(0, len(indices[c2]))
        triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
    return np.array(triplets)


def generate_idx_label_pairs(labels, num_triplets, batch_size):
    def create_indices(_labels):
        inds = dict()

        ## take all labels and group them into same value lists
        for idx, ind in enumerate(_labels):
            if ind not in inds:
                inds[ind] = []
            inds[ind].append(idx)
        return inds
    triplets = []
    indices = create_indices(np.asarray(labels))
    unique_labels = np.unique(np.asarray(labels))
    n_classes = unique_labels.shape[0]
    #print("n_classes: ", n_classes)
    # add only unique indices in batch
    # i.e. a single batch will only contain unique classes
    already_idxs = set()
    
    dataset = []
    for x in tqdm(range(num_triplets), disable=True): # disable progressbar
        if len(already_idxs) >= batch_size:
            # flush out when exceed batch size
            already_idxs = set()
        # get random unique class for anchor and positive point
        c1 = np.random.randint(0, n_classes)
        while c1 in already_idxs:
            c1 = np.random.randint(0, n_classes)
        
        n1 = np.random.randint(0, len(indices[c1]))
        already_idxs.add(c1)

        dataset.append([n1, c1])
        #triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
    
    #perm = np.random.permutation(len(labels))
    #sel_idx = perm[:num_triplets] # select first 0->num_triplets objects
    #sel_lbl = np.array(labels)[sel_idx] # select labels
    
    # map (idx_array), lbl_array to [[idx,lbl] ... [idx,lbl]]
    return np.array(dataset)#np.array([[sel,lbl] for sel, lbl in zip(sel_idx, sel_lbl)])


class HPatches():
    def __init__(self, train=True, transform=None, download=False, train_fnames=[],
                 test_fnames=[], denoise_model=None, use_clean=False):
        self.train = train
        self.transform = transform
        self.train_fnames = train_fnames
        self.test_fnames = test_fnames
        self.denoise_model = denoise_model
        self.use_clean = use_clean

    def set_denoise_model(self, denoise_model):
        self.denoise_model = denoise_model

    def denoise_patches(self, patches):
        batch_size = 1000 # 100
        for i in tqdm(range(int(len(patches) / batch_size)), file=sys.stdout):
            batch = patches[i * batch_size:(i + 1) * batch_size]
            batch = np.expand_dims(batch, -1)
            batch = np.clip(self.denoise_model.predict(batch).astype(int),
                                        0, 255).astype(np.uint8)[:,:,:,0]
            patches[i*batch_size:(i+1)*batch_size] = batch
        batch = patches[i*batch_size:]
        batch = np.expand_dims(batch, -1)
        batch = np.clip(self.denoise_model.predict(batch).astype(int),
                        0, 255).astype(np.uint8)[:,:,:,0]
        patches[i*batch_size:] = batch
        return patches

    def read_image_file(self, data_dir, train = 1):
        """Return a Tensor containing the patches
        """
        if self.denoise_model and not self.use_clean:
            print('Using denoised patches')
        elif not self.denoise_model and not self.use_clean:
            print('Using noisy patches')
        elif self.use_clean:
            print('Using clean patches')
        sys.stdout.flush()
        patches = []
        labels = []
        counter = 0
        hpatches_sequences = [x[1] for x in os.walk(data_dir)][0]
        if train:
            list_dirs = self.train_fnames
        else:
            list_dirs = self.test_fnames

        for directory in tqdm(hpatches_sequences, file=sys.stdout):
           if (directory in list_dirs):
            for tp in tps:
                if self.use_clean:
                    sequence_path = os.path.join(data_dir, directory, tp)+'.png'
                else:
                    sequence_path = os.path.join(data_dir, directory, tp)+'_noise.png'
                image = cv2.imread(sequence_path, 0)
                h, w = image.shape
                n_patches = int(h / w)
                for i in range(n_patches):
                    patch = image[i * (w): (i + 1) * (w), 0:w]
                    patch = cv2.resize(patch, (32, 32))
                    patch = np.array(patch, dtype=np.uint8)
                    patches.append(patch)
                    labels.append(i + counter)
            counter += n_patches

        patches = np.array(patches, dtype=np.uint8)
        if self.denoise_model and not self.use_clean:
            import pickle
            print('Denoising patches...')
            if os.path.isfile('cache/dataset/denoised.pth'):
                patches, labels = pickle.load(open('cache/dataset/denoised.pth', 'rb'))
            else:
                os.makedirs('cache/dataset', exist_ok=True)
                patches = self.denoise_patches(patches)
                pickle.dump((patches, labels), open('cache/dataset/denoised.pth', 'wb'))
                print('Saved cache!')
        return patches, labels


class DataGeneratorDesc(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, data, labels, num_triplets = 1000000, batch_size=50, dim=(32,32), n_channels=1, shuffle=True,
                out_triplets = True):
        # 'Initialization'
        self.transform = None
        self.out_triplets = out_triplets #True
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data = data
        self.labels = labels
        self.num_triplets = num_triplets
        self.on_epoch_end()

    def get_image(self, t):
        def transform_img(img):
            if self.transform is not None:
                img = transform(img.numpy())#self.transform(img.numpy())
            return img

        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a).astype(float)
        img_p = transform_img(p).astype(float)
        img_n = transform_img(n).astype(float)

        img_a = np.expand_dims(img_a, -1)
        img_p = np.expand_dims(img_p, -1)
        img_n = np.expand_dims(img_n, -1)
        if self.out_triplets:
            return img_a, img_p, img_n
        else:
            return img_a, img_p

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.triplets) / self.batch_size))
    
    def get_full_len(self):
        return len(self.triplets)
                
    def __getitem__(self, index):
        '''
            get_item returns a BATCH of patches indexed as a whole by index
            it is assumed returned achors in batch have all different classes!
        '''
        y = np.zeros((self.batch_size, 1))
        img_a = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_p = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_n = None # placeholder for pair returning only
        if self.out_triplets:
            img_n = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        for i in range(self.batch_size):
            t = self.triplets[self.batch_size*index + i]    
            img_a_t, img_p_t, img_n_t = self.get_image(t)
            img_a[i] = img_a_t
            img_p[i] = img_p_t
            if self.out_triplets:
                img_n[i] = img_n_t

        return {'a': img_a, 'p': img_p, 'n': img_n}, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        # note batch size must be consistent with batch size used not just 32!
        # this way u ensure all BATCH_SIZE achors are from a different class!
        # ***EDITED***
        self.triplets = generate_triplets(self.labels, self.num_triplets, 32)#self.batch_size # 32

    
    


class DataGeneratorDescWithLabel(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, data, labels, num_triplets = 1000000, batch_size=50, dim=(32,32), n_channels=1, shuffle=True,
                out_triplets = True):
        # 'Initialization'
        self.transform = None
        self.out_triplets = out_triplets #True
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data = data
        self.labels = labels
        self.num_triplets = num_triplets
        self.on_epoch_end()

    def get_image(self, t):
        def transform_img(img):
            if self.transform is not None:
                img = transform(img.numpy())#self.transform(img.numpy())
            return img

        a = self.data[t]

        img_a = transform_img(a).astype(float)

        img_a = np.expand_dims(img_a, -1)

        return img_a

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.triplets) / self.batch_size))
    
    def get_full_len(self):
        return len(self.triplets)
                
    def __getitem__(self, index):
        '''
            get_item returns a BATCH of patches indexed as a whole by index
            it is assumed returned achors in batch have all different classes!
        '''
        ## tf wants y values to be of shape self,batchsize!
        y = np.zeros((self.batch_size)) # label
        img_a = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        

        for i in range(self.batch_size):
            # now t == img_id, label
            img_id, lbl = self.triplets[self.batch_size*index + i]    
            img_a[i], y[i] = self.get_image(img_id), lbl
            # img_a[i] = img_a_t
            # y[i] = y_t

        return (img_a, y)

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        # note batch size must be consistent with batch size used not just 32!
        # this way u ensure all BATCH_SIZE achors are from a different class!
        # ***EDITED***
        self.triplets = generate_idx_label_pairs(self.labels, self.num_triplets, 32)#self.batch_size # 32



'''
    new combined data loaders
'''

class MultiDataLoader():
    '''
        Just pass it a denoise model if you want or not if you want to work
        on noisy data and it will do all the magic for you!

        Avail Objects:
            After `load_denoiser_dataset()`
                denoise_generator
                denoise_generator_val
            After `load_descr_dataset(denoise_model)`
                training_generator: For descriptor
                val_generator: For descriptor

    '''
    def __init__(self):

        #### data loading I ####
        self.hpatches_dir = './hpatches'
        self.splits_path = './splits.json'

        self.splits_json = json.load(open(self.splits_path, 'rb'))
        self.split = self.splits_json['a']

        self.train_fnames = self.split['train']
        self.test_fnames = self.split['test']

        seqs = glob.glob(self.hpatches_dir+'/*')
        seqs = [os.path.abspath(p) for p in seqs]   
        
        self.seqs = seqs
        self.seqs_train = list(filter(lambda x: os.path.split(x)[-1] in self.train_fnames, seqs)) 
        self.seqs_test = list(filter(lambda x: os.path.split(x)[-1] in self.split['test'], seqs)) 


        #### data loading II ####
        self.denoise_generator = None
        self.denoise_generator_val = None
        self.hPatches = None
        self.training_generator = None
        self.val_generator = None



    def load_denoiser_dataset(self, train_batch_size=50, val_batch_size=50, reduced_dataset=False):
        # Uncomment following lines for using all the data to train the denoising model
        print("=> Loading denoiser dataset...") # dataset for denoiser
        
        if reduced_dataset:
            import warnings
            warnings.warn("Reduced dataset for denoiser!")
            self.denoise_generator = DenoiseHPatches(random.sample(self.seqs_train, 3),
                                                     batch_size=train_batch_size, train_mode=True,
                                                     toy_data=True, cache_overwrite=True)
            self.denoise_generator_val = DenoiseHPatches(random.sample(self.seqs_test, 1), 
                                                         batch_size=val_batch_size, train_mode=False,
                                                        toy_data=True, cache_overwrite=True)
        else:
            self.denoise_generator = DenoiseHPatches(self.seqs_train, batch_size=train_batch_size, train_mode=True)
            self.denoise_generator_val = DenoiseHPatches(self.seqs_test, batch_size=val_batch_size, train_mode=False)

    

    def load_descr_dataset(self, denoise_model=None, use_clean=False,
                           train_batch_size=50, val_batch_size=50, data_loader_params={},
                           train_triplets=1000000, test_triplets=10000, return_img_lbl_pair=False):
        print("=> Loading descriptor dataset...") # dataset for descriptor
        #print("DESCR: batchSz: ")
        if use_clean == True:
            from warnings import warn
            warn("Descriptor will be trained & validated on clean dataset!")
        ### Descriptor loading and training
        # Loading images; # Creating training generator; # Creating validation generator
        self.hPatches = HPatches(train_fnames=self.train_fnames, test_fnames=self.test_fnames,
                            denoise_model=denoise_model, use_clean=use_clean)

        if not return_img_lbl_pair:
            ## each sample is a triplet usual case
            self.training_generator = DataGeneratorDesc(*self.hPatches.read_image_file(self.hpatches_dir, train=1),
                                                        num_triplets=train_triplets,  batch_size=train_batch_size)
            self.val_generator = DataGeneratorDesc(*self.hPatches.read_image_file(self.hpatches_dir, train=0),
                                                    num_triplets=test_triplets,  batch_size=val_batch_size)
        else:
            print("Using new img, lbl data loader...")
            self.training_generator = DataGeneratorDescWithLabel(*self.hPatches.read_image_file(self.hpatches_dir, train=1),
                                                        num_triplets=train_triplets,  batch_size=train_batch_size)
            self.val_generator = DataGeneratorDescWithLabel(*self.hPatches.read_image_file(self.hpatches_dir, train=0),
                                                    num_triplets=test_triplets,  batch_size=val_batch_size)            



if __name__ == "__main__":
    ### print dataset cardinalities

    data_loader = MultiDataLoader()
    
    
    # data_loader.load_denoiser_dataset()
    # print("Denoiser Dataset Cardinalities:\n", "TrainSet: ", 
    #       data_loader.denoise_generator.get_full_len(),
    #       "ValSet: ",
    #        data_loader.denoise_generator_val.get_full_len())
    # print("Denoiser Dataset 'BatchSize x NumBatches':\n", "TrainSet: %d x %d" % \
    #     (data_loader.denoise_generator.batch_size, \
    #           len(data_loader.denoise_generator)),
    #       "ValSet: %d x %d" % (data_loader.denoise_generator_val.batch_size, \
    #           len(data_loader.denoise_generator_val)))

    data_loader.load_descr_dataset(denoise_model=None, use_clean=True, return_img_lbl_pair=True, train_triplets=100, test_triplets=10)
    print("\nDescriptor Dataset Cardinalities:\n", "TrainSet: ", 
          data_loader.training_generator.get_full_len(),
          "ValSet: ",
           data_loader.val_generator.get_full_len())
    print("nDescriptor Dataset 'BatchSize x NumBatches':\n", "TrainSet: %d x %d" % \
        (data_loader.training_generator.batch_size, \
              len(data_loader.training_generator)),
          "ValSet: %d x %d" % (data_loader.val_generator.batch_size, \
              len(data_loader.val_generator)))