from models import ResNetCallig, VGG16, Model
from util import load_data, ALL_CLASSES
from IAN_diffmaps import diffusionMapFromK

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import sample

# metrics
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans

# diffusion maps math tools
from scipy.sparse.linalg import eigs
from scipy.linalg import inv
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import fractional_matrix_power as power

class CNNKernel:
    def __init__(self, model, skip_final=8):
        self.model = model
        self.data = model.data
        self.name = f'{self.model.model.name}_{self.data.num_classes}'
        self.f = lambda img: self.model.get_intermediate_activation(img, skip_final)

class Embedding:
    def __init__(self, kernel, num_samples, subset=None, rand_subset_size=None):
        self.kernel = kernel
        self.data = self.kernel.data
        self.kernname = self.kernel.name
        self.phi = self.kernel.f 

        self.num_classes = self.data.num_classes
        self.num_samples = num_samples
        if not subset:
            not_in_train = set(ALL_CLASSES) - set(self.data.classes)
            subset = sample(sorted(not_in_train), rand_subset_size)
        self.subset = subset
        self.subset_str = '_'.join(self.subset)
        self.name = f'{self.kernname}_{self.subset_str}_{self.num_samples}'

        self.X = self.data.X
        self.y = self.data.y
        self.num_samples = num_samples
        self.X_sub_lin, self.y_sub = self.data.prep_for_embed(subset, num_samples)

        self.phi_x = np.apply_along_axis(self.phi, 1, self.X_sub_lin)
        #self.phi_x = np.array([self.phi(x) for x in self.X_sub_lin])
        self.weight_matrix = self.weight_matrix()

    def weight_matrix(self):
        norms = np.linalg.norm(self.phi_x, axis=1, keepdims=True)
        norm_phi_x = self.phi_x / norms
        cosine_similarity = np.matmul(norm_phi_x, norm_phi_x.T)
        return cosine_similarity
    
    def embed(self):
        start = time.time()
        print('Computing dmap:', self.name)
        dmap, evals = diffusionMapFromK(self.weight_matrix, 5)
        self.dmap = dmap
        saved = SavedDmap(self.name)
        saved.save(dmap)
        print('Saved dmap to', saved.path, 'in', time.time()-start, "secs")
        return dmap

    def kmeans_labs(self, dmap=False):
        # fix shapes
        if not dmap:
            x = self.phi_x
        else:
            x = self.dmap
        shape = x.shape
        if len(shape) > 2:
            x = x.reshape(shape[0], -1)

        # convert to real if complex (diffusion coord)
        x = np.real(x)

        kmeans = KMeans(n_clusters=self.num_classes, random_state=0, n_init="auto").fit(x)
        return kmeans.labels_

    def nmi_labs(self, dmap=False):
        y = self.y_sub
        if len(y.shape) > 1:
            y = y.reshape(-1)
        predict_labs = self.kmeans_labs(dmap)
        return normalized_mutual_info_score(predict_labs, y)

"""
def diffmap(xs, t, m, kernel, **kwargs):
    W = kernel(xs, kwargs)
    D = np.diag(np.sum(W, axis=1))
    Dinv12 = power(inv(D),1/2)
    M = inv(D)@W
    M_s = Dinv12@(W@Dinv12)
    
    evals, evecs = eigs(M_s,k=len(xs))
    indices = np.argsort(-1*evals)
    evals = evals[indices]
    evecs = evecs[:,indices]
    evecs = Dinv12@evecs
    
    dmap = (evecs@(np.diag(evals)**t))[:,1:m+1]
    return dmap, (evals[1:m+1])**t
"""

class SavedDmap:
    def __init__(self, name):
        self.name = name
        self.path = f'saved_objs/SavedDmap/{self.name}.pkl'

    def save(self, dmap):
        with open(self.path, 'wb') as f:
            pickle.dump(dmap, f)
            
    def load(self):
        with open(self.path, 'rb') as f:
            return pickle.load(f)

def load_models(model_type_str='ResNet', model_type=ResNetCallig):
    model_names = os.listdir('saved_models/')
    models = {}

    datasets = load_data(refresh=False)
    for model_name in model_names:
        if model_type_str in model_name:
            num_classes = int(model_name[-6:-4])
            for data in datasets:
                if data.num_classes == num_classes:
                    model = Model(model_type, data)
                    model.load_model()
                    models[num_classes] = model
                    break

    return models

def compare_nmi(kerns, X, y, num_classes, dmaps=None):
    baseline_nmi = nmi_labs(X, y, num_classes)
    print(f'{baseline_nmi=}')
    #cheat_nmi = nmi_labs(np.real(dmaps_cheat[0][:,:2]), y_subset, c)
    #print(f'{cheat_nmi=}')
    for i in range(len(kerns)):
        phi_x = np.array([kerns[i](x) for x in X_sub_lin])
        nmi = nmi_labs(phi_x, y_subset, c)
        print(f'{kern_names[i]} nmi: {nmi}')
        if dmaps:
            pass
            #diff_nmi = nmi_labs(dmaps[i], y_subset, c)
            #print(f'{kern_names[i]} diffusion nmi: {diff_nmi}')
            
#plot_dmaps(dmaps, y_subset, kern_names, 'Diffusion Using Conv layer Output', coords=[(0,1),(0,2), (1,2)])

def get_dmap(model, skip_final, num_samples, subset):
    kern = CNNKernel(models[10], skip_final=1)
    e1 = Embedding(kern, 200, ['oyx', 'lx', 'lgq', 'yzq'])
    #e1 = Embedding(kern, kern.data.classes, 200)
    print(e1.nmi_labs())
    dmap = e1.embed()
    print(e1.nmi_labs(True))

if __name__ == '__main__':
    models = load_models()

    
    #print(e1.data.classes)
    #plt.scatter(dmap[:,0], dmap[:,1], c=e1.y_sub)
    #plt.show()