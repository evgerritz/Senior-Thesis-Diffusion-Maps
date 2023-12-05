from models import ResNetCallig, VGG16, Model
from util import load_data, ALL_CLASSES
from visualize import plot_dmap, plot_3d, plot_images_as_points, dash_visualizer, plot_cluster_images
from IAN_diffmaps import diffusionMapFromK

import os
import time
import pickle
import numpy as np
import matplotlib
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

# enable Chinese support in plots
matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP']

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

#vgg_models = load_models(model_type_str='', model_type=ResNetCallig):
all_models = load_models()
all_data = all_models[20].data

class CNNKernel:
    def __init__(self, model, skip_final=8):
        self.model = model
        self.data = model.data
        self.name = f'{self.model.model.name}_{self.data.num_classes}'
        self.f = lambda img: self.model.get_intermediate_activation(img, skip_final)

class Embedding:
    def __init__(self, kernel, num_samples, subset=None, rand_subset_size=None, print_subset=False):
        self.kernel = kernel
        self.data = all_data
        self.train_data = self.kernel.data
        self.kernname = self.kernel.name
        self.phi = self.kernel.f 

        self.num_classes = self.data.num_classes
        self.num_samples = num_samples
        if not subset:
            not_in_train = set(ALL_CLASSES) - set(self.train_data.classes)
            subset = sample(sorted(not_in_train), rand_subset_size)
        self.subset = subset
        if print_subset:
            print(subset)
        self.subset_str = '_'.join(self.subset)
        self.name = f'{self.kernname}_{self.subset_str}_{self.num_samples}'

        self.X = self.data.X
        self.y = self.data.y
        self.num_samples = num_samples
        self.X_sub_lin, self.y_sub = self.data.prep_for_embed(subset, num_samples)

        self.phi_x = np.apply_along_axis(self.phi, 1, self.X_sub_lin)
        self.weight_matrix = self.weight_matrix()

    def weight_matrix(self):
        norms = np.linalg.norm(self.phi_x, axis=1, keepdims=True)
        norm_phi_x = self.phi_x / norms
        cosine_similarity = np.matmul(norm_phi_x, norm_phi_x.T)
        return cosine_similarity
    
    def embed(self, print_info=False):
        start = time.time()
        if print_info:
            print('Computing dmap:', self.name)
        dmap, evals = diffusionMapFromK(self.weight_matrix, 5)
        self.dmap = dmap
        if print_info:
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

        kmeans = KMeans(n_clusters=len(self.subset), random_state=0, n_init="auto").fit(x)
        return kmeans.labels_

    def nmi_labs(self, dmap=False):
        y = self.y_sub
        if len(y.shape) > 1:
            y = y.reshape(-1)
        predict_labs = self.kmeans_labs(dmap)
        return normalized_mutual_info_score(predict_labs, y)

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

def get_embedding(model, skip_final, num_samples, subset=None, rand_subset_size=None, print_subset=False):
    kern = CNNKernel(model, skip_final=skip_final)
    embedding = Embedding(kern, 200, subset=subset, rand_subset_size=rand_subset_size, print_subset=print_subset)
    embedding.embed()
    return embedding

def skip_layer_nmi(model_num_classes=15, print_results=True):
    nmis = []
    for skip_final in range(1,10):
        model = all_models[model_num_classes]
        rand_size = 4 if model_num_classes != 18 else 2
        embedding = get_embedding(model, skip_final, 500, rand_subset_size=rand_size)    
        nmi = embedding.nmi_labs(True)
        nmis.append(nmi)
        if print_results:
            print(f'{model.data.num_classes} {skip_final} from ll: {nmi} with {embedding.subset}')
    return np.argmax(nmis) + 1
    
def print_layers(model):
    for i, layer in enumerate(model.model.children()):
        print(i)
        print(layer)

def plot_embedding(e, title, fname):
    full_label = lambda y, zh: e.data.full_label(e.data.get_label(class_no=y), chinese=zh)

    fig = plt.figure(figsize=(6,6))
    min_x = np.inf; max_y = -np.inf
    for c_no in np.unique(e.y_sub):
        where = (e.y_sub == c_no).reshape(-1)
        x = e.dmap[where, 0]
        y = e.dmap[where, 1]
        min_x = min((min_x, min(x)))
        max_y = max((max_y, max(y)))
        plt.scatter(x,y, cmap='tab20', alpha=0.75, s=30, label=f'{full_label(c_no, False)} ({full_label(c_no, True)})')
        pointwise_nmi = e.nmi_labs(True)
        plt.legend()
    plt.text(min_x, max_y, f'NMI: {round(pointwise_nmi,3)}', ha='left', va='top', fontsize=12)

    plt.title(title, fontsize=15)
    plt.xticks([]);
    plt.yticks([]);
    plt.legend(loc='upper right', fontsize='small')
    fig.tight_layout()
    plt.savefig(f'../../res/{fname}.png')
    plt.show()

def cos_sim_plot():
    kern = CNNKernel(all_models[15], skip_final=0)
    kern.f = lambda img : img
    sub = ['shz', 'mf', 'wzm', 'csl', 'hy']
    e = Embedding(kern, 200, subset=sub, print_subset=False)
    e.embed()
    plot_embedding(e, 'Cosine Similarity Diffusion Maps Embedding', 'cos_sim_plot')

def vgg_plot_in_sample():
    sub = ['shz', 'mf', 'wzm', 'csl', 'hy'] # fix
    e = get_embedding(vgg_models[15], 3, 200, subset=sub, print_subset=True)
    plot_embedding(e, 'VGG16 Kernel Diffusion Maps Embedding of In-sample Calligraphers', 'vgg_plot_in_sample')

def vgg_plot_out_of_sample():
    sub = ['shz', 'mf', 'wzm', 'csl', 'hy']
    e = get_embedding(vgg_models[15], 3, 200, rand_subset_size=5, print_subset=True)
    plot_embedding(e, 'ResNet-18 Kernel Diffusion Maps Embedding', 'vgg_plot_out_of_sample')

def resnet_plot_oos():
    e = get_embedding(all_models[15], 3, 200, rand_subset_size=5, print_subset=True)
    plot_embedding(e, 'ResNet-18 Kernel Diffusion Maps Embedding', 'resnet_plot_oos')

def resnet_plot_10():
    # maybe line straightness left to right?
    # line width variation top to bottom?
    # does triangle shape mean anything?
    e = get_embedding(all_models[15], 3, 200, subset=ALL_CLASSES[:len(ALL_CLASSES)//2], print_subset=True)
    plot_cluster_images(e, 'ReseNet-18 Kernel Diffusion Maps Embedding', 'resnet_plot_10_imgs')
    #plot_embedding(e, 'ResNet-18 Kernel Diffusion Maps Embedding', 'resnet_plot_10')

if __name__ == '__main__':
    #print(skip_layer_nmi(model_num_classes=10, print_results=True))
    #print(skip_layer_nmi(model_num_classes=15, print_results=True))
    #print(skip_layer_nmi(model_num_classes=16, print_results=True))
    #print(skip_layer_nmi(model_num_classes=18, print_results=True))
    #e = get_embedding(all_models[15], 3, 500, subset=ALL_CLASSES[:len(ALL_CLASSES)//2])
    #e = get_embedding(all_models[16], 3, 500, rand_subset_size=4)
    #plot_dmap(e, [(0,1), (1,2), (0,1)])
    #plot_3d(e)
    #plot_images_as_points(e, 0, 1)
    #dash_visualizer(e).run()
    #cos_sim_plot()
    resnet_plot_10()

