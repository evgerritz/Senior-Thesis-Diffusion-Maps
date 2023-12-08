from models import ResNetCallig, VGG16, Model
from util import load_data, ALL_CLASSES
from visualize import plot_dmap, plot_3d, plot_images_as_points, dash_visualizer
from IAN_diffmaps import diffusionMapFromK

import os
import time
import pickle
from random import sample, shuffle
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# metrics, embeddings
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

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

vgg_models = load_models(model_type_str='VGG16', model_type=VGG16)
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
        self.weight_matrix = self.get_weight_matrix()

    def get_weight_matrix(self):
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

def diffmap(W, t, m, **kwargs):
    D = np.diag(np.sum(W, axis=1))
    Dinv12 = power(inv(D),1/2)
    M = inv(D)@W
    M_s = Dinv12@(W@Dinv12)
    
    evals, evecs = eigs(M_s,k=m+1)
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

def plot_embedding(embedding, title, fname=None, centroid_images=False, coord_f1=0, coord_f2=1, print_nmi=False):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    dmap = embedding.dmap
    full_label = lambda y, zh: embedding.data.full_label(embedding.data.get_label(class_no=y), chinese=zh)
    min_x = np.inf
    max_y = -np.inf

    for y_val in np.unique(embedding.y_sub):
        where = (embedding.y_sub == y_val).reshape(-1)
        xs = dmap[where, coord_f1]
        ys = dmap[where, coord_f2]
        min_x = min((min_x, min(xs)))
        max_y = max((max_y, max(ys)))
        ax.scatter(xs, ys, cmap='tab20', alpha=0.75, s=30, label=f'{full_label(y_val, False)} ({full_label(y_val, True)})')

        if centroid_images:
            imgs = embedding.X_sub_lin.reshape(-1, 64, 64)
            mean_x = np.mean(xs)
            mean_y = np.mean(ys)

            dists = ((xs - mean_x) ** 2 + (ys - mean_y) ** 2)
            closest = np.argmin(dists)
            img = imgs[where][closest]
            x_jitter = np.random.random() * np.mean(dmap[:, coord_f1]) * 50
            y_jitter = np.random.random() * np.mean(dmap[:, coord_f2]) * 50
            ab = AnnotationBbox(OffsetImage(img, zoom=0.4, cmap='gray', alpha=0.9),
                                (xs[closest] + x_jitter, ys[closest] + y_jitter), frameon=False)
            ax.add_artist(ab)

    pointwise_nmi = embedding.nmi_labs(True)
    if print_nmi:
        print(title)
        print(f'NMI: {pointwise_nmi}\n')

    ax.text(min_x, max_y, f'NMI: {round(pointwise_nmi, 3)}', ha='left', va='top', fontsize=11)

    ax.set_title(title, fontsize=12)
    if centroid_images:
        ax.legend(loc='lower right', fontsize=7)
    else:
        plt.legend(loc='upper right', fontsize='small')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if fname:
        plt.savefig(f'../../res/{fname}.png')
    #plt.show()

def cos_sim_plot():
    kern = CNNKernel(all_models[15], skip_final=0)
    kern.f = lambda img : img
    sub = ['shz', 'wzm', 'csl', 'hy']
    e = Embedding(kern, 200, subset=sub, print_subset=False)
    e.embed()
    plot_embedding(e, 'Cosine Similarity Diffusion Maps Embedding', 'cos_sim_plot')

def vgg_plot_in_sample():
    sub = ['yzq', 'lx', 'oyx', 'lqs']
    e = get_embedding(vgg_models[15], 3, 200, subset=sub, print_subset=True)
    plot_embedding(e, 'VGG16 Kernel Diffusion Maps Embedding of In-sample Calligraphers', 'vgg_plot_in_sample')

def vgg_plot_out_of_sample():
    sub = ['shz', 'wzm', 'csl', 'hy']
    e = get_embedding(vgg_models[15], 8, 100, subset=sub, print_subset=True)
    plot_embedding(e, 'VGG16 Kernel Diffusion Maps Embedding of Out-of-sample Calligraphers', 'vgg_plot_out_of_sample')

def resnet_plot_oos():
    sub = ['shz', 'wzm', 'csl', 'hy']
    e = get_embedding(all_models[15], 3, 200, rand_subset_size=4, print_subset=True)
    plot_embedding(e, 'ResNet-18 Kernel Diffusion Maps Embedding of Out-of-sample Calligraphers', 'resnet_plot_out_of_sample')

def resnet_plot_10(subset=ALL_CLASSES[:len(ALL_CLASSES)//2]):
    # maybe line straightness left to right?
    # line width variation top to bottom?
    # does triangle shape mean anything?
    e = get_embedding(all_models[15], 3, 50, subset=subset, print_subset=True)
    plot_embedding(e, 'ResNet-18 Kernel Diffusion Maps Embedding', 'resnet_plot_10_imgs', True, 0, 1)
    plot_embedding(e, 'ResNet-18 Kernel Diffusion Maps Embedding', 'resnet_plot_10_imgs_1_2', True, 1, 2)
    plot_embedding(e, 'ResNet-18 Kernel Diffusion Maps Embedding', 'resnet_plot_10_imgs_0_2', True, 0, 2)
    #plot_embedding(e, 'ResNet-18 Kernel Diffusion Maps Embedding', 'resnet_plot_10')

oos_5 = ['shz', 'wzm', 'csl', 'hy', 'mf']
oos_4 = ['shz', 'wzm', 'csl', 'hy']
def find_good_subset():
    oos = ['shz', 'wzm', 'csl', 'hy', 'mf']
    subs = list(itertools.combinations(set(ALL_CLASSES) - set(oos), 5))
    shuffle(subs)
    for half_subset in subs:
        subset = oos + list(half_subset)
        kern = CNNKernel(all_models[15], skip_final=3)
        embedding = Embedding(kern, 50, subset=subset, print_subset=True)
        print(embedding.nmi_labs(False))

def embed_non_dmap(model, dmap_func, skip_final):
    kern = CNNKernel(model, skip_final=skip_final)
    e = Embedding(kern, 200, subset=oos_5, print_subset=True)
    e.dmap = dmap_func(e)
    return e

def plot_all_non_dmap(embed_func, embed_name):
    e = embed_non_dmap(all_models[15], lambda e: embed_func(e.X_sub_lin), 3)
    plot_embedding(e, f'{embed_name} Embedding of Out-of-sample Calligraphers', f'{embed_name.lower()}_no_kern_plot', print_nmi=True)
    e2 = embed_non_dmap(vgg_models[15], lambda e: embed_func(e.phi_x), 3)
    plot_embedding(e2, f'VGG16 {embed_name} Embedding of Out-of-sample Calligraphers', f'{embed_name.lower()}_vgg16_plot', print_nmi=True)
    e3 = embed_non_dmap(all_models[15], lambda e: embed_func(e.phi_x), 3)
    plot_embedding(e3, f'ResNet-18 {embed_name} Embedding of Out-of-sample Calligraphers', f'{embed_name.lower()}_resnet_plot', print_nmi=True)

def plot_pca():
    plot_all_non_dmap(PCA(n_components=3).fit_transform, 'PCA')

def plot_tsne():
    plot_all_non_dmap(TSNE(n_components=3).fit_transform, 'TSNE')

def plot_kernel_pca():
    plot_all_non_dmap(KernelPCA(n_components=3, kernel='rbf').fit_transform, 'KernelPCA')

if __name__ == '__main__':
    #print(skip_layer_nmi(model_num_classes=18, print_results=True))
    #e = get_embedding(all_models[15], 3, 500, subset=ALL_CLASSES[:len(ALL_CLASSES)//2])
    #e = get_embedding(all_models[16], 3, 500, rand_subset_size=4)
    #plot_3d(e)
    #plot_images_as_points(e, 0, 1)
    #dash_visualizer(e).run()

    #cos_sim_plot()
    #vgg_plot_in_sample()
    #vgg_plot_out_of_sample()
    #resnet_plot_oos()
    #resnet_plot_10(['zmf', 'mzd', 'fwq', 'shz', 'gj', 'mf', 'bdsr', 'lqs', 'yzq', 'lx'])

    plot_pca()
    plot_tsne()
    plot_kernel_pca()

