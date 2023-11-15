from models import ResNetCallig, VGG16
from util import load_data
from IAN_diffmaps import diffusionMapFromK
import os

# metrics
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans

# diffusion maps math tools
from scipy.sparse.linalg import eigs
from scipy.linalg import inv
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import fractional_matrix_power as power

datasets = load_data()

class CNNKernel:
    def __init__(self, model):
        self.model = model
        self.data = model.data
        self.name = f'{self.model.name}_{self.data.num_classes}'

    def f(self, img):
        activation = lambda img: self.model.get_intermediate_activation(img, skip_final=8)
        return activation


class Embedding:
    def __init__(self, kernel, subset, num_samples):
        self.kernel = kernel
        self.data = self.kernel.data
        self.kernname = self.kernel.name
        self.X = self.data.X
        self.y = self.data.y
        self.num_classes = self.data.num_classes
        self.subset = subset
        self.subset_str = self.subset.join('_')
        self.num_samples = num_samples
        self.X_sub_lin, self.y_sub = self.data.prep_for_embed(subset, num_samples)
        self.phi_x = self.phi_weight_matrix()
        self.name = f'{self.kernname}_{self.subset_str}_{self.num_samples}'
        

    def phi_weight_matrix(self):
        norm = np.linalg.norm
        f = self.kernel.f
        n = self.X_sub_lin.shape[0]
        W = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i >= j:
                    xi = f(self.X_sub_lin[i])
                    xj = f(self.X_sub_lin[j])
                    norm_prod = norm(xi)*norm(xj)
                    cos_sim = np.dot(xi, xj)/norm_prod
                    W[i,j] = cos_sim
                    W[j,i] = cos_sim
                else:
                    break
        return W
    
    def embed(self):
        print('Computing dmap:', self.name)
        dmap = diffusionMapFromK(self.phi_x, 5)
        saved = SavedDmap(self.name)
        saved.save(dmap)
        print('Saved dmap to', saved.path)
        return dmap

    def kmeans_labs(self):
        # fix shapes
        shape = self.phi_x.shape
        phi_x = self.phi_x
        if len(shape) > 2:
            phi_x = phi_x.reshape(shape[0], -1)

        # convert to real if complex (diffusion coord)
        phi_x = np.real(phi_x)

        kmeans = KMeans(n_clusters=self.num_classes, random_state=0, n_init="auto").fit(phi_x)
        return kmeans.labels_

    def nmi_labs(self):
        y = self.y
        if len(y.shape) > 1:
            y = y.reshape(-1)
        predict_labs = self.kmeans_labs()
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
    model_names = os.listdir('models/restnet')
    models = {}

    datasets = load_data(transform = some_transforms, batch_size=256)
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
            #diff_nmi = nmi_labs(dmaps[i], y_subset, c)
            #print(f'{kern_names[i]} diffusion nmi: {diff_nmi}')
            
#plot_dmaps(dmaps, y_subset, kern_names, 'Diffusion Using Conv layer Output', coords=[(0,1),(0,2), (1,2)])