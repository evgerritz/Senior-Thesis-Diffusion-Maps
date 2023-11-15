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

def load_models(model_type_str='ResNet', model_type=ResNetCallig):
    model_names = os.listdir('models/restnet')
    models = {}

    datasets = load_data([10], transform = some_transforms, batch_size=256)
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

def get_rn_kernels():
    rn_models = load_models(model_type_str='ResNet', model_type=ResNetCallig)
    rn_kernels = [lambda img: model.get_intermediate_activation(img, skip_final=8) for model in rn_models]
    return rn_kernels

def kmeans_labs(phi_x, num_classes):
    kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init="auto").fit(phi_x)
    return kmeans.labels_


def nmi_labs(X, y, num_classes):
    # fix shapes
    shape = X.shape
    if len(shape) > 2:
        X = X.reshape(shape[0], -1)
    if len(y.shape) > 1:
        y = y.reshape(-1)

    # convert to real if complex (diffusion coord)
    X = np.real(X)
        
    predict_labs = kmeans_labs(X, num_classes)
    return normalized_mutual_info_score(predict_labs, y)

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

def kern_gaussian_euclid(xs, kwargs):
    sigma = kwargs['sigma']    
    dists = squareform(pdist(xs))
    W = np.exp(-dists**2 / (2*(sigma**2)))
    return W

def kern_cos_sim(xs, kwargs):
    norm = np.linalg.norm
    # f: embedding function
    f = kwargs.setdefault('f', lambda x: x)
    n = xs.shape[0]
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i >= j:
                xi = f(xs[i])
                xj = f(xs[j])
                norm_prod = norm(xi)*norm(xj)
                cos_sim = np.dot(xi, xj)/norm_prod
                W[i,j] = cos_sim
                W[j,i] = cos_sim
            else:
                break
    return W

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

def get_K(xs, kernel, **kwargs):
    return kernel(xs, kwargs)

def get_dmaps(X, kerns, kern_labs, t=1, m=10):
    dmaps = []
    for i in range(len(kerns)):
        kern = kerns[i]
        dmap, evalues = diffmap(X, t, m, kern_cos_sim, f=kern)
        dmaps.append(dmap)
        print(f'{kern_labs[i]}: {np.sum(evalues)}')
    return dmaps

def plot_dmaps(dmaps, y_callig, kern_labs, title, coords=[(0,1)]):
    nrows = len(coords)
    ncols = len(dmaps)
    fig, axes = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*4))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=.6) #adjust vertical spacing between subplots
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col] if nrows > 1 else axes[col]
            coord0, coord1 = coords[row]
            ax.scatter(dmaps[col][:,coord0], dmaps[col][:,coord1], c=y_callig)
            
            ax.set_title(kern_labs[col])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f'{coord0=}')
            ax.set_ylabel(f'{coord1=}')
    fig.tight_layout()
    

def save_dmap(name, dmap):
    with open(f'saved_objs/{name}.pkl', 'wb') as f:
        pickle.dump(dmap, f)
        
def load_dmap(name):
    with open(f'saved_objs/{name}.pkl', 'rb') as f:
        return pickle.load(f)

#plot_dmaps(dmaps, y_subset, kern_names, 'Diffusion Using Conv layer Output', coords=[(0,1),(0,2), (1,2)])