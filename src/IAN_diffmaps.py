"""
Code from: https://github.com/dyballa/IAN
Copyright (c) 2022, Luciano Dyballa
"""
import numpy as np
import scipy as sp

def diffusionMapFromK(K, n_components, alpha=0, t=1, tol=1e-8, lambdaScale=True,
    returnPhi=False, returnOrtho=False, unitNorm=False, sparse_eigendecomp=True, use_svd=True):
    """Computes a diffusion map embedding from a kernel matrix"""

    if sp.sparse.issparse(K):
        print('sparse!')
        return diffusionMapSparseK(K, n_components, alpha, t, lambdaScale,
                    returnPhi, returnOrtho, unitNorm, use_svd)
    else:
        return diffusionMapK(K, n_components, alpha, t, tol, lambdaScale,
                    returnPhi, returnOrtho, unitNorm, sparse_eigendecomp, use_svd)


def diffusionMapSparseK(K, n_components, alpha=0, t=1, lambdaScale=True, 
    returnPhi=False, returnOrtho=False, unitNorm=False, use_svd=True):
    """ alpha: 0 (markov), 1 (laplace-beltrami), 0.5 (fokker-plank) """
    assert alpha >= 0
    assert sp.sparse.issparse(K)
        
    N = K.shape[0]

    eps = np.finfo(K.dtype).eps

    #normalize (kernel)-adjacency matrix by each node's degree
    if alpha > 0:
        D = np.array(K.sum(axis=1)).ravel()
        Dalpha = np.power(D, -alpha)
        #right-normalize
        Dalpha = sp.sparse.spdiags(Dalpha, 0, N, N)
        K = Dalpha * K * Dalpha


    sqrtD = np.sqrt(np.array(K.sum(axis=1)).ravel()) + eps

    #symmetrizing Markov matrix by scaling rows and cols by 1/sqrt(D)
    sqrtDs = sp.sparse.spdiags(1/sqrtD, 0, N, N)


    Ms = sqrtDs * K * sqrtDs

    #ensure symmetric numerically
    Ms = Ms.maximum(Ms.transpose()).tocsr()

    SPARSETOL = 2*eps
    #bring out the zeros before eigendecomp (equiv. to: Ms[Ms < SPARSETOL] = 0)
    for i in range(N):
        Ms[i].data[Ms[i].data < SPARSETOL] = 0
    Ms.eliminate_zeros()

    k = n_components + 1

    assert k <= N, "sparse routines require n_components + 1 < N"

    if use_svd:
        U, lambdas, _ = sp.sparse.linalg.svds(Ms, k=k, return_singular_vectors='u')
    else:
        lambdas, U = sp.sparse.linalg.eigsh(Ms,k)#,overwrite_a=True,check_finite=False)


    #sort in decreasing order of evals
    idxs = lambdas.argsort()[::-1]
    lambdas = lambdas[idxs]
    U = U[:,idxs] #Phi

    if returnOrtho:
        Psi = U / U[:,0:1]
    elif returnPhi:
        assert sqrtD.ndim == 1 #assert col vector
        Psi = U * sqrtD[:,None]
    else:
        assert sqrtD.ndim == 1 #assert col vector
        Psi = U / sqrtD[:,None]
        #Phi = U * sqrtD
        #assert np.all(np.isclose((Phi.T @ Psi),np.eye(Psi.shape[1])))

    #make Psi vectors have unit norm
    if unitNorm:
        Psi = Psi / np.linalg.norm(Psi,axis=0,keepdims=1)

    if lambdaScale:
        assert (lambdas.ndim == 1 and lambdas.shape[0] == Psi.shape[1])
        if t == 0:
            diffmap = Psi * np.power(-1/(lambdas-1),.5)
        else:
            diffmap = Psi * np.power(lambdas,t)

    else: diffmap = Psi

    return diffmap[:,1:],lambdas[1:]

def diffusionMapK(K, n_components, alpha=0, t=1, tol=1e-8, lambdaScale=True,
    returnPhi=False, returnOrtho=False, unitNorm=False, sparse_eigendecomp=True, use_svd=True):
    """ alpha: 0 (markov), 1 (laplace-beltrami), 0.5 (fokker-plank) """

    assert alpha >= 0
    if sp.sparse.issparse(K):
        K = K.toarray()

    #assert symmetry
    assert np.all(np.isclose(K - K.T,0))

    N = K.shape[0]

    eps = np.finfo(K.dtype).eps

    #normalize (kernel) adjacency matrix by each node's degree
    if alpha > 0:
        #find degree q for each row
        D = K.sum(axis=1,keepdims=1) #always >= 1
        K = K / np.power(D @ D.T,alpha)

        
    #ignore kernel vals that are too small
    K[K < tol] = 0

    #symmetrizing Markov matrix by scaling rows and cols by 1/sqrt(D)
    sqrtD = np.sqrt(K.sum(axis=1,keepdims=1)) + eps #could be zero!
    Ms = K / (sqrtD @ sqrtD.T)

    #ensure symmetric numerically
    Ms = 0.5*(Ms + Ms.T)

    SPARSETOL = 2*eps
    Ms[Ms < SPARSETOL] = 0 #bring out the zeros before converting to sparse

    k = n_components + 1

    assert k <= N
    if k == N:
        sparse_eigendecomp = False #cannot compute all evals/evecs when sparse!

    if sparse_eigendecomp:
        sMs = sp.sparse.csc_matrix(Ms)
        if use_svd: #sparse svd
            U, lambdas, _ = sp.sparse.linalg.svds(sMs, k=k, return_singular_vectors='u')
        else:
            lambdas, U = sp.sparse.linalg.eigsh(sMs,k)#,overwrite_a=True,check_finite=False)
    else:
        if use_svd:
            U, lambdas, _ = np.linalg.svd(Ms, full_matrices=False)
        else:
            lambdas, U = np.linalg.eigh(Ms)


    #sort in decreasing order of eigenvalues
    idxs = lambdas.argsort()[::-1]
    lambdas = lambdas[idxs]
    U = U[:,idxs] #Phi
    
    if not sparse_eigendecomp:
        lambdas = lambdas[:k+1]
        U = U[:,:k+1]

    if returnOrtho:
        Psi = U / U[:,0:1]
    elif returnPhi:
        assert sqrtD.shape[1] == 1 #assert col vector
        Psi = U * sqrtD
    else:
        assert sqrtD.shape[1] == 1 #assert col vector
        Psi = U / sqrtD
        #Phi = U * sqrtD
        #assert np.all(np.isclose((Phi.T @ Psi),np.eye(Psi.shape[1])))

    #make Psi vectors have unit norm before scaling by eigenvalues
    if unitNorm:
        Psi = Psi / np.linalg.norm(Psi,axis=0,keepdims=1)

    if lambdaScale:
        assert (lambdas.ndim == 1 and lambdas.shape[0] == Psi.shape[1])
        diffmap = Psi * np.power(lambdas,t)
    else: diffmap = Psi

    return diffmap[:,1:],lambdas[1:]