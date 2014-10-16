import _tsne


def vectors_to_distances(array):
    return _tsne.vectors_to_distances(array)


def distances_to_probabilities(array, tolerance=1e-5, perplexity=30.0):
    return _tsne.distances_to_probabilities(array, tolerance, perplexity)


def tsne_core(array, num_dims=2, **kwargs):
    return _tsne.tsne(array, num_dims, **kwargs)


def pca(X, no_dims=50):
    """
    Runs PCA on the NxD array X in order to reduce its dimensionality to
    no_dims dimensions.
    """

    import numpy as Math

    print "Preprocessing the data using PCA..."
    (n, d) = X.shape
    X = X - Math.tile(Math.mean(X, 0), (n, 1))
    X = Math.asarray(X, "float32")
    (l, M) = Math.linalg.eig(Math.dot(X.T, X))
    M = Math.asarray(M, "float32")
    Y = Math.dot(X, M[:, 0:no_dims])

    print "Y", Y.dtype

    return Y


def tsne(X, num_dims=2, initial_dims=50, perplexity=30.0, use_pca=True,
         **kwargs):

    if use_pca:
        X = pca(X, initial_dims)
    (n, d) = X.shape

    D = vectors_to_distances(X)

    P = distances_to_probabilities(D, perplexity=perplexity)

    print kwargs

    return tsne_core(P, num_dims, **kwargs)
