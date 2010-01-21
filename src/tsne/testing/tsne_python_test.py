import numpy as Math
import pylab as Plot
import sys
import tsne

print "numpy version", Math.version.version

def test_vectors_to_distances():

    from gzip import GzipFile

    X = Math.loadtxt(GzipFile("tsne/testing/mnist2500_X_min.txt.gz"));
    labels = Math.loadtxt(GzipFile("tsne/testing/mnist2500_labels_min.txt.gz"));

    nrows = X.shape[0];

    # Smaller number of labels for debugging...
    nrows = 100

    X = X[range(nrows), ...]
    labels = labels[range(nrows), ...]

    D = tsne.vectors_to_distances(X)

    print D

#    Y = tsne(X, 2, 50, 20.0, use_pca=False);
#    Plot.scatter(Y[:,0], Y[:,1], 20, labels);
#    Plot.legend(loc='lower left')
#    Plot.show()

test_vectors_to_distances()

sys.exit(1)

