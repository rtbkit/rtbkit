import numpy
import pylab as Plot
import sys

# WARNING: security risk; don't do this for anything that might be installed
sys.path.append('../build/x86_64/bin')

import tsne

print "numpy version", numpy.version.version

try:
    tsne.vectors_to_distances(None)
except:
    pass
else:
    assert ~"exception should have been thrown"

from gzip import GzipFile

digits = numpy.loadtxt(GzipFile("tsne/testing/mnist2500_X_min.txt.gz"));
labels = numpy.loadtxt(GzipFile("tsne/testing/mnist2500_labels_min.txt.gz"));


def test_vectors_to_distances():
    global digits, labels

    nrows = digits.shape[0];

    # Smaller number of labels for debugging...
    nrows = 100

    X = digits[range(nrows), ...]
    D = tsne.vectors_to_distances(X)

    print D

    P = tsne.distances_to_probabilities(D)

    print P

    print numpy.sum(P)

#    Y = tsne(X, 2, 50, 20.0, use_pca=False);
#    Plot.scatter(Y[:,0], Y[:,1], 20, labels);
#    Plot.legend(loc='lower left')
#    Plot.show()




test_vectors_to_distances()


def test_tsne():
    global digits, labels

    nrows = digits.shape[0];
    # Smaller number of labels for debugging...
    nrows = 500

    X = digits[range(nrows), ...]
    L = labels[range(nrows), ...]

    Y = tsne.tsne(X, 2, 50, 20.0, use_pca=True, max_iter=1000)
    #Y = tsne.tsne(X, 2, 50, 20.0, use_pca=False)
    Plot.scatter(Y[:,0], Y[:,1], 20, L)
    Plot.legend(loc='lower left')
    Plot.show()


test_tsne()

sys.exit(1)

