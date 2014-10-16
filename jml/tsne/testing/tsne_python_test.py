import numpy
import pylab as Plot
import sys
from os import getenv

# WARNING: security risk; don't do this for anything that might be installed
sys.path.append(getenv("BIN"))

import tsne

print "numpy version", numpy.version.version

try:
    tsne.vectors_to_distances(None)
except:
    pass
else:
    assert ~"exception should have been thrown"

from gzip import GzipFile

digits = numpy.loadtxt(GzipFile(getenv("JML_TOP") + "/tsne/testing/mnist2500_X_min.txt.gz"));
labels = numpy.loadtxt(GzipFile(getenv("JML_TOP") + "/tsne/testing/mnist2500_labels_min.txt.gz"));


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

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-s", "--show-graph", dest="show_graph",
                  action="store_true", default=False,
                  help="display the graph once it's been generated")

parser.add_option("-a", "--all-data", dest="all_data",
                  action="store_true", default=False,
                  help="use all data (not just 250 rows)")

(options, args) = parser.parse_args()

def test_tsne():
    global digits, labels

    nrows = digits.shape[0];
    # Smaller number of labels for debugging...
    if not options.all_data:
        nrows = 250

    X = digits[range(nrows), ...]
    L = labels[range(nrows), ...]

    Y = tsne.tsne(X, 2, 50, 20.0, use_pca=True, max_iter=1000)
    #Y = tsne.tsne(X, 2, 50, 20.0, use_pca=False)

    for i in xrange(10):
        idxs = [idx for idx in xrange(len(L)) if L[idx] == i]
        c = Plot.get_cmap()(0.1 * i)
        Plot.scatter(Y[idxs,0], Y[idxs,1], 20, c, label="%d" % i)
    #Plot.axis('off')
    Plot.xticks([])
    Plot.yticks([])
    Plot.legend(loc='upper left', scatterpoints=1)
    if options.show_graph:
        Plot.show()


test_tsne()

#sys.exit(1)

