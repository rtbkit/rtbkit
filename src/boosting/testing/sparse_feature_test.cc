/* sparse_feature_test.cc
   Jeremy Barnes, 26 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Test of the sparse feature space.
*/

#include "jml/boosting/sparse_features.h"

using namespace std;
using namespace ML;

int main(int argc, char ** argv)
try
{
    if (argc < 2) {
        cerr << "usage: " << argv[0] << " filename" << endl;
        exit(1);
    }

    Sparse_Training_Data data(argv[1]);

    cerr << "label frequency: " << data.label_freq << endl;
    cerr << "example_count = " << data.example_count() << endl;
    cerr << "feature count = " << data.feature_index.size() << endl;
}
catch (const std::exception & exc) {
    cerr << "error: " << exc.what() << endl;
}
