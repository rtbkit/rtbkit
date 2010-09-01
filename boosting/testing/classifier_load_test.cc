/* classifier_load_test.cc
   Jeremy Barnes, 23 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Test that the classifiers can be loaded properly.
*/

#include "jml/boosting/classifier.h"
#include <iostream>

using namespace std;
using namespace ML;

int main(int argc, char ** argv)
try
{
    if (argc != 2) {
        cerr << "usage: " << argv[0] << " classifier.cls" << endl;
        exit(1);
    }

    Classifier classifier;
    classifier.load(argv[1]);
    //cerr << "classifier().feature_space().get() = "
    //     << classifier.feature_space().get() << endl;

    cerr << "classifier is of type " << classifier.impl->class_id()
         << endl;
    cerr << "feature space is of type "
         << classifier.feature_space()->class_id() << endl;
}
catch (const std::exception & exc) {
    cerr << "error: " << exc.what() << endl;
}
catch (...) {
    cerr <<"error: unknown exception thrown" << endl;
}
