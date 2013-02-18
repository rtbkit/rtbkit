/* stump_test1.cc
   Jeremy Barnes, 22 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Test program for the stump code.  This tests that the first iteration of
   training is exactly the same for both the binary-symmetric optimisation
   and the non-optimised versions.
*/

#include "jml/boosting/stump_training.h"
#include "jml/boosting/stump_training_core.h"
#include "jml/boosting/stump_training_bin.h"
#include "stump_testing.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/sparse_features.h"
#include "jml/boosting/stump_accum.h"


using namespace ML;
using namespace std;


void test_dense(const string & dataset)
{
    std::shared_ptr<Dense_Feature_Space> feature_space
        (new Dense_Feature_Space());
    std::shared_ptr<Dense_Training_Data> data;

    data.reset(new Dense_Training_Data(dataset, feature_space, 2));
    cerr << endl;
    cerr << "dataset \'" << dataset << "\': "
         << data->all_features().size() << " features, "
         << data->example_count() << " rows." << endl;
    cerr << "label freq: " << data->label_freq << endl;

    vector<Feature> features = data->all_features();

    /* Use the instrumented versions of the W and Z objects. */
    typedef W_testing<W_binsym, W_normal> W;
    typedef Z_testing<Z_binsym, Z_normal> Z;
    typedef C_normal C;

    //typedef Stump_Accum<W, Z, C, Stream_Tracer> Accum;
    //typedef Stump_Trainer<W, Z, Stream_Tracer> Trainer;
    typedef Stump_Accum<W, Z, C> Accum;
    typedef Stump_Trainer<W, Z> Trainer;
        
    Accum accum(*feature_space, true, 0);
    Trainer trainer;

    Trainer::All_Examples all;

    /* Test with one dimensional weights. */
    cerr << "one dimensional..." << endl;

    boost::multi_array<float, 2> weights1(data->example_count(), 1);
    weights1.fill(1.0 / (float)(data->example_count() * 2));

    cerr << "default w... " << endl;
    W default_w = trainer.calc_default_w(*data, all, weights1);
    cerr << "done." << endl;
    
    for (unsigned i = 0;  i < features.size();  ++i) {
        cerr << "feature " << i << " (" << feature_space->print(features[i])
             << ")... ";
        trainer.test(features[i], *data, weights1, all, default_w, accum);
        cerr << "done." << endl;
    }

    
    /* Test with 2 dimensional weights. */
    cerr << "two dimensional..." << endl;
    boost::multi_array<float, 2> weights2(data->example_count(), 2);
    weights2.fill(1.0 / (float)(data->example_count() * 2));

    cerr << "default w... " << endl;
    default_w = trainer.calc_default_w(*data, all, weights2);
    cerr << "done." << endl;

    for (unsigned i = 0;  i < features.size();  ++i) {
        cerr << "feature " << i << " (" << feature_space->print(features[i])
             << ")... ";
        trainer.test(features[i], *data, weights2, all, default_w, accum);
        cerr << "done." << endl;
    }
}

void test_sparse(const string & dataset)
{
    std::shared_ptr<Sparse_Feature_Space>
        feature_space(new Sparse_Feature_Space());

    Sparse_Training_Data data(dataset, feature_space);

    cerr << "dataset \'" << dataset << "\': "
         << data.all_features().size() << " features, "
         << data.example_count() << " rows." << endl;
    cerr << "label freq: " << data.label_freq << endl;

    vector<Feature> features = data.all_features();

    /* Use the instrumented versions of the W and Z objects. */
    typedef W_testing<W_binsym, W_normal> W;
    typedef Z_testing<Z_binsym, Z_normal> Z;
    typedef C_normal C;

    //typedef Stump_Accum<W, Z, C, Stream_Tracer> Accum;
    //typedef Stump_Trainer<W, Z, Stream_Tracer> Trainer;
    typedef Stump_Accum<W, Z, C> Accum;
    typedef Stump_Trainer<W, Z> Trainer;
        
    Accum accum(*feature_space, true, 0);
    Trainer trainer;

    Trainer::All_Examples all;

    /* Test with one dimensional weights. */
    cerr << "one dimensional..." << endl;

    boost::multi_array<float, 2> weights1(data.example_count(), 1);
    weights1.fill(1.0 / (float)(data.example_count() * 2));

    cerr << "default w... " << endl;
    W default_w = trainer.calc_default_w(data, all, weights1);
    cerr << "done." << endl;
    
    for (unsigned i = 0;  i < features.size();  ++i) {
        cerr << "feature " << i << " (" << feature_space->print(features[i])
             << ")... ";
        trainer.test(features[i], data, weights1, all, default_w, accum);
        cerr << "done." << endl;
    }

    
    /* Test with 2 dimensional weights. */
    cerr << "two dimensional..." << endl;
    boost::multi_array<float, 2> weights2(data.example_count(), 2);
    weights2.fill(1.0 / (float)(data.example_count() * 2));

    cerr << "default w... " << endl;
    default_w = trainer.calc_default_w(data, all, weights2);
    cerr << "done." << endl;

    for (unsigned i = 0;  i < features.size();  ++i) {
        cerr << "feature " << i << " (" << feature_space->print(features[i])
             << ")... ";
        trainer.test(features[i], data, weights2, all, default_w, accum);
        cerr << "done." << endl;
    }
}

int main(int argc, char ** argv)
try
{
    test_sparse("algorithms/machine_learning/boosting/datasets/"
                "presence-train-bin-1.txt");
    test_sparse("algorithms/machine_learning/boosting/datasets/"
                "path_scorer_2500.txt");
   test_dense("algorithms/machine_learning/boosting/datasets/lso-conf.txt");
    return 0;
}
catch (const std::exception & exc)
{
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
