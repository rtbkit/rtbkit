/* stump_test4.cc
   Jeremy Barnes, 22 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Tests the adjust_w function.  Required dataset:

   SPARSE feature space
   1 isone:1
   0 iszero:1
*/

#include "jml/boosting/stump_training.h"
#include "jml/boosting/stump_training_core.h"
#include "jml/boosting/stump_training_bin.h"
#include "stump_testing.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/sparse_features.h"

using namespace ML;
using namespace std;

void test()
{
    //const char * dataset1 =
    //    "SPARSE feature space\n"
    //    "1 isone:1\n"
    //    "0 iszero:1\n";
    
    std::shared_ptr<Sparse_Feature_Space>
        feature_space(new Sparse_Feature_Space());

    Sparse_Training_Data data("algorithms/machine_learning/boosting/datasets/"
                              "presence-test1.txt",
                              feature_space, 2);

    vector<Feature> features = data.all_features();

    /* Use the instrumented versions of the W and Z objects. */
    typedef W_testing<W_binsym, W_normal> W;
    typedef Stump_Trainer<W> Trainer;
    Trainer trainer;

    boost::multi_array<float, 2> weights(data.example_count(), 2);
    weights.fill(1.0 / (float)(data.example_count() * 2));
    
    W default_w = trainer.calc_default_w(data, weights);

    float good_default_w[2][3][2] =
        { { { 0.00, 0.00 }, {0.00, 0.00}, {0.25, 0.25} },
          { { 0.00, 0.00 }, {0.00, 0.00}, {0.25, 0.25} } };

    for (unsigned i = 0;  i < 2;  ++i) {
        for (unsigned j = 0;  j < 3;  ++j) {
            for (unsigned k = 0;  k < 2;  ++k) {
                if (default_w(i, j, k) != good_default_w[i][j][k])
                    throw Exception(format("default_w(%d, %d, %d) = %f,"
                                           "should be %f", i, j, k,
                                           (float)default_w(i, j, k),
                                           good_default_w[i][j][k]));
            }
        }
    }
    
    trainer.adjust_w(default_w, data, weights, Feature(0, 0, 0), true);

    float good_w[2][3][2] =
        { { { 0.00, 0.00 }, {0.25, 0.00}, {0.00, 0.25} },
          { { 0.00, 0.00 }, {0.00, 0.25}, {0.25, 0.00} } };
    
    for (unsigned i = 0;  i < 2;  ++i) {
        for (unsigned j = 0;  j < 3;  ++j) {
            for (unsigned k = 0;  k < 2;  ++k) {
                if (default_w(i, j, k) != good_w[i][j][k])
                    throw Exception(format("w(%d, %d, %d) = %f, "
                                           "should be %f", i, j, k,
                                           (float)default_w(i, j, k),
                                           good_w[i][j][k]));
            }
        }
    }
    
}

int main(int argc, char ** argv)
try
{
    test();
}
catch (const std::exception & exc)
{
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
