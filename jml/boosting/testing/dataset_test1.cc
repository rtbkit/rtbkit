/* boosting_test1.cc
   Jeremy Barnes, 11 March 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Tests feature identification for boosting.
*/

#include "jml/boosting/training_data.h"
#include "jml/boosting/sparse_features.h"

using namespace ML;
using namespace std;


int main(int argc, char ** argv)
try
{
    string dataset1
        = "algorithms/machine_learning/boosting/datasets/gin.txt";
    string dataset2
        = "algorithms/machine_learning/boosting/datasets/gin-test.txt";
    int label_count = 49;

    /* Variables for when we are sparse. */
    std::shared_ptr<Sparse_Feature_Space>
        feature_space(new Sparse_Feature_Space());
    Sparse_Training_Data data1(dataset1, feature_space, label_count);
    Sparse_Training_Data data2(dataset2, feature_space, label_count);

    /* Find the feature for surface-gin */
    Feature make_feature = feature_space->get_feature("surface-make");
    if (feature_space->info(make_feature) == INUTILE)
        throw Exception("feature surface-make is not meant to be INUTILE");

    /* Find the feature for surface-make */
    Feature gin_feature = feature_space->get_feature("lemma-gin");
    if (feature_space->info(make_feature) != REAL)
        throw Exception("feature lemma-gin is meant to be REAL");
    
    return 0;
}
catch (const std::exception & exc)
{
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
