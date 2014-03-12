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
    {
        string dataset
            = "algorithms/machine_learning/boosting/datasets/presence-train-1.txt";
        
        /* Variables for when we are sparse. */
        std::shared_ptr<Sparse_Feature_Space>
            feature_space(new Sparse_Feature_Space());
        Sparse_Training_Data data(dataset, feature_space);
        
        /* Find the feature for surface-gin */
        Feature chord_feature = feature_space->get_feature("surf_p2-chord");
        if (feature_space->info(chord_feature) != PRESENCE)
            throw Exception("feature surf_p2-chord is meant to be PRESENCE"
                            ", is " + ostream_format(feature_space
                                                     ->info(chord_feature)));
        
    }

    {
        string dataset
            = "algorithms/machine_learning/boosting/datasets/presence-3.txt";
        
        /* Variables for when we are sparse. */
        std::shared_ptr<Sparse_Feature_Space>
            feature_space(new Sparse_Feature_Space());
        Sparse_Training_Data data(dataset, feature_space);
        
        /* Find the feature for surface-gin */
        Feature barely_feature = feature_space->get_feature("lemma-barely");
        if (feature_space->info(barely_feature) != REAL)
            throw Exception("feature lemma-barely is meant to be REAL"
                            ", is " + ostream_format(feature_space
                                                     ->info(barely_feature)));
        
    }
    
    return 0;
}
catch (const std::exception & exc)
{
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
