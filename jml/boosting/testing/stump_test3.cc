/* stump_test2.cc
   Jeremy Barnes, 11 March February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Test program for the stump code.  We check that presence features are
   working properly.

   cat wsd-word-train-make-VERB-15-2-2.txt | tr '\n' '@' | sed 's/@/ @ /' | tr ' ' '\n' | grep -v 'concept\|lemma\|hypernym\|coarse\|fine\|surface' | tr '\n' ' ' | sed 's/ @ /@/' | tr '@' '\n' > algorithms/machine_learning/boosting/datasets/presence-train-1.txt
*/

#include "stump_testing.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/sparse_features.h"
#include "jml/boosting/stump.h"



using namespace ML;
using namespace std;


int main(int argc, char ** argv)
try
{
    string dataset
        = "algorithms/machine_learning/boosting/datasets/presence-train-1.txt";
    
    std::shared_ptr<Sparse_Feature_Space>
        feature_space(new Sparse_Feature_Space());

    Sparse_Training_Data data(dataset, feature_space);

    vector<Feature> features = data.all_features();

    cerr << "testing " << features.size() << " features..." << endl;
    for (unsigned i = 0;  i < features.size();  ++i) {
        cerr << "feature " << i << " (" << feature_space->print(features[i])
             << ")... ";

        vector<Feature> current_feature(1, features[i]);
        Training_Params params;
        params["features"] = current_feature;

        boost::multi_array<float, 2> weights2(data.example_count(), 2);
        weights2.fill(1.0 / (float)(data.example_count() * 2));
        boost::multi_array<float, 2> weights3(data.example_count(), 3);
        weights3.fill(1.0 / (float)(data.example_count() * 2));
        for (unsigned i = 0;  i < data.example_count();  ++i)
            weights3[i][2] = 0.0;

        Stump stump2(feature_space, 2);
        Stump stump3(feature_space, 3);

        Stump trained2 = stump2.train_all(data, params, weights2).at(0);
        Stump trained3 = stump3.train_all(data3, params, weights3).at(0);

        if (trained2.Z != trained3.Z)
            throw Exception(format("Z values differ: %f vs %f",
                                   trained2.Z, trained3.Z));
        
        cerr << "done." << endl;
    }
    
    return 0;
}
catch (const std::exception & exc)
{
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
