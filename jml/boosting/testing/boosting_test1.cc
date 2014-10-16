/* boosting_test1.cc
   Jeremy Barnes, 11 March 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Tests feature identification for boosting.
*/

#include "jml/boosting/training_data.h"
#include "jml/boosting/sparse_features.h"
#include "jml/boosting/boosted_stumps.h"

using namespace ML;

using namespace std;


void run_boosting(const Training_Data & training_set, vector<Feature> features)
{
    unsigned nl = training_set.label_count();
    std::shared_ptr<const Feature_Space> fs = training_set.feature_space();
    Boosted_Stumps stumps(fs, nl);
    Boosted_Stumps best(fs, nl);
    
    Training_Params params;
    //params["trace"] = 1;

    boost::multi_array<float, 2> training_output(training_set.example_count(), nl);
    training_output.fill(0.0);

    boost::multi_array<float, 2> weights = Boosted_Stumps::get_weights(training_set);
    
    cerr << "  it   train     val   Z    true false  miss     arg feature"
         << endl;
    
    float train_acc = 0.0;
    float best_acc = 0.0;
    int best_iter = 0;

    //vector<Feature> features = training_set.all_features();
    
    for (unsigned i = 0;  i < 100;  ++i) {

        Stump stump = stumps.train_iteration(training_set, params, weights,
                                             features);

        stumps.update_scores(training_output, training_set, stump);
        
        train_acc = stumps.accuracy(training_output, training_set);
        
        cerr << format("%4d %6.2f%% %6.2f%% %5.3f",
                       i, train_acc * 100.0, train_acc * 100.0,
                       stump.Z);
        cerr << format(" %5.2f %5.2f", stump.pred_true[1],
                       stump.pred_false[1]);
        if (abs(stump.pred_missing[1]) > 0.01)
            cerr << format(" %5.2f", stump.pred_missing[1]);
        else cerr << "      ";
        cerr << format("%8.4g %-16s", stump.arg,
                       fs
                       ->print(stump.feature).c_str());
        cerr << endl;

        if (train_acc > best_acc) {
            best = stumps;
            best_acc = train_acc;
            best_iter = i;
        }
    }
    
    cerr << format("best was %6.2f%% on iteration %d", best_acc * 100.0,
                   best_iter)
         << endl;

    cerr << "best accuracy = " << best.accuracy(training_set) << endl;
}


int main(int argc, char ** argv)
try
{
    string dataset
        = "algorithms/machine_learning/boosting/datasets/conf-train-2.txt";
    
    std::shared_ptr<Sparse_Feature_Space>
        feature_space(new Sparse_Feature_Space());
    Sparse_Training_Data data(dataset, feature_space);

    /* Use some features we already know are bad. */
    vector<Feature> features = data.all_features();

#if 0
    features.push_back(feature_space->get_feature("hypernym-artifact/N1"));
    features.push_back(feature_space->get_feature
                       ("concept-sumo_SubjectiveAssessmentAttribute"));
    features.push_back(feature_space->get_feature("lemma-treatment"));
#endif

    run_boosting(data, features);

#if 0
    Boosted_Stumps stumps(feature_space, data.label_count());
    Training_Params params;

    stumps.train(data, params);
#endif
}
catch (const std::exception & exc)
{
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
