/* boosted_stump_test1.cc                                          -*- C++ -*-
   Jeremy Barnes, 4 March 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Test of the boosted stumps over a given dataset.  It runs various iterations
   over various datasets, all the time checking that the invariants hold.
*/
#include "jml/boosting/stump_training.h"
#include "jml/boosting/stump_training_core.h"
#include "jml/boosting/stump_training_bin.h"
#include "stump_testing.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"


using namespace ML;
using namespace std;


void run_boosting(const Training_Data & data)
{
    unsigned nl = data.label_count();
    vector<Feature> features = data.all_features();
    Boosted_Stumps stumps(data.feature_space(), nl);

    Training_Params params;

    boost::multi_array<float, 2> training_output(data.example_count(), nl);
    training_output.fill(0.0);

    boost::multi_array<float, 2> weights
        = Boosted_Stumps::get_weights(data);
    
    boost::scoped_ptr<boost::progress_display> progress;

    cerr << "training " << max_iter << " iterations..." << endl;
    progress.reset(new boost::progress_display(max_iter));
    
    float train_acc = 0.0;

    bool bin_sym = check_bin_sym(weights, label_count());

    for (unsigned i = 0;  i < max_iter;  ++i) {
        if (progress) ++(*progress);
        
        /* Find the best stump */
        Stump stump(feature_space(), label_count());
        
        Training_Params stump_params = params;
        
        stump.train_weighted(data, stump_params, weights);
        
        /* Insert it */
        insert(stump);
        
        /* Update the d distribution. */
        double total = 0.0;
        
        if (bin_sym) {
            typedef Boosting_Loss Loss;
            typedef Update_Weights<Loss> Update;
            Update update;
            
            total = update(stump, weights, data);
        }
        else {
            typedef Boosting_Loss Loss;
            typedef Update_Weights<Loss> Update;
            Update update;
            
            total = update(stump, weights, data);
        }
        
        float * start = weights.data_begin();
        float * finish = weights.data_end();
        Math::simd_vec_scale(start, 1.0 / total, start, finish - start);






        stumps.update_scores(training_output, data, stump);
        
        train_acc = stumps.accuracy(training_output, data);
        
        cerr << format("%4d %6.2f%% %6.2f%% %5.3f",
                       i, train_acc * 100.0, validate_acc * 100.0,
                       stump.Z);
        if (stump.label_count() == 2
            && (abs(stump.pred_true[0]+stump.pred_true[1]) < 0.001)) {
            cerr << format(" %5.2f %5.2f", stump.pred_true[1],
                           stump.pred_false[1]);
            if (abs(stump.pred_missing[1]) > 0.01)
                cerr << format(" %5.2f", stump.pred_missing[1]);
            else cerr << "      ";
        }
        else if (nl <= 5)
            for (unsigned i = 0;  i < std::min(nl, 5U);  ++i)
                cerr << format("%6.2f", stump.pred_true[i]);
        else if (verbosity > 3)
            for (unsigned i = 0;  i < nl;  ++i)
                cerr << format("%6.2f", stump.pred_true[i]);
        
        cerr << format("%8.4g %-16s", stump.arg,
                       data.feature_space()
                       ->print(stump.feature).c_str());
        cerr << endl;
    }

    return best;
}


int main(int argc, char ** argv)
try
{
    cerr << "loading dataset... ";
    string dataset
        = "algorithms/machine_learning/boosting/datasets/lso-conf.txt";
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

    typedef Stump_Accum<W, Z, C> Accum;
    typedef Stump_Trainer<W> Trainer;
        
    Accum accum(true, 0);
    Trainer trainer;

    /* Test with one dimensional weights. */
    cerr << "one dimensional..." << endl;

    boost::multi_array<float, 2> weights1(data->example_count(), 1);
    weights1.fill(1.0 / (float)(data->example_count() * 2));

    cerr << "default w... " << endl;
    W default_w = trainer.calc_default_w(*data, weights1);
    cerr << "done." << endl;
    
    for (unsigned i = 0;  i < features.size();  ++i) {
        cerr << "feature " << i << " (" << feature_space->print(features[i])
             << ")... ";
        trainer.test(features[i], *data, weights1, default_w, accum);
        cerr << "done." << endl;
    }

    
    /* Test with 2 dimensional weights. */
    cerr << "two dimensional..." << endl;
    boost::multi_array<float, 2> weights2(data->example_count(), 2);
    weights2.fill(1.0 / (float)(data->example_count() * 2));

    cerr << "default w... " << endl;
    default_w = trainer.calc_default_w(*data, weights2);
    cerr << "done." << endl;

    for (unsigned i = 0;  i < features.size();  ++i) {
        cerr << "feature " << i << " (" << feature_space->print(features[i])
             << ")... ";
        trainer.test(features[i], *data, weights2, default_w, accum);
        cerr << "done." << endl;
    }
    

    return 0;
}
catch (const std::exception & exc)
{
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
