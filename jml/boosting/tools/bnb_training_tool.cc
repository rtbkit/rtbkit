/* boosting_training_tool.cc                                       -*- C++ -*-
   Jeremy Barnes, 12 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Tool to use to train boosting with.
*/

#include "jml/boosting/boosted_stumps.h"
#include "jml/boosting/naive_bayes.h"

#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/sparse_features.h"
#include "jml/boosting/probabilizer.h"
#include "jml/boosting/decoded_classifier.h"
#include "jml/boosting/boosting_tool_common.h"
#include "jml/boosting/weighted_training.h"
#include "jml/boosting/training_index.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/command_line.h"
#include <boost/progress.hpp>
#include <boost/timer.hpp>
#include "jml/stats/moments.h"
#include "datasets.h"

#include <iterator>
#include <iostream>
#include <set>


using namespace std;

using namespace ML;
using namespace Math;
using namespace Stats;

#if 0
// Stump representation
    Feature feature;                   ///< which feature to operate on?
    Feature_Type type;           ///< what type of feature is it?
    float arg;                         ///< p ::=  x > arg
    float Z;                           ///< Z score of the rule
    distribution<float> pred_true;     ///< predictions if p holds
    distribution<float> pred_false;    ///< predictions if p doesn't hold
    distribution<float> pred_missing;  ///< predictions if feature absent

// Boosted stumps representation
    typedef std::map<std::pair<Feature, float>, Stump> stumps_type;
    stumps_type stumps;

// Naive bayes representation
    /** This structure holds information about a feature used by Naive Bayes
        to turn a (feature, value) pair into a predicate.
    */
    struct Bayes_Feature {
        Bayes_Feature(const Feature & feature = Feature(), float arg = 0)
            : feature(feature), arg(arg)
        {
        }

        Feature feature;    ///< Feature to split on
        float arg;          ///< Value to split on

        bool operator < (const Bayes_Feature & other) const
        {
            if (feature < other.feature) return true;
            else if (feature == other.feature && arg < other.arg) return true;
            else return false;
        }
    };

    std::vector<Bayes_Feature> features;
    boost::multi_array<float, 3> probs;  /* num features x 3 x num labels matrix */
    distribution<float> label_priors;
    distribution<float> missing_total; /* sum of all missing distributions. */
#endif

Boosted_Stumps
bayes_to_boost(const Naive_Bayes & bayes)
{
    Boosted_Stumps result(bayes.feature_space(), bayes.label_count());
    int nl = bayes.label_count();
    static const int MISSING=2;

    cerr << "nl = " << nl << " bayes.features.size() = "
         << bayes.features.size() << " bayes.shape()[0] = " << bayes.probs.shape()[0]
         << "  bayes.shape()[1] = " << bayes.probs.shape()[1] << " bayes.dim(2) = "
         << bayes.probs.dim(2) << endl;

    for (unsigned f = 0;  f < bayes.features.size();  ++f) {
        Stump this_stump(bayes.feature_space(), nl);
        this_stump.feature = bayes.features[f].feature;
        this_stump.arg     = bayes.features[f].arg;
        this_stump.pred_false
            = distribution<float>(&bayes.probs[f][false][0],
                                  &bayes.probs[f][false][0] + nl);
        
        this_stump.pred_true
            = distribution<float>(&bayes.probs[f][true][0],
                                  &bayes.probs[f][true][0] + nl);
        this_stump.pred_missing
            = distribution<float>(&bayes.probs[f][MISSING][0],
                                  &bayes.probs[f][MISSING][0] + nl);
        
        result.insert(this_stump);
    }

    result.bias = log(bayes.label_priors);

    return result;
}

Boosted_Stumps
run_boosting(const Training_Data & training_set,
             const Training_Data & validation_set_,
             unsigned max_iter, unsigned min_iter, int verbosity,
             double equalize_beta, vector<Feature> features,
             const vector<Feature> & equalize_features,
             const vector<float> & equalize_betas, bool true_only,
             bool profile, bool fair,
             float feature_prop, int committee_size, int trace,
             int cost_function, int output_function, int update_alg,
             float ignore_highest, float sample_prop, int weight_function,
             const vector<Classifier_Impl::Weight_Spec> & weight_spec)
{
    boost::timer timer;
    unsigned nl = training_set.label_count();
    std::shared_ptr<const Feature_Space> feature_space
        = training_set.feature_space();

    Boosted_Stumps stumps(feature_space, nl);
    Boosted_Stumps best(feature_space, nl);
    float best_acc = 0.0;
    int best_iter = 0;
    
    bool validate_is_train = false;
    if (validation_set_.example_count() == 0)
        validate_is_train = true;

    const Training_Data & validation_set
        = (validate_is_train ? training_set : validation_set_);

    //cerr << "true_only = " << true_only << endl;
    //cerr << "trace = " << trace << endl;
    Training_Params params;
    params["true_only"] = true_only;
    params["rule_proportion"] = feature_prop;
    params["sample_proportion"] = sample_prop;
    if (trace) params["trace"] = trace;
    params["cost_function"] = Boosted_Stumps::Cost_Function(cost_function);
    stumps.output = Boosted_Stumps::Output(output_function);
    params["update_alg"] = update_alg;
    params["ignore_highest"] = ignore_highest;
    params["weight_function"] = weight_function;

    boost::multi_array<float, 2> training_output(training_set.example_count(), nl);
    training_output.fill(0.0);

    boost::multi_array<float, 2> validation_output(validation_set.example_count(),
                                            nl);
    validation_output.fill(0.0);
    
    distribution<float> training_ex_weights
        = Classifier_Impl::apply_weight_spec(training_set, weight_spec);
    distribution<float> validate_ex_weights
        = Classifier_Impl::apply_weight_spec(validation_set, weight_spec);

    boost::multi_array<float, 2> weights
        = Classifier_Impl::expand_weights(training_set, training_ex_weights);

    float validate_acc = 0.0;
    float train_acc = 0.0;

#if 1
    cerr << "training bayes... ";
    Naive_Bayes bayes(training_set.feature_space(), nl);
    Training_Params bayes_params;
    if (trace) bayes_params["trace"] = trace;
    bayes_params["features"] = features;
    bayes.train_weighted(training_set, bayes_params, weights);
    cerr << "done." << endl;
    
    train_acc = bayes.accuracy(training_set);
    validate_acc = bayes.accuracy(validation_set);
    
    cerr << "bayes accuracy: " << train_acc << " training "
         << validate_acc << " validation" << endl;

#if 0
    cerr << "converting to boosted stumps... ";
    stumps = bayes_to_boost(bayes);
    cerr << "done." << endl;

    cerr << "updating weights... ";
    for (Boosted_Stumps::const_iterator it = stumps.begin();
         it != stumps.end();  ++it) {
        stumps.update_scores(training_output, training_set, *it);
        stumps.update_scores(validation_output, validation_set, *it);
    }
    cerr << "done." << endl;
#endif
#endif

#if 1
    distribution<float> bias(nl, 0.0);
    for (unsigned i = 0;  i < 5;  ++i) {
        Stump stump = Stump::get_bias(training_set, params, weights);
        bias += stump.pred_missing;
        stumps.update_weights(weights, stump, training_set, params);
        stumps.update_scores(training_output, training_set, stump);
        stumps.update_scores(validation_output, validation_set, stump);
    }

    cerr << "bias = " << bias << endl;
    stumps.bias = bias;
#endif

    boost::scoped_ptr<boost::progress_display> progress;

    if (verbosity <= 5) {
        cerr << "training " << features.size() << " features..." << endl;
        progress.reset(new boost::progress_display(features.size(), cerr));
    }

#if 1
    boost::multi_array<float, 2> start_weights = weights;

    for (unsigned i = 0;  i < features.size();  ++i) {
        vector<Feature> f(1, features[i]);
        Stump stump(feature_space, nl);
        Training_Params params2 = params;
        params2["features"] = f;

        stump.train_weighted(training_set, params2, start_weights);
        stumps.insert(stump);

        stumps.update_scores(training_output, training_set, stump);
        stumps.update_scores(validation_output, validation_set, stump);

        train_acc = stumps.accuracy(training_output, training_set);
        if (validate_is_train) validate_acc = train_acc;
        else validate_acc
                 = stumps.accuracy(validation_output, validation_set);
        
        if (verbosity > 5) {
            cerr << format("%4d %6.2f%% %6.2f%% %5.3f ",
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
            else if (verbosity > 6)
                for (unsigned i = 0;  i < nl;  ++i)
                    cerr << format("%6.2f", stump.pred_true[i]);
            
            cerr << format("%9.4g %-16s", stump.arg,
                           training_set.feature_space()
                           ->print(stump.feature).c_str());
            cerr << endl;
        }
        else ++(*progress);
    }

    cerr << "after bayesian part: train_acc = " << train_acc
         << " validate_acc = " << validate_acc << endl;
#endif

    if (verbosity == 1 || verbosity == 2) {
        cerr << "training " << max_iter << " iterations..." << endl;
        progress.reset(new boost::progress_display(max_iter, cerr));
    }
    
    if (min_iter > max_iter)
        throw Exception("min_iter is greater than max_iter");

    if (verbosity > 2) {
        cerr << "  it   train     val   Z    ";
        if (nl == 2) cerr << "true false  miss";
        else if (nl <= 5)
            for (unsigned i = 0;  i < std::min(nl, 5U);  ++i)
                cerr << format("lbl%d  ", i);
        else if (verbosity > 3)
            for (unsigned i = 0;  i < nl;  ++i)
                cerr << format("lbl%-3d", i);
        cerr << "       arg feature" << endl;
    }
    
    for (unsigned i = 0;  i < max_iter;  ++i) {
        if (progress) ++(*progress);
        //vector<ML::Feature> features;


        if (!fair) {
#if 0
            vector<Stump> all_stumps
                = stumps.train_iteration(training_set, params, weights,
                                         features, committee_size);
            stumps.update_scores(training_output, training_set, all_stumps);
            stumps.update_scores(validation_output, validation_set,all_stumps);
            
            train_acc = stumps.accuracy(training_output, training_set);
            validate_acc
                = stumps.accuracy(validation_output, validation_set);
            
            if (verbosity > 2) {
                cerr << format("%4d %6.2f%% %6.2f%% ",
                               i, train_acc * 100.0, validate_acc * 100.0);
                for (unsigned i = 0;  i < all_stumps.size();  ++i) {
                    if (i != 0) cerr << "                     ";
                    cerr << format("%6.4f", all_stumps[i].Z);
                    const Stump & stump = all_stumps[i];
                    if (stump.label_count() == 2
                        && (abs(stump.pred_true[0]+stump.pred_true[1])
                            < 0.001)) {
                        cerr << format(" %5.2f %5.2f", stump.pred_true[1],
                                       stump.pred_false[1]);
                        if (abs(stump.pred_missing[1]) > 0.01)
                            cerr << format(" %5.2f", stump.pred_missing[1]);
                        else cerr << "      ";
                    }
                    else {
                        if (nl <= 5 || verbosity > 3) 
                            for (unsigned l = 0;  l < stump.pred_true.size();
                                 ++l)
                                cerr << format(" %5.2f", stump.pred_true[l]);
                    }
                    cerr << format("%8.4f %s", stump.arg,
                                   training_set.feature_space()
                                   ->print(stump.feature).c_str());
                    if (i != all_stumps.size() - 1) cerr << ",";
                    cerr << endl;
                }
            }

#else
            Stump stump;
            stump = stumps
                .train_iteration(training_set, params, weights, features);

            stumps.update_scores(training_output, training_set, stump);
            stumps.update_scores(validation_output, validation_set, stump);
            
            train_acc = stumps.accuracy(training_output, training_set);
            if (validate_is_train) validate_acc = train_acc;
            else validate_acc
                     = stumps.accuracy(validation_output, validation_set);

#if 0            
            cerr << "train_acc = " << train_acc << "  validate_acc = "
                 << validate_acc << endl;

            train_acc = stumps.accuracy(training_set);
            validate_acc = stumps.accuracy(validation_set);

            cerr << "train_acc = " << train_acc << "  validate_acc = "
                 << validate_acc << endl;
#endif

            if (verbosity > 2) {
                cerr << format("%4d %6.2f%% %6.2f%% %5.3f ",
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

                cerr << format("%9.4g %-16s", stump.arg,
                               training_set.feature_space()
                                   ->print(stump.feature).c_str());
                cerr << endl;
            }
#endif
        }
        else {
            vector<Stump> all_stumps
                = stumps.train_iteration_fair(training_set, params, weights,
                                              features, committee_size);
            stumps.update_scores(training_output, training_set, all_stumps);
            stumps.update_scores(validation_output, validation_set,all_stumps);
            
            float train_acc = stumps.accuracy(training_output, training_set);
            validate_acc
                = stumps.accuracy(validation_output, validation_set);
            
            if (verbosity > 2) {
                cerr << format("%4d %6.2f%% %6.2f%% ",
                               i, train_acc * 100.0, validate_acc * 100.0);
                for (unsigned i = 0;  i < all_stumps.size();  ++i) {
                    if (i != 0) cerr << "                     ";
                    cerr << format("%6.4f", all_stumps[i].Z);
                    const Stump & stump = all_stumps[i];
                    if (stump.label_count() == 2
                        && (abs(stump.pred_true[0]+stump.pred_true[1])
                            < 0.001)) {
                        cerr << format(" %5.2f %5.2f", stump.pred_true[1],
                                       stump.pred_false[1]);
                        if (abs(stump.pred_missing[1]) > 0.01)
                            cerr << format(" %5.2f", stump.pred_missing[1]);
                        else cerr << "      ";
                    }
                    else {
                        if (nl <= 5 || verbosity > 3) 
                            for (unsigned l = 0;  l < stump.pred_true.size();
                                 ++l)
                                cerr << format(" %5.2f", stump.pred_true[l]);
                        if (verbosity > 4) {
                            cerr << endl;
                            cerr << "                           ";
                            for (unsigned l = 0;  l < stump.pred_true.size();
                                 ++l)
                                cerr << format(" %5.2f", stump.pred_false[l]);
                            cerr << endl;
                            cerr << "                           ";
                            for (unsigned l = 0;  l < stump.pred_true.size();
                                 ++l)
                                cerr << format(" %5.2f", stump.pred_missing[l]);
                        }
                    }
                    cerr << format("%8.4f %s", stump.arg,
                                   training_set.feature_space()
                                   ->print(stump.feature).c_str());
                    if (i != all_stumps.size() - 1) cerr << ",";
                    cerr << endl;
                }
            }
        }
        
        if (validate_acc > best_acc && i >= min_iter) {
            best = stumps;
            best_acc = validate_acc;
            best_iter = i;
        }
    }

    if (profile)
        cerr << "training time: " << timer.elapsed() << "s" << endl;
    
    if (verbosity > 0) {
        cerr << format("best was %6.2f%% on iteration %d", best_acc * 100.0,
                       best_iter)
             << endl;
    }
    
    cerr << "best accuracy = " << best.accuracy(validation_set) << endl;

    best.calc_sum_missing();

    return best;
}

int main(int argc, char ** argv)
try
{
    ios::sync_with_stdio(false);

    int max_iter            = 1000;
    int min_iter            = 5;
    float validation_split  = 0.0;
    float testing_split     = 0.0;
    bool randomize_order    = false;
    int probabilize_mode    = 1;
    int probabilize_data    = 0;
    bool probabilize_weighted = false;
    bool variable_header    = false;
    bool data_is_sparse     = false;
    int label_count         = -1;
    float equalize_beta     = 0.0;
    string weight_spec      = "";
    int draw_graphs         = false;
    //Stop_At stop_at        = BEST_RECOG;
    string scale_output;
    string scale_input;
    string output_file;
    string variable_name_file;
    string equalize_name = "label";
    int verbosity           = 1;
    vector<string> ignore_features;
    vector<string> optional_features;
    int cross_validate      = 1;
    int repeat_trials       = 1;
    bool val_with_test      = false;
    bool dump_testing       = false;
    bool true_only          = false;
    int min_feature_count   = 1;
    bool remove_aliased     = false;
    bool profile            = false;
    bool fair               = false;
    int committee_size      = 1;
    float feature_prop      = 1.0;
    int print_confusion     = false;
    int trace               = 0;
    string probabilize_link = "logit";
    int cost_function       = 0;
    int output_function     = 0;
    int update_alg          = 0;
    int weight_function     = 0;
    float ignore_highest    = 0.0;
    float sample_prop       = 1.0;
    bool is_regression      = false;  // Set to be a regression?
    bool is_regression_set  = false;  // Value of is_regression has been set?
    int num_buckets         = Training_Data::DEFAULT_NUM_BUCKETS;
    string group_feature_name = "";
    bool eval_by_group      = false;  // Evaluate a group at a time?
    bool eval_by_group_set  = false;  // Value of eval_by_group has been set?

    vector<string> extra;
    {
        using namespace CmdLine;

        static const Option data_options[] = {
            { "validation-split", 'V', validation_split, validation_split,
              false, "split X% of training data for validation", "0-100" },
            { "testing-split", 'T', testing_split, testing_split,
              false, "split X% of training data for testing", "0-100" },
            { "validate-with-testing", 0, NO_ARG, flag(val_with_test),
              false, "(CHEAT) use testing data for validation" },
            { "randomize-order", 'R', NO_ARG, flag(randomize_order),
              false, "randomize the order of data before splitting" },
            { "variable-names", 0, variable_name_file, variable_name_file,
              false, "read variables names from FILE", "FILE" },
            { "variable-header", 'd', NO_ARG, flag(variable_header),
              false, "use first line of file as a variable header" },
            { "label-count", 'L', label_count, label_count,
              false, "force number of labels to be N", "N" },
            { "ignore-var", 'z', string(), push_back(ignore_features),
              false,"ignore variable with name matching REGEX","REGEX|@FILE" },
            { "optional-feature", 'O', string(), push_back(optional_features),
              false, "feature with name REGEX is optional", "REGEX|@FILE" },
            { "sparse-data", 'S', NO_ARG, flag(data_is_sparse),
              false, "dataset is in sparse format" },
            { "min-feature-count", 'c', min_feature_count, min_feature_count,
              true, "don't consider features seen < NUM times", "NUM" },
            { "remove-aliased", 'A', NO_ARG, flag(remove_aliased),
              false, "remove aliased training rows from the training data" },
            { "regression", 'N', NO_ARG, flag(is_regression,is_regression_set),
              false, "force dataset to be a regression problem" },
            { "num-buckets", '1', num_buckets, num_buckets,
              true, "number of buckets to discretize into (0=off)", "INT" },
            { "group-feature", 'g', group_feature_name, group_feature_name,
              false, "use FEATURE to group examples in dataset", "FEATURE" },
            Last_Option
        };
        
        static const Option training_options[] = {
            { "max-iterations", 'm', max_iter, max_iter,
              true, "run no more than NUM iterations", "NUM" },
            { "min-iterations", 'n', min_iter, min_iter,
              true, "output must be trained on at least N iterations", "NUM" },
            { "cross-validate", 'X', cross_validate, cross_validate,
              false, "run N-fold cross validation", "INT" },
            { "repeat-trials", 'r', repeat_trials, repeat_trials,
              false, "repeat experiment N times", "INT" },
            { "true-only", '9', NO_ARG, flag(true_only),
              false, "don't false or missing predicates to infer labels" },
            { "fair", 'f', NO_ARG, flag(fair),
              false, "treat mutiple equal features in a fair manner" },
            { "feature-proportion", '3', feature_prop, feature_prop,
              false, "lazy train on this proportion of features", "0.0-1.0" },
            { "committee-size", '4', committee_size, committee_size,
              true, "(fair=1 only): learn committees of N classifiers", "INT"},
            { "cost-function", '5', cost_function, cost_function,
              true, "cost function: 0 = boosting, 1 = logitboost", "0-1" },
            { "output-function", '6', output_function, output_function,
              true, "output fn: 0 = raw, 1 = prob, 2 = probnorm", "0-2" },
            { "update-alg", '7', update_alg, update_alg,
              true, "update alg: 0 = boosting, 1 = gentle", "0-1" },
            { "ignore-highest", '8', ignore_highest, ignore_highest,
              true, "ignore highest x proportion of weights", "0.0-1.0" },
            { "sample-proportion", '9', sample_prop, sample_prop,
              true, "use only X proportion of samples each iter", "0.0-1.0" },
            { "weight-function", 'a', weight_function, weight_function,
              true, "weight function: 0 = normal, 1 = conf", "0,1" },
            Last_Option
        };

        static const Option weight_options[] = {
            { "equalize-beta", 'E', equalize_beta, equalize_beta,
              true, "equalize labels (b in w=freq^(-b); 0.0=off)", "0.0-1.0" },
            { "equalize-feature", 'F', equalize_name, equalize_name,
              true, "equalize based upon feature FEATURE","FEATURE||'label'" },
            { "weight-spec", 'W', weight_spec, weight_spec,
              false, "use SPEC for weights", "VAR1(BETA1),VAR2(BETA2)..." },
            Last_Option
        };

        static const Option probabilize_options[] = {
            { "probabilize-mode", 'p', probabilize_mode, probabilize_mode,
              true, "prob mode: 0=matrix, 1=pervar, 2=oneonly, 3=off", "0-3" },
            { "probabilize-link", 'K', probabilize_link, probabilize_link,
              true, "prob link function", "logit|log" },
            { "probabilize-data", 'P', probabilize_data, probabilize_data,
              true, "data for prob: 0 = train, 1 = validate", "0|1" },
            { "probabilize-weighted", 'Q', NO_ARG, flag(probabilize_weighted),
              false, "train probabilizer using weights also" },
            Last_Option
        };

        static const Option output_options[] = {
            { "output-file", 'o', output_file, output_file,
              false, "write output network to FILE", "FILE" },
            { "quiet", 'q', NO_ARG, assign(verbosity, 0),
              false, "don't write any non-fatal output" },
            { "verbosity", 'v', optional(2), verbosity,
              false, "set verbosity to LEVEL (0-3)", "LEVEL" },
            { "profile", 'l', NO_ARG, flag(profile),
              false, "profile execution time" },
            { "draw-graphs", 'G', NO_ARG, increment(draw_graphs),
              false, "draw graphs for two-class predictor" },
            { "dump-testing", 'D', NO_ARG, flag(dump_testing),
              false, "dump output of classifier on testing sets" },
            { "print-confusion", 'C', NO_ARG, increment(print_confusion),
              false, "print confusion matrix" },
            { "trace", 't', trace, trace,
              false, "set trace level to LEVEL (0 off)", "LEVEL" },
            { "eval-by-group", 0, NO_ARG,flag(eval_by_group,eval_by_group_set),
              false, "evaluate by group rather than by example" },
            Last_Option
        };

        static const Option options[] = {
            group("Data options", data_options),
            group("Weight options", weight_options),
            group("Training options", training_options),
            group("Probabilizer options", probabilize_options),
            group("Output options",   output_options),
            Help_Options,
            Last_Option };

        Command_Line_Parser parser("boosting_training_tool", argc, argv,
                                   options);
        
        bool res = parser.parse();
        if (res == false) exit(1);
        
        extra.insert(extra.end(), parser.extra_begin(), parser.extra_end());
    }

    if (val_with_test && validation_split != 0.0)
        throw Exception("can't validate with testing and validation data");

    /* The first extra argument is the training set.  The second, if it
       exists, is the validation set.  Any further ones are testing sets.

       We can also choose to split our training data according to a given
       percentage, for the case where we only have one set.
    */
    
    if (extra.empty()) {
        cerr << "error: need to specify (at least) training data" << endl;
        exit(1);
    }

    Datasets datasets;
    datasets.init(extra, verbosity, profile);

    std::shared_ptr<Feature_Space> feature_space = datasets.feature_space;

    vector<Feature> features;
    map<string, Feature> feature_index;

    boost::tie(features, feature_index)
        = do_features(*data[0], feature_space, ignore_features,
                      optional_features, min_feature_count, verbosity);

    if (group_feature_name != "") {
        if (!feature_index.count(group_feature_name))
            throw Exception("grouping feature " + group_feature_name
                            + " not found in data");
        Feature group_feature = feature_index[group_feature_name];
        data[0]->group_feature_ = group_feature;
        
        if (!eval_by_group_set) {
            if (verbosity > 0)
                cerr << "note: overriding eval-by-group=1, use "
                     << "--no-eval-by-group to avoid" << endl;
            eval_by_group = true;
        }
    }

    float training_split = 100.0 - validation_split - testing_split;
    datasets.split(training_split, validation_split, testing_split,
                   randomize_order, group_feature);

    /* Write a null classifier, if there is no training data or we can't
       train for some reason. */
    if (training->example_count() == 0
        || (label_count <= 1 && !regression_problem)) {
        write_null_classifier(output_file, label_count, verbosity);
        return 0;
    }

    vector<Feature> equalize_features;
    vector<float> equalize_betas;

    boost::tie(equalize_betas, equalize_features)
        = parse_weight_spec(feature_index, equalize_name, equalize_beta,
                            weight_spec);

    vector<Classifier_Impl::Weight_Spec> trained_weight_spec
        = Classifier_Impl::get_weight_spec(*training, equalize_betas,
                                           equalize_features);

    /* Now for the training. */
    
    vector<distribution<float> > accum_acc(testing.size());

    for (unsigned i = 0;  i < repeat_trials;  ++i) {
        if (repeat_trials > 1) cerr << "trial " << (i + 1) << endl;

        if (!training->finished()) {
            boost::timer timer;
            training->finish(num_buckets);
            if (profile)
                cerr << "[finish training data: " << timer.elapsed() << "s]"
                     << endl;
        }
        
        if (verbosity > 0) {
            unsigned testing_rows = 0;
            for (unsigned i = 0;  i < testing.size();  ++i) {
                testing_rows += testing[i]->example_count();
            }
            
            cerr << "data sizes: training " << training->example_count()
                 << "  validation: " << validation->example_count()
                 << "  testing: " << testing_rows << endl;
        }
        
        if (remove_aliased)
            remove_aliased_examples(*training, verbosity, profile);
        
        vector<Feature> equalize_features;
        vector<float> equalize_betas;
        
        boost::tie(equalize_betas, equalize_features)
            = parse_weight_spec(feature_index, equalize_name, equalize_beta,
                                weight_spec);
        
        Boosted_Stumps current
            = run_boosting(*training, *validation, max_iter, min_iter,
                           verbosity, equalize_beta, features,
                           equalize_features, equalize_betas,
                           true_only,
                           profile, fair, feature_prop, committee_size, trace,
                           cost_function, output_function, update_alg,
                           ignore_highest, sample_prop, weight_function,
                           trained_weight_spec);
        GLZ_Probabilizer prob;
        Training_Params params;
        params["glz_probabilizer_mode"] = probabilize_mode;
        params["glz_probabilizer_link"] = probabilize_link;

        if (probabilize_data < 0 || probabilize_data > 1)
            throw Exception("probabilize-data must be 0 or 1 (currently "
                            + ostream_format(probabilize_data) + ")");
        std::shared_ptr<const Training_Data> prob_set
            = (probabilize_data == 0 ? training : validation);
        
        distribution<float> pr_weights(prob_set->example_count(), 1.0);

        if (probabilize_weighted)
            pr_weights
                = Classifier_Impl::
                    apply_weight_spec(*prob_set, trained_weight_spec);
        
        prob.train(*prob_set, params, current, pr_weights);

        if (repeat_trials == 1 || verbosity > 2) {
            cerr << "Stats over training set: " << endl;
            calc_stats(current, prob, *training, draw_graphs, dump_testing,
                       print_confusion, eval_by_group);
            
            cerr << "Stats over validation set: " << endl;
            calc_stats(current, prob, *validation, draw_graphs, dump_testing,
                       print_confusion, eval_by_group);
        }

        Classifier output_wrapped(current);
        Decoded_Classifier output(output_wrapped, Decoder(prob));
        Classifier cl(output);
        
        if (output_file != "") {
            if (verbosity > 0)
                cerr << "writing to \'" << output_file << "\'... ";
            cl.save(output_file);
            if (verbosity > 0) cerr << "done." << endl;
        }
        
        /* Test all of the testing datasets. */
        for (unsigned j = 0;  j < testing.size();  ++j) {
            cerr << "Stats over testing set " << j << ":" << endl;
            
            calc_stats(current, prob, *testing[j], draw_graphs,
                       dump_testing, print_confusion, eval_by_group);
            
            if (repeat_trials > 1) {
                float acc = current.accuracy(*testing[j]);
                accum_acc[j].push_back(acc);
                cerr << "trial " << i + 1 << ": accuracy " << acc * 100.0
                     << "%, average over all trials = "
                     << accum_acc[j].total() * 100.0 / (i + 1)
                     << "%" << endl;
            }
        }
        
        if (i != repeat_trials - 1) {
            /* Re-shuffle the data. */
            datasets.resuffle();
        }
    }

    if (repeat_trials > 1) {
        for (unsigned j = 0;  j < accum_acc.size();  ++j) {
            float mean = Stats::mean(accum_acc[j].begin(), accum_acc[j].end(),
                                     0.0);
            float std_dev
                = Stats::std_dev(accum_acc[j].begin(), accum_acc[j].end(),
                                 mean);
            cerr << "testing set " << j << ": " << repeat_trials
                 << " trials accuracy: mean "
                 << mean * 100.0 << "% std " << std_dev * 100.0 << "%."
                 << endl;
        }
    }
}
catch (const std::exception & exc) {
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
