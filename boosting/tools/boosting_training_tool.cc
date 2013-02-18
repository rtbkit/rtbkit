/* boosting_training_tool.cc                                       -*- C++ -*-
   Jeremy Barnes, 12 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Tool to use to train boosting with.
*/

#include "jml/boosting/boosted_stumps.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/sparse_features.h"
#include "jml/boosting/probabilizer.h"
#include "jml/boosting/decoded_classifier.h"
#include "jml/boosting/boosting_tool_common.h"
#include "jml/boosting/weighted_training.h"
#include "jml/boosting/training_index.h"
#include "jml/boosting/boosted_stumps_generator.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/command_line.h"
#include <boost/progress.hpp>
#include <boost/timer.hpp>
#include "jml/stats/moments.h"
#include "datasets.h"
#include "jml/utils/info.h"

#include <iterator>
#include <iostream>
#include <set>


using namespace std;

using namespace ML;
using namespace Math;
using namespace Stats;


void print_weights(const boost::multi_array<float, 2> & weights,
                   const Training_Data & training_set, int num_print_weights)
{
    if (num_print_weights == 0) return;

    int nx = weights.shape()[0];
    int nl = weights.shape()[1];

    vector<pair<int, float> > sorted(nx);
    for (unsigned x = 0;  x < nx;  ++x) {
        sorted[x].first = x;
        for (unsigned l = 0;  l < nl;  ++l) {
            sorted[x].second += weights[x][l];
        }
        sorted[x].second *= nx;
    }

    sort_on_second_descending(sorted);

    cerr << "Hardest examples: " << endl;
    for (unsigned i = 0;  i < num_print_weights;  ++i) {
        if (i >= sorted.size()) break;
        int ex = sorted[i].first;
        cerr << "  " << i << ": example " << ex << " with weight "
             << sorted[i].second << endl;
        cerr << training_set.feature_space()->print(training_set[ex]);
        cerr << endl;
    }
}

string print_pred(const Stump & stump)
{
    const Feature_Space & fs = *stump.feature_space();
    
    const char * op = (fs.info(stump.feature).categorical() ? "=" : "<");

    string result
        = format("%10s %s %-6s",
                 fs.print(stump.feature, stump.arg).c_str(), op,
                 fs.print(stump.feature).c_str());

    return result;
}

Boosted_Stumps
train_boosting(const Training_Data & training_set,
               const Training_Data & validation_set,
               unsigned max_iter, unsigned min_iter, int verbosity,
               vector<Feature> features,
               const Feature & predicted,
               bool true_only,
               int num_print_weights, bool profile, bool fair,
               float feature_prop, int committee_size, int trace,
               int cost_function, int output_function, int update_alg,
               float ignore_highest, float sample_prop, int weight_function,
               const std::vector<Weight_Spec> & weight_spec)
{
    Boosted_Stumps_Generator generator;
    generator.init(training_set.feature_space(), predicted);
    generator.defaults();

    generator.max_iter = max_iter;
    generator.min_iter = min_iter;
    generator.verbosity = verbosity;
    generator.profile = profile;
    generator.fair = fair;
    generator.weak_learner.feature_prop = feature_prop;
    generator.weak_learner.committee_size = committee_size;
    generator.weak_learner.trace = trace;
    generator.cost_function = (Cost_Function)cost_function;
    generator.output_function = (Boosted_Stumps::Output)output_function;
    generator.weak_learner.update_alg = (Stump::Update)update_alg;

    distribution<float> training_ex_weights
        = apply_weight_spec(training_set, weight_spec);
    distribution<float> validate_ex_weights
        = apply_weight_spec(validation_set, weight_spec);


    return generator.generate_stumps(training_set, validation_set,
                                     training_ex_weights,
                                     validate_ex_weights, features, -1);
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
    bool data_is_sparse     = false;
    float equalize_beta     = 0.0;
    string weight_spec      = "";
    int draw_graphs         = false;
    //Stop_At stop_at        = BEST_RECOG;
    string scale_output;
    string scale_input;
    string output_file;
    string variable_name_file;
    string equalize_name = "LABEL";
    int verbosity           = 1;
    vector<string> ignore_features;
    vector<string> optional_features;
    int cross_validate      = 1;
    int repeat_trials       = 1;
    bool dump_testing       = false;
    bool true_only          = false;
    int num_print_weights   = 0;
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
    int num_buckets         = DEFAULT_NUM_BUCKETS;
    string group_feature_name = "";
    bool eval_by_group      = false;  // Evaluate a group at a time?
    bool eval_by_group_set  = false;  // Value of eval_by_group has been set?
    string predicted_name   = "LABEL";
    vector<string> type_overrides;

    vector<string> extra;
    {
        using namespace CmdLine;

        static const Option data_options[] = {
            { "validation-split", 'V', validation_split, validation_split,
              false, "split X% of training data for validation", "0-100" },
            { "testing-split", 'T', testing_split, testing_split,
              false, "split X% of training data for testing", "0-100" },
            { "randomize-order", 'R', NO_ARG, flag(randomize_order),
              false, "randomize the order of data before splitting" },
            { "variable-names", 0, variable_name_file, variable_name_file,
              false, "read variables names from FILE", "FILE" },
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
            { "regression", 'N', NO_ARG, flag(is_regression, is_regression_set),
              false, "force dataset to be a regression problem" },
            { "num-buckets", '1', num_buckets, num_buckets,
              true, "number of buckets to discretize into (0=off)", "INT" },
            { "group-feature", 'g', group_feature_name, group_feature_name,
              false, "use FEATURE to group examples in dataset", "FEATURE" },
            { "type-override", 'Y', string(), push_back(type_overrides),
              false, "override feature types for matching", "REGEX=TYPE,..." },
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
            { "sample-proportion", 0, sample_prop, sample_prop,
              true, "use only X proportion of samples each iter", "0.0-1.0" },
            { "weight-function", 'a', weight_function, weight_function,
              true, "weight function: 0 = normal, 1 = conf", "0,1" },
            { "predict-feature", 'L', predicted_name, predicted_name,
              true, "train classifier to predict FEATURE", "FEATURE" },
            Last_Option
        };

        static const Option weight_options[] = {
            { "equalize-beta", 'E', equalize_beta, equalize_beta,
              true, "equalize labels (b in w=freq^(-b); 0.0=off)", "0.0-1.0" },
            { "equalize-feature", 'F', equalize_name, equalize_name,
              true, "equalize based upon feature FEATURE","FEATURE" },
            { "weight-spec", 'W', weight_spec, weight_spec,
              false, "use SPEC for weights", "VAR1(BETA1),..." },
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
            { "print-weights", 0, num_print_weights, num_print_weights,
              false, "print NUM higest weighted training examples", "NUM" },
            { "print-confusion", 'C', NO_ARG, increment(print_confusion),
              false, "print confusion matrix" },
            { "trace", 't', trace, trace,
              false, "set trace level to LEVEL (0 off)", "LEVEL" },
            { "eval-by-group", 0, NO_ARG, flag(eval_by_group, eval_by_group_set),
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

    if (extra.empty()) {
        cerr << "error: need to specify (at least) training data" << endl;
        exit(1);
    }

    cerr << all_info() << endl;

    Datasets datasets;
    datasets.init(extra, verbosity, profile);

    std::shared_ptr<Mutable_Feature_Space> feature_space = datasets.feature_space;

    vector<Feature> features;
    map<string, Feature> feature_index;
    Feature predicted;


    do_features(*datasets.data[0], feature_space, predicted_name,
                ignore_features, optional_features,
                min_feature_count, verbosity, features,
                predicted, feature_index, type_overrides);
    
    Feature group_feature = MISSING_FEATURE;

    if (group_feature_name != "") {
        if (!feature_index.count(group_feature_name)) {
            cerr << "feature_index.size() = " << feature_index.size()
                 << endl;
            throw Exception("grouping feature " + group_feature_name
                            + " not found in data");
        }
        
        group_feature = feature_index[group_feature_name];

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
    
    if (remove_aliased)
        remove_aliased_examples(*datasets.training, predicted, verbosity, profile);
        
    /* Write a null classifier, if there is no training data or we can't
       train for some reason. */
    if (datasets.training->example_count() == 0) {
        write_null_classifier(output_file, predicted, feature_space, verbosity);
        return 0;
    }

    vector<Weight_Spec> untrained_weight_spec
        = parse_weight_spec(*feature_space, equalize_name, equalize_beta,
                            weight_spec, group_feature);

    vector<Weight_Spec> trained_weight_spec
        = train_weight_spec(*datasets.training, untrained_weight_spec);
    
    if (verbosity > 4)
        print_weight_spec(trained_weight_spec, feature_space);

    /* Now for the training. */
        vector<distribution<float> > accum_acc(datasets.testing.size());

    for (unsigned i = 0;  i < repeat_trials;  ++i) {
        if (repeat_trials > 1) cerr << "trial " << (i + 1) << endl;

        Boosted_Stumps current
            = train_boosting(*datasets.training, *datasets.validation,
                             max_iter, min_iter,
                             verbosity, features, predicted,
                             true_only, num_print_weights,
                             profile, fair, feature_prop, committee_size, trace,
                             cost_function, output_function, update_alg,
                             ignore_highest, sample_prop, weight_function, 
                             trained_weight_spec);

        if (current.stumps.size() == 0) {
            write_null_classifier(output_file, predicted, feature_space,
                                  verbosity);
            continue;
        }

        GLZ_Probabilizer prob;
        if (probabilize_data < 0 || probabilize_data > 1)
            throw Exception("probabilize-data must be 0 or 1 (currently "
                            + ostream_format(probabilize_data) + ")");

        std::shared_ptr<const Training_Data> prob_set
            = (probabilize_data == 0 ? datasets.training : datasets.validation);
        
        distribution<float> pr_weights(prob_set->example_count(), 1.0);

        if (probabilize_weighted)
            pr_weights
                = apply_weight_spec(*prob_set, trained_weight_spec);

        prob.train(*prob_set, current, pr_weights,
                   probabilize_mode, probabilize_link);

        if (repeat_trials == 1 || verbosity > 2) {
            boost::timer t;
            cerr << "Stats over training set: " << endl;
            calc_stats(current, prob, *datasets.training, draw_graphs, dump_testing,
                       print_confusion, eval_by_group, group_feature);
            
            cerr << "Stats over validation set: " << endl;
            calc_stats(current, prob, *datasets.validation, draw_graphs,
                       dump_testing, print_confusion, eval_by_group,
                       group_feature);

            if (profile)
                cerr << "   [accuracy: " << t.elapsed() << "s]" << endl;
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
        for (unsigned j = 0;  j < datasets.testing.size();  ++j) {
            cerr << "Stats over testing set " << j << ":" << endl;
            
            calc_stats(current, prob, *datasets.testing[j], draw_graphs,
                       dump_testing, print_confusion, eval_by_group,
                       group_feature);
            
            if (repeat_trials > 1) {
                float acc = current.accuracy(*datasets.testing[j]);
                accum_acc[j].push_back(acc);
                cerr << "trial " << i + 1 << ": accuracy " << acc * 100.0
                     << "%, average over all trials = "
                     << accum_acc[j].total() * 100.0 / (i + 1)
                     << "%" << endl;
            }
        }
        
        if (i != repeat_trials - 1)
            datasets.reshuffle();
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
