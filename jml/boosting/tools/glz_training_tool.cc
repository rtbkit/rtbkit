/* glz_training_tool.cc
   Jeremy Barnes, 6 August 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   A tool to train GLZ classifiers.
*/

#include "jml/boosting/config.h"
#include "jml/boosting/glz_classifier.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/sparse_features.h"
#include "jml/boosting/probabilizer.h"
#include "jml/boosting/decoded_classifier.h"
#include "jml/boosting/boosting_tool_common.h"
#include "jml/boosting/weighted_training.h"
#include "jml/boosting/training_index.h"
#include "jml/boosting/null_decoder.h"
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


int main(int argc, char ** argv)
try
{
    ios::sync_with_stdio(false);

    float validation_split  = 0.0;
    float testing_split     = 0.0;
    bool randomize_order    = false;
    bool variable_header    = false;
    bool data_is_sparse     = false;
    int label_count         = -1;
    float equalize_beta     = 0.0;
    int draw_graphs         = 0;
    string scale_output;
    string scale_input;
    string output_file;
    string variable_name_file;
    string weight_spec      = "";
    string equalize_name    = "label";
    int verbosity           = 1;
    vector<string> ignore_features;
    vector<string> optional_features;
    int cross_validate      = 1;
    int repeat_trials       = 1;
    bool val_with_test      = false;
    bool dump_testing       = false;
    int min_feature_count   = 1;
    bool remove_aliased     = false;
    bool profile            = false;
    float feature_prop      = 1.0;
    int print_confusion     = 0;
    int trace = 0;
    bool is_regression      = false;  // Set to be a regression?
    bool is_regression_set  = false;  // Value of is_regression set?
    string link_function = "default";
    string group_feature_name = "";
    bool eval_by_group      = false;  // Evaluate a group at a time?
    bool eval_by_group_set  = false;  // Value of eval_by_group has been set?
    int num_buckets         = DEFAULT_NUM_BUCKETS;
    string predicted_name   = "LABEL";

    vector<string> extra;
    {
        using namespace ML::CmdLine;

        static const Option data_options[] = {
            { "validation-split", 'V', validation_split, validation_split,
              false, "split X% of training data for validation", "0-100" },
            { "testing-split", 'T', testing_split, testing_split,
              false, "split X% of training data for testing", "0-100" },
            { "validate-with-testing", '1', NO_ARG, flag(val_with_test),
              false, "(CHEAT) use testing data for validation" },
            { "randomize-order", 'R', NO_ARG, flag(randomize_order),
              false, "randomize the order of data before splitting" },
            { "variable-names", 'n', variable_name_file, variable_name_file,
              false, "read variables names from FILE", "FILE" },
            { "variable-header", 'd', NO_ARG, flag(variable_header),
              false, "use first line of file as a variable header" },
            { "ignore-feature", 'z', string(), push_back(ignore_features),
              false,"ignore feature with name matching REGEX","REGEX|@FILE" },
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
            Last_Option
        };
        
        static const Option training_options[] = {
            { "cross-validate", 'X', cross_validate, cross_validate,
              false, "run N-fold cross validation", "INT" },
            { "repeat-trials", 'r', repeat_trials, repeat_trials,
              false, "repeat experiment N times", "INT" },
            { "feature-proportion", '3', feature_prop, feature_prop,
              false, "lazy train on this proportion of features", "0.0-1.0" },
            { "link-function", 'K', link_function, link_function,
              true, "link function for GLZ", "log|logit|linear..." },
            { "predict-feature", 'L', predicted_name, predicted_name,
              true, "train classifier to predict FEATURE", "FEATURE" },
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
            { "eval-by-group", 0, NO_ARG, flag(eval_by_group, eval_by_group_set),
              false, "evaluate by group rather than by example" },
            Last_Option
        };

        static const Option options[] = {
            group("Data options", data_options),
            group("Weight options", weight_options),
            group("Training options", training_options),
            group("Output options",   output_options),
            Help_Options,
            Last_Option };

        Command_Line_Parser parser("naive_bayes_training_tool", argc, argv,
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
    Feature predicted;


    do_features(*datasets.data[0], feature_space, predicted_name,
                ignore_features, optional_features,
                min_feature_count, verbosity, features,
                predicted, feature_index);
    
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
        write_null_classifier(output_file, predicted, verbosity);
        return 0;
    }

    vector<Feature> equalize_features;
    vector<float> equalize_betas;

    boost::tie(equalize_betas, equalize_features)
        = parse_weight_spec(feature_index, equalize_name, equalize_beta,
                            weight_spec);

    vector<Weight_Spec> trained_weight_spec
        = get_weight_spec(*datasets.training, equalize_betas,
                                           equalize_features);

    if (verbosity > 4)
        print_weight_spec(trained_weight_spec, feature_space);
    
    bool regression_problem = false;  // TODO: fix

    if (link_function == "default")
        link_function = (regression_problem ? "linear" : "logit");
    
    /* Now for the training. */
    
    float accum_acc = 0.0;

    for (unsigned i = 0;  i < repeat_trials;  ++i) {
        if (repeat_trials > 1) cerr << "trial " << (i + 1) << endl;


        distribution<float> ex_weights
            = apply_weight_spec(*datasets.training, trained_weight_spec);

        fixed_array<float, 2> weights
            = expand_weights(*datasets.training, ex_weights, group_feature);
        
        GLZ_Classifier current(feature_space, predicted);
        current.link = parse_link_function(link_function);

        Training_Params params;
        params["features"] = features;
        current.train_weighted(*datasets.training, params, weights);

        if (verbosity > 2) {
            cerr << endl << "Learned GLZ function: " << endl;
            int nl = label_count;
            cerr << "feature                                 ";
            if (nl == 2 && false)
                cerr << "       label1";
            else
                for (unsigned l = 0;  l < nl;  ++l)
                    cerr << format("    label%-4d", l);
            cerr << endl;

            for (unsigned i = 0;  i < current.features.size() + current.add_bias;
                 ++i) {
                string feat;
                if (i == current.features.size()) feat = "BIAS";
                else feat = current.feature_space()->print(current.features[i]);
                cerr << format("%-40s", feat.c_str());
                if (nl == 2 && false)
                    cerr << format("%13f", current.weights[1][i]);
                else
                    for (unsigned l = 0;  l < nl;  ++l)
                        cerr << format("%13f", current.weights[l][i]);
                cerr << endl;
            }
            cerr << endl;
        }

        cerr << "Stats over training set: " << endl;
        calc_stats(current, Null_Decoder(), *datasets.training, draw_graphs - 1,
                   dump_testing, print_confusion, eval_by_group, group_feature);
        
        cerr << "Stats over validation set: " << endl;
        calc_stats(current, Null_Decoder(), *datasets.validation,
                   draw_graphs - 1,
                   dump_testing, print_confusion, eval_by_group, group_feature);
        
        if (output_file != "") {
            if (verbosity > 0)
                cerr << "writing to \'" << output_file << "\'... ";
            Classifier cl(current);
            cl.save(output_file);
            if (verbosity > 0) cerr << "done." << endl;
        }
        
        if (!datasets.testing.empty()) {
            /* Test all of the testing datasets. */
            for (unsigned j = 0;  j < datasets.testing.size();  ++j) {
                cerr << "Stats over testing set " << j << ":" << endl;
                
                calc_stats(current, Null_Decoder(), *datasets.testing[j],
                           draw_graphs, dump_testing, print_confusion,
                           eval_by_group, group_feature);
                
                if (repeat_trials > 1 && j == 0) {
                    float acc = current.accuracy(*datasets.testing[j]);
                    accum_acc += acc;
                    cerr << "trial " << i + 1 << ": accuracy " << acc * 100.0
                         << "%, average over all trials = "
                         << accum_acc * 100.0 / (i + 1) << "%" << endl;
                }
            }
        }
        
        if (i != repeat_trials - 1) {
            /* Re-shuffle the data. */
            datasets.reshuffle();
        }
    }

    if (repeat_trials > 1)
        cerr << "average accuracy over " << repeat_trials << " trials was "
             << xdiv<float>(accum_acc, repeat_trials) * 100.0
             << "%." << endl;
}
catch (const std::exception & exc) {
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
