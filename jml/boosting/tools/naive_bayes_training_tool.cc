/* naive_bayes_training_tool.cc                                       -*- C++ -*-
   Jeremy Barnes, 12 June 2003
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

   Tool to use to train naive_bayes with.  Very similar in spirit to the MLP
   training tool.
*/

#include "jml/boosting/naive_bayes.h"
#include "jml/boosting/boosted_stumps.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/sparse_features.h"
#include "jml/boosting/probabilizer.h"
#include "jml/boosting/decoded_classifier.h"
#include "jml/boosting/null_classifier.h"
#include "jml/boosting/null_feature_space.h"
#include "jml/boosting/boosting_tool_common.h"
#include "jml/boosting/tools/datasets.h"
#include "jml/boosting/weighted_training.h"
#include "jml/boosting/training_index.h"
#include "jml/utils/command_line.h"
#include "jml/utils/file_functions.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/parse_context.h"
#include "jml/utils/fixed_array.h"
#include "jml/utils/sgi_numeric.h"
#include "jml/utils/vector_utils.h"
#include "jml/stats/distribution.h"
#include "jml/stats/sparse_distribution.h"
#include "jml/math/selection.h"
#include <boost/progress.hpp>
#include <boost/timer.hpp>
#include "jml/stats/glz.h"
#include "jml/sections/sink_stream.h"
#include "jml/utils/metrics.h"
#include <boost/regex.hpp>
#include "jml/stats/moments.h"

#include <iterator>
#include <iostream>
#include <set>


using namespace std;

using namespace ML;
using namespace ML::Math;
using namespace ML::Stats;
using namespace ML::Metrics;


int main(int argc, char ** argv)
try
{
    ios::sync_with_stdio(false);

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
    int draw_graphs        = false;
    string scale_output;
    string scale_input;
    string output_file;
    string variable_name_file;
    string weight_var;
    string equalize_name    = "LABEL";
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
    int print_confusion    = false;
    int trace = 0;
    string weight_spec      = "";
    bool is_regression      = false;  // Set to be a regression?
    bool is_regression_set  = false;  // Value of is_regression has been set?
    int num_buckets         = DEFAULT_NUM_BUCKETS;
    string probabilize_link = "logit";
    string group_feature_name = "";
    bool eval_by_group      = false;  // Evaluate a group at a time?
    bool eval_by_group_set  = false;  // Value of eval_by_group has been set?
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
            { "regression", 'N', NO_ARG, flag(is_regression,is_regression_set),
              false, "force dataset to be a regression problem" },
            { "num-buckets", '1', num_buckets, num_buckets,
              true, "number of buckets to discretize into (0=off)", "INT" },
            { "remove-aliased", 'A', NO_ARG, flag(remove_aliased),
              false, "remove aliased training rows from the training data" },
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
            { "predict-feature", 'L', predicted_name, predicted_name,
              true, "train classifier to predict FEATURE", "FEATURE" },
            Last_Option
        };

        static const Option weight_options[] = {
            { "equalize-beta", 'E', equalize_beta, equalize_beta,
              true, "equalize labels (b in w=freq^(-b); 0.0=off)", "0.0-1.0" },
            { "equalize-feature", 'F', equalize_name, equalize_name,
              true, "equalize based upon feature FEATURE","FEATURE||'label'" },
            { "weight-var", 'W', weight_var, weight_var,
              false, "use var with NAME or NUM for weights", "NAME|#NUM" },
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

        Command_Line_Parser parser("naive_bayes_training_tool", argc, argv,
                                   options);
        
        bool res = parser.parse();
        if (res == false) exit(1);
        
        extra.insert(extra.end(), parser.extra_begin(), parser.extra_end());
    }

    if (equalize_beta != 0.0 && weight_var != "")
        throw Exception("can't specify equalize_beta and weight_var at the "
                        "same time");
    
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

    for (unsigned i = 0;  i < trained_weight_spec.size();  ++i) {
        cerr << "weight spec " << i << ":" << endl;
        cerr << "  feature = "
             << feature_space->print(trained_weight_spec[i].feature)
             << endl;
        if (trained_weight_spec[i].weights.size() < 10) {
            for (map<float, float>::const_iterator it
                     = trained_weight_spec[i].weights.begin();
                 it != trained_weight_spec[i].weights.end();  ++it)
                cerr << "    " << it->first << " / " << it->second << endl;
        }
        else {
            cerr << "  " << trained_weight_spec[i].weights.size() << " entries."
                 << endl;
        }
    }
    
    /* Now for the training. */
    
    vector<distribution<float> > accum_acc(datasets.testing.size());

    for (unsigned i = 0;  i < repeat_trials;  ++i) {
        if (repeat_trials > 1) cerr << "trial " << (i + 1) << endl;

        if (remove_aliased)
            remove_aliased_examples(*datasets.training, predicted, verbosity,
                                    profile);
        
        vector<Feature> features;
        map<string, Feature> feature_index;
        
        do_features(*datasets.training, feature_space, predicted_name,
                    ignore_features, optional_features,
                    min_feature_count, verbosity, features,
                    predicted, feature_index);
        
        vector<Feature> equalize_features;
        vector<float> equalize_betas;
        
        boost::tie(equalize_betas, equalize_features)
            = parse_weight_spec(feature_index, equalize_name, equalize_beta,
                                weight_spec);

        vector<Weight_Spec> trained_weight_spec
            = get_weight_spec(*datasets.training, equalize_betas,
                                               equalize_features);

        distribution<float> ex_weights
            = apply_weight_spec(*datasets.training, trained_weight_spec);

        fixed_array<float, 2> weights
            = expand_weights(*datasets.training, ex_weights, predicted);
        
        Naive_Bayes current(feature_space, predicted);
        Training_Params params;
        params["trace"] = trace;
        params["features"] = features;
        current.train_weighted(*datasets.training, params, weights);

        if (verbosity > 2) {
            cerr << endl << "Learned Bayes classifier function: " << endl;
            int nl = label_count;
            cerr << "feature                       split";
            if (nl == 2 && false)
                cerr << "       label1";
            else
                for (unsigned l = 0;  l < nl;  ++l)
                    cerr << format("    label%-4d", l);
            cerr << endl;

            for (unsigned i = 0;  i < current.features.size();  ++i) {
                string feat
                    = current.feature_space()
                        ->print(current.features[i].feature);

                string spaces = feat;
                std::fill(spaces.begin(), spaces.end(), ' ');
                
                for (unsigned j = 0;  j < 3;  ++j) {
                    if (j == 2) {
                        float sum_missing = 0.0;
                        
                        for (unsigned l = 0;  l < nl;  ++l)
                            sum_missing += current.probs[i][j][l];
                        
                        if (sum_missing == 0.0) continue;
                    }
                    if (j == 0) cerr << format("%-30s", feat.c_str());
                    else cerr << format("%-30s", spaces.c_str());
                    
                    if (j == 0) cerr << format(" <  %13f", current.features[i].arg);
                    else if (j == 1)
                        cerr << format(" >= %13f", current.features[i].arg);
                    else cerr << " MISSING         ";
                    
                    if (nl == 2 && false)
                        cerr << format("%9.4f", current.probs[i][j][1]);
                    else
                        for (unsigned l = 0;  l < nl;  ++l)
                            cerr << format("%9.4f", current.probs[i][j][l]);
                    cerr << endl;
                }
                cerr << endl;
            }
        }
        
        GLZ_Probabilizer prob;
        Training_Params pr_params;
        pr_params["glz_probabilizer_mode"] = probabilize_mode;
        params["glz_probabilizer_link"] = probabilize_link;
        
        if (probabilize_data < 0 || probabilize_data > 1)
            throw Exception("probabilize-data must be 0 or 1 (currently "
                            + ostream_format(probabilize_data) + ")");
        std::shared_ptr<const Training_Data> prob_set
            = (probabilize_data == 0 ? datasets.training : datasets.validation);
        
        distribution<float> pr_weights(prob_set->example_count(), 1.0);

        if (probabilize_weighted)
            pr_weights
                = apply_weight_spec(*prob_set, trained_weight_spec);

        prob.train(*prob_set, params, current, pr_weights);

        if (repeat_trials == 1 || verbosity > 2) {
            cerr << "Stats over training set: " << endl;
            calc_stats(current, prob, *datasets.training, draw_graphs,
                       dump_testing, print_confusion, eval_by_group,
                       group_feature);
            
            cerr << "Stats over validation set: " << endl;
            calc_stats(current, prob, *datasets.validation, draw_graphs,
                       dump_testing, print_confusion, eval_by_group,
                       group_feature);
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
        
        
        if (i != repeat_trials - 1) {
            /* Re-shuffle the data. */
            datasets.reshuffle();
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
