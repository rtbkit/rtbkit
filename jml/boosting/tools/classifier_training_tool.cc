/* classifier_training_tool.cc                                       -*- C++ -*-
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2006 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Tool to train an arbitrary classifier.
*/

#include "jml/boosting/classifier_generator.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/sparse_features.h"
#include "jml/boosting/probabilizer.h"
#include "jml/boosting/decoded_classifier.h"
#include "boosting_tool_common.h"
#include "jml/boosting/weighted_training.h"
#include "jml/boosting/training_index.h"
#include "jml/boosting/transform_list.h"
#include "jml/utils/vector_utils.h"
#include <boost/progress.hpp>
#include <boost/timer.hpp>
#include "jml/stats/moments.h"
#include "datasets.h"
#include "jml/utils/info.h"

#include <iterator>
#include <iostream>
#include <set>

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>


using namespace std;

using namespace ML;

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
    bool data_is_sparse     = false;
    float equalize_beta     = 0.0;
    string weight_spec      = "";
    int draw_graphs         = false;
    string output_file;
    string equalize_name    = "";
    int verbosity           = 1;
    vector<string> ignore_features;
    vector<string> optional_features;
    int repeat_trials       = 1;
    bool dump_testing       = false;
    int min_feature_count   = 1;
    bool remove_aliased     = false;
    bool profile            = false;
    int print_confusion     = false;
    string probabilize_link = "logit";
    int num_buckets         = DEFAULT_NUM_BUCKETS;
    string group_feature_name = "";
    bool eval_by_group      = false;  // Evaluate a group at a time?
    bool no_eval_by_group   = false;
    string predicted_name   = "LABEL";
    vector<string> type_overrides;
    vector<string> transformations;
    string config_file_name;
    string trainer_name;
    string trainer_type;
    string testing_filter;
    bool help_config        = false;

    vector<string> dataset_files;
    namespace opt = boost::program_options;

    opt::options_description data_options("Data options");
    opt::options_description training_options("Training options");
    opt::options_description weight_options("Weight options");
    opt::options_description probabilize_options("Probabilizer options");
    opt::options_description output_options("Output options");
    {
        using namespace boost::program_options;

        data_options.add_options()
            ("dataset", value<vector<string> >(&dataset_files))
            ("validation-split,V", value<float>(&validation_split),
             "split X% of training data for validation [0-100]"     )
            ( "testing-split,T", value<float>(&testing_split),
              "split X% of training data for testing [0-100]" )
            ( "randomize-order,R", value<bool>(&randomize_order)->zero_tokens(),
              "randomize the order of data before splitting" )
            ( "ignore-var,z", value<vector<string> >(&ignore_features),
              "ignore variable with name matching REGEX [REGEX|@FILE]" )
            ( "optional-feature,O", value<vector<string> >(&optional_features),
              "feature with name REGEX is optional [REGEX|@FILE]" )
            ( "sparse-data,S", value<bool>(&data_is_sparse)->default_value(true),
              "dataset is in sparse format" )
            ( "min-feature-count", value<int>(&min_feature_count),
              "don't consider features seen < NUM times [NUM]" )
            ( "remove-aliased,A", value<bool>(&remove_aliased)->default_value(true),
              "remove aliased training rows from the training data" )
            ( "num-buckets,1", value(&num_buckets),
              "number of buckets to discretize into (0=off) [INT]" )
            ( "group-feature,g", value(&group_feature_name),
              "use FEATURE to group examples in dataset [FEATURE]" )
            ( "type-override,Y", value<vector<string> >(&type_overrides),
              "override feature types for matching [REGEX=TYPE,...]" )
            ( "transformation,N", value<vector<string> >(&transformations),
              "add to list of dataset transformations [TRANS SPEC]")
            ( "testing-filter,f", value(&testing_filter),
              "add an extra test set filtered with filter [FILTER SPEC]");

        training_options.add_options()
            ( "repeat-trials,r", value<int>(&repeat_trials),
              "repeat experiment N times [INT]" )
            ( "predict-feature,L", value<string>(&predicted_name),
              "train classifier to predict feature [FEATURE NAME]");

        weight_options.add_options()
            ( "equalize-beta,E", value<float>(&equalize_beta),
              "equalize labels (b in w=freq^(-b); 0.0=off) [0.0-1.0]" )
            ( "equalize-feature,F", value<string>(&equalize_name),
              "equalize based upon feature FEATURE" )
            ( "weight-spec,W", value(&weight_spec),
              "use SPEC for weights [SPEC1,...]" );

        probabilize_options.add_options()
            ( "probabilize-mode,p", value(&probabilize_mode),
              "prob mode: 0=matrix, 1=pervar, 2=oneonly, 3=off" )
            ( "probabilize-link,K", value(&probabilize_link),
              "prob link function [logit|log]" )
            ( "probabilize-data,P", value(&probabilize_data),
              "data for prob: 0 = train, 1 = validate" )
            ( "probabilize-weighted,Q", value(&probabilize_weighted),
              "train probabilizer using weights also" );

        output_options.add_options()
            ( "output-file,o", value(&output_file),
              "write output network to FILE" )
            ( "verbosity,v", value(&verbosity),
              "set verbosity to LEVEL [0-3]" )
            ( "profile,l", value(&profile),
              "profile execution time" )
            ( "draw-graphs,G", value(&draw_graphs),
              "draw graphs for two-class predictor" )
            ( "dump-testing,D", value(&dump_testing),
              "dump output of classifier on testing sets" )
            ( "print-confusion,C", value(&print_confusion),
              "print confusion matrix" )
            ( "eval-by-group", value(&eval_by_group)->zero_tokens(),
              "evaluate by group rather than by example" )
            ( "no-eval-by-group", value(&no_eval_by_group)->zero_tokens(),
              "evaluate by example rather than by group" );

        positional_options_description p;
        p.add("dataset", -1);

        options_description all_opt;
        all_opt.add_options()
            ("configuration-file,c", value(&config_file_name),
             "read configuration parameters from FILE [FILE]")
            ("trainer-name,n", value(&trainer_name),
             "trainer is given by NAME in config [STRING]")
            ("trainer-type,t", value(&trainer_type),
             "trainer has type TYPE if not in config file [STRING]");
        all_opt
            .add(data_options).add(training_options).add(weight_options)
            .add(probabilize_options).add(output_options);
        all_opt.add_options()
            ("help,h", "print this message")
            ("help-config", value(&help_config)->zero_tokens(),
             "print out config options for the generator");
        
        variables_map vm;
        store(command_line_parser(argc, argv)
              .options(all_opt)
              .positional(p)
              .run(),
              vm);
        notify(vm);

        if (vm.count("help")) {
            cerr << all_opt << endl;
            return 1;
        }
    }

    /* Those extra options that contain an "=" sign are assumed to be
       configuration parameters.  The others are assumed to be datasets
       to train on.
    */

    vector<string> dataset_params;
    vector<string> option_params;

    for (unsigned i = 0;  i < dataset_files.size();  ++i) {
        string param = dataset_files[i];
        if (param.find('=') != string::npos)
            option_params.push_back(param);
        else dataset_params.push_back(param);
    }

    Configuration config;
    if (config_file_name != "") config.load(config_file_name);
    config.parse_command_line(option_params);

    if (trainer_type != "")
        config[(trainer_name == "" ? string("type") : trainer_name + ".type")]
            = trainer_type;

    std::shared_ptr<Classifier_Generator> generator
        = get_trainer(trainer_name, config);

    if (help_config) {
        Config_Options options = generator->options();
        options.dump(cerr);
        exit(0);
    }

    if (dataset_params.empty()) {
        cerr << "error: need to specify (at least) training data" << endl;
        exit(1);
    }

    if (verbosity > 0) cerr << all_info() << endl;

    Datasets datasets;
    datasets.init(dataset_params, verbosity, profile);

    std::shared_ptr<Mutable_Feature_Space> feature_space
        = datasets.feature_space;

    vector<Feature> features;
    map<string, Feature> feature_index;
    Feature predicted;

    int first_nonempty = -1;
    for (unsigned i = 0;  i < datasets.data.size();  ++i) {
        cerr << "dataset " << i << " has " << datasets.data[i]->example_count()
             << " examples" << endl;
        if (datasets.data[i]->example_count()) {
            first_nonempty = i;
            break;
        }
    }

    if (first_nonempty == -1) {
        /* Write the null classifier; there is no data */
        write_null_classifier(output_file, predicted, feature_space, verbosity);
        return 0;
    }

    do_features(*datasets.data[first_nonempty], feature_space, predicted_name,
                ignore_features, optional_features,
                min_feature_count, verbosity, features,
                predicted, feature_index, type_overrides);

    generator->init(feature_space, predicted);
    
    Feature group_feature = MISSING_FEATURE;

    if (group_feature_name != "") {
        if (!feature_index.count(group_feature_name)) {
            cerr << "feature_index.size() = " << feature_index.size()
                 << endl;
            throw Exception("grouping feature " + group_feature_name
                            + " not found in data");
        }
        
        group_feature = feature_index[group_feature_name];

        if (!no_eval_by_group)
            eval_by_group = true;

        cerr << "no_eval_by_group = " << no_eval_by_group << endl;
        cerr << "eval_by_group = " << eval_by_group << endl;
    }

    vector<Feature> grouping_features;
    if (group_feature != MISSING_FEATURE)
        grouping_features.push_back(group_feature);

    for (unsigned i = 0;  i < features.size();  ++i) {
        if (feature_space->info(features[i]).grouping())
            grouping_features.push_back(features[i]);
    }

    make_vector_set(grouping_features);

    cerr << "grouping_features = " << grouping_features << endl;
    
    if (grouping_features.size())
        datasets.fixup_grouping(grouping_features);
    

    float training_split = 100.0 - validation_split - testing_split;
    datasets.split(training_split, validation_split, testing_split,
                   randomize_order, group_feature, testing_filter);
    
    if (remove_aliased)
        remove_aliased_examples(*datasets.training, predicted, verbosity,
                                profile);
        
    Feature_Transformer transformer;
    if (!transformations.empty()) {
        std::shared_ptr<Transform_List> transforms
            (new Transform_List(datasets.feature_space));
        transforms->parse(transformations);
        transformer.init(transforms);
        datasets.transform(transformer);
    }
    
    cerr << "datasets.training->example_count() = "
         << datasets.training->example_count() << endl;

    /* Write a null classifier, if there is no training data or we can't
       train for some reason. */
    if (datasets.training->example_count() == 0) {
        write_null_classifier(output_file, predicted, feature_space, verbosity);
        return 0;
    }

    vector<Weight_Spec> untrained_weight_spec;

    if (equalize_name != "" || weight_spec != "") {
        untrained_weight_spec
            = parse_weight_spec(*feature_space, equalize_name, equalize_beta,
                                weight_spec, group_feature);
    }
    
    vector<Weight_Spec> trained_weight_spec
        = train_weight_spec(*datasets.training, untrained_weight_spec);

    if (verbosity > 4)
        print_weight_spec(trained_weight_spec, feature_space);
    
    
    /* Now for the training. */
    vector<distribution<float> > accum_acc(datasets.testing.size());
    
    Thread_Context context;

    for (unsigned i = 0;  i < repeat_trials;  ++i) {
        if (repeat_trials > 1) cerr << "trial " << (i + 1) << endl;

        cerr << "training weights:" << endl;
        distribution<float> training_ex_weights
            = apply_weight_spec(*datasets.training, trained_weight_spec);
        
        cerr << "validation weights:" << endl;
        distribution<float> validate_ex_weights
            = apply_weight_spec(*datasets.validation, trained_weight_spec);
        
        std::shared_ptr<Classifier_Impl> current
            = generator->generate(context,
                                  *datasets.training, *datasets.validation,
                                  training_ex_weights, validate_ex_weights,
                                  features);

        if (current->all_features().empty()) {
            cerr << "no features used by classifier; breaking" << endl;
            write_null_classifier(output_file, predicted, feature_space,
                                  verbosity);
            continue;
        }
        
        if (probabilize_data < 0 || probabilize_data > 1)
            throw Exception("probabilize-data must be 0 or 1 (currently "
                            + ostream_format(probabilize_data) + ")");
        
        Optimization_Info opt_info
            = current->optimize(feature_space->dense_features());

        std::shared_ptr<const Training_Data> prob_set
            = (probabilize_data == 0 ? datasets.training : datasets.validation);
        
        distribution<float> pr_weights(prob_set->example_count(), 1.0);
        
        if (probabilize_weighted)
            pr_weights
                = apply_weight_spec(*prob_set, trained_weight_spec);
        
        GLZ_Probabilizer prob;
        prob.train(*prob_set, *current, opt_info, pr_weights,
                   probabilize_mode, probabilize_link);

        cerr << prob.print() << endl;

        if (repeat_trials == 1 || verbosity > 2) {
            cerr << "Stats over training set: " << endl;
            calc_stats(*current, opt_info, prob,
                       *datasets.training, draw_graphs,
                       dump_testing,
                       print_confusion, eval_by_group, group_feature);
            
            cerr << "Stats over validation set: " << endl;
            calc_stats(*current, opt_info, prob,
                       *datasets.validation, draw_graphs,
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
            
            calc_stats(*current, opt_info,
                       prob, *datasets.testing[j], draw_graphs,
                       dump_testing, print_confusion, eval_by_group,
                       group_feature);
            
            if (repeat_trials > 1) {
                pair<float, float> acc
                    = current->accuracy(*datasets.testing[j]);
                accum_acc[j].push_back(acc.first);
                cerr << "trial " << i + 1 << ": accuracy " << acc.first * 100.0
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
            float mean = ML::mean(accum_acc[j].begin(), accum_acc[j].end());
            float std_dev
                = ML::std_dev(accum_acc[j].begin(), accum_acc[j].end(),
                              mean);
            cerr << "testing set " << j << ": " << repeat_trials
                 << " trials accuracy: mean "
                 << mean * 100.0 << "% std " << std_dev * 100.0 << "%."
                 << endl;
        }
    }

    if (dump_testing) {
        // Redump the datasets with all comments, etc intact
        // ...
    }
}
catch (const std::exception & exc) {
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
