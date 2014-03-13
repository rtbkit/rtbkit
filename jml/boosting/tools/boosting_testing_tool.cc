/* boosting_training_tool.cc                                       -*- C++ -*-
   Jeremy Barnes, 12 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Tool to use to train boosting with.  Very similar in spirit to the MLP
   training tool.
*/

#include "jml/boosting/training_data.h"
#include "jml/boosting/training_index.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/sparse_features.h"
#include "jml/boosting/classifier.h"
#include "jml/boosting/boosted_stumps.h"
#include "jml/boosting/boosted_stumps_impl.h"
#include "jml/boosting/probabilizer.h"
#include "jml/boosting/decoded_classifier.h"
#include "jml/boosting/boosting_tool_common.h"
#include "jml/boosting/null_decoder.h"
#include "jml/utils/command_line.h"
#include "jml/utils/file_functions.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/parse_context.h"
#include <boost/multi_array.hpp>
#include "jml/utils/sgi_numeric.h"
#include "jml/utils/vector_utils.h"
#include "jml/stats/distribution.h"
#include "jml/math/selection.h"
#include <boost/progress.hpp>
#include "jml/sections/sink_stream.h"
#include "jml/utils/metrics.h"
#include <boost/regex.hpp>
#include "jml/stats/distribution_ops.h"
#include "jml/boosting/evaluation.h"

#include <iterator>
#include <iostream>
#include <set>


using namespace std;

using namespace ML;
using namespace Math;
using namespace Stats;
using namespace Metrics;


struct Tracer {
    Tracer(size_t label_count)
        : nl(label_count)
    {
    }

    /* Record a result against the correct feature. */
    void operator () (const distribution<float> & dist, float weight,
                      const Feature & feature) const
    {
        if (scores[feature].empty())
            scores[feature] = distribution<float>(nl, 0.0);
        scores[feature] += weight * dist;
    }
    
    /* Indexed by the feature number and then the label number.  Tells us
       how much effect that feature has had on that label. */
    typedef map<Feature, distribution<float> > scores_type;
    mutable scores_type scores;

    size_t nl;

    /* Return the actual result. */
    distribution<float> totals() const
    {
        distribution<float> result(nl);

        for (scores_type::const_iterator it = scores.begin();
             it != scores.end();  ++it) {
            result += it->second;
        }

        return result;
    }
};

/** Small structure to accumulate the effect of a variable. */
struct Effect1 {
    Effect1()
        : correct(0.0), incorrect(0.0), 
          margin_correct(0.0), margin_incorrect(0.0)
    {
    }

    void add(const distribution<float> & vals, int label, int highest,
             int next_highest)
    {
        bool is_correct = (vals[highest] == vals[label]);
        if (is_correct) {
            correct += vals[label];
            abs_correct += abs(vals[label]);
            double margin = vals[label] - vals[next_highest];
            margin_correct += margin;
        }
        else {
            incorrect += vals[label];
            abs_incorrect += abs(vals[label]);
            double margin = vals[highest] - vals[label];
            margin_incorrect += margin;
        }
    }

    double correct;
    double abs_correct;
    double incorrect;
    double abs_incorrect;
    double margin_correct;
    double margin_incorrect;

    static string key()
    {
        return "   score    corr abs_corr      inc  abs_inc";//"   m_corr    m_inc";
    }

    string print() const
    {
        return format("%8.3f %8.3f %8.3f %8.3f %8.3f",//" %8.3f %8.3f",
                      score(), correct, abs_correct, incorrect, abs_incorrect);
        //margin_correct, margin_incorrect);
    }
    
    double score() const
    {
        return correct - incorrect;
    }

    bool operator < (const Effect1 & other) const
    {
        return score() < other.score();
    }
};

struct Effect2 {
    Effect2()
    {
    }

    void add(const distribution<float> & vals, int label, int highest,
             int next_highest)
    {
        if (correct.size() < vals.size()) correct.resize(vals.size());
        if (incorrect.size() < vals.size()) incorrect.resize(vals.size());
        if (abs_correct.size() < vals.size()) abs_correct.resize(vals.size());
        if (abs_incorrect.size() < vals.size()) abs_incorrect.resize(vals.size());
        for (unsigned l = 0;  l < vals.size();  ++l) {
            bool is_correct = (label == l);
            if (is_correct) {
                correct[l] += vals[l];
                abs_correct[l] += abs(vals[l]);
            }
            else {
                incorrect[l] += vals[l];
                abs_incorrect[l] += abs(vals[l]);
            }
        }
    }

    distribution<double> correct;
    distribution<double> abs_correct;
    distribution<double> incorrect;
    distribution<double> abs_incorrect;

    static string key()
    {
        return "   score    corr abs_corr      inc  abs_inc";//"   m_corr    m_inc";
    }

    string print() const
    {
        string result = format("%12.3f", score());
        //for (unsigned l = 0;  l < correct.size();  ++l)
        //    result += format(" %7.1f %7.3f", correct[l] - incorrect[l],
        //                     (correct[l] / abs_correct[l])
        //                     * (incorrect[l] / abs_incorrect[l]));
        return result;
        //return format("%8.3f %8.3f %8.3f %8.3f %8.3f",//" %8.3f %8.3f",
        //              score(), correct, abs_correct, incorrect, abs_incorrect);
        //margin_correct, margin_incorrect);
    }
    
    double score() const
    {
        //return (((correct / abs_correct) * (incorrect / abs_incorrect))
        //        * (correct - incorrect)).total();
        return (correct * incorrect).total();
    }

    bool operator < (const Effect2 & other) const
    {
        return score() < other.score();
    }
};

typedef Effect2 Effect;


void trace_output(const Classifier & current, const Training_Data & data,
                  int feature_tracing, int verbosity = 1,
                  float min_output = 0.0, float max_output = 1.0)
{
    const Decoded_Classifier & decoded
        = dynamic_cast<const Decoded_Classifier &>(*current.impl);
    const Boosted_Stumps & stumps
        = dynamic_cast<const Boosted_Stumps &>(*decoded.classifier().impl);
    const GLZ_Probabilizer & prob
        = dynamic_cast<const GLZ_Probabilizer &>(decoded.decoder().impl());

    Feature predicted = current.predicted();
    
    size_t nl = current.label_count();
    size_t nx = data.example_count();

    map<Feature, Effect> effects;

    for (unsigned x = 0;  x < nx;  ++x) {
        Tracer tracer(nl);

        stumps.predict_core(data[x], tracer);

        distribution<float> input = tracer.totals();

        if (stumps.output == 1 || stumps.output == 2) {
            for (unsigned i = 0;  i < input.size();  ++i) {
                double e = exp(input[i]);
                input[i] = e / (e + 1.0 / e);
            }
            if (stumps.output == 2) input.normalize();
        }

        distribution<float> result = prob.apply(input);

        if (result[1] < min_output || result[1] > max_output) continue;

        int highest, highest_2;
        boost::tie(highest, highest_2) = idxmax2(result.begin(), result.end());

        int label    = (int)data[x][predicted]; 
        bool correct = highest == label;

        /* Merge the two maps in an efficient manner. */
        Tracer::scores_type::const_iterator it1 = tracer.scores.begin();
        Tracer::scores_type::const_iterator e1 = tracer.scores.end();
        map<Feature, Effect>::iterator it2 = effects.begin();
        map<Feature, Effect>::iterator e2 = effects.end();
        
        while (it1 != e1 && it2 != e2) {
            if (it1->first == it2->first) {
                it2->second.add(it1->second, label, highest, highest_2);
                ++it1;  ++it2;
            }
            else if (it1->first < it2->first) {
                it2->second.add(it1->second, label, highest, highest_2);
                ++it1;
            }
            else ++it2;
        }
        while (it1 != e1) { // new ones at end
            effects[it1->first].add(it1->second, label, highest, highest_2);
            ++it1;
        }
        
        if (verbosity > 1) {
            cout << "example " << x << ":  label = " << label
                 << (correct ? " correct" : " incorrect")
                 << " (" << result[label] << ")" << endl;
        }
        
        for (unsigned l = 0;  l < nl;  ++l) {
            if (nl == 2 && l == 0) continue;  // trace true for binary

            vector<pair<Feature, float> > features;
            double total = 0.0;
            for (Tracer::scores_type::const_iterator it
                     = tracer.scores.begin();
                 it != tracer.scores.end();  ++it) {
                features.push_back(make_pair(it->first, abs(it->second[l])));
                total += it->second[l];
            }

            if (verbosity > 1) {
                cout << "  label " << l
                     << format("  prob: %6.3f, raw: %6.3f, total: %6.3f",
                               result[l], input[l], total)
                     << endl;
            }
            
            sort_on_second_descending(features);

            if (verbosity > 2) {
                for (unsigned i = 0;
                     i < std::min(features.size(), (size_t)20);  ++i) {
                    Feature_Set::const_iterator first, last;
                    boost::tie(first, last)
                        = data[x].find(features[i].first);
                    if (first == last)
                        throw Exception("Feature not found");

                    cout << format("    %8.4f [%8.3g] ",
                                   tracer.scores[features[i].first][l],
                                   (*first).second)
                         << stumps.feature_space()->print(features[i].first)
                         << endl;
                }
            }
        }
    }

    if (feature_tracing != 0) {
        vector<pair<Feature, Effect> > sorted(effects.begin(), effects.end());
        sort_on_second_ascending(sorted);
        
        cout << "features from worst to best:" << endl;
        cout << endl;
        cout << Effect::key() << " feature" << endl;
        
        for (unsigned i = 0;  i < sorted.size();  ++i)
            cout << sorted[i].second.print() << " "
                 << stumps.feature_space()->print(sorted[i].first) << endl;
        cout << endl;
    }
}

void do_features(const Training_Data & data,
                 std::shared_ptr<const Feature_Space> feature_space,
                 std::vector<Feature> & features,
                 std::map<std::string, Feature> & feature_index)
{
    features.clear();
    vector<string> feature_names;

    features = data.all_features();
    std::sort(features.begin(), features.end());
    
    for (unsigned i = 0;  i < features.size();  ++i)
        feature_names.push_back(feature_space->print(features[i]));

    /* Get an index of all of our variables. */
    feature_index.clear();
    for (unsigned i = 0;  i < features.size();  ++i)
        feature_index[feature_space->print(features[i])] = features[i];
    
    //guess_all_info(data, *feature_space, true);
}


int main(int argc, char ** argv)
try
{
    ios::sync_with_stdio(false);

    bool sparse_data        = false;
    int draw_graphs         = 0;
    int print_confusion     = 0;
    bool trace_features     = false;
    int verbosity           = 1;
    int feature_tracing     = 1;
    string classifier_file;
    string group_feature_name = "";
    bool eval_by_group      = false;  // Evaluate a group at a time?
    bool eval_by_group_set  = false;  // Value of eval_by_group has been set?

    vector<string> extra;
    {
        using namespace CmdLine;

        static const Option classifier_options[] = {
            { "classifier-file", 'C', classifier_file, classifier_file,
              false, "load classifier from FILE", "FILE" },
            Last_Option
        };

        static const Option data_options[] = {
            { "sparse-data", 'S', NO_ARG, flag(sparse_data),
              false, "dataset is in sparse format" },
            Last_Option
        };
        
        static const Option output_options[] = {
            { "quiet", 'q', NO_ARG, assign(verbosity, 0),
              false, "don't write any non-fatal output" },
            { "verbosity", 'v', optional(2), verbosity,
              false, "set verbosity to LEVEL (0-3)", "LEVEL" },
            { "draw-graphs", 'G', NO_ARG, increment(draw_graphs),
              false, "draw graphs for two-class predictor" },
            { "print-confusion", 'C', NO_ARG, increment(print_confusion),
              false, "print confusion matrix" },
            { "trace-features", 'T', NO_ARG, flag(trace_features),
              false, "trace operation of boosting over its features" },
            { "feature-tracing", 't', feature_tracing, feature_tracing,
              true, "feature tracing: 1=correct, 2=margins", "1-2" },
            { "group-feature", 'g', group_feature_name, group_feature_name,
              false, "use FEATURE to group examples in dataset", "FEATURE" },
            { "eval-by-group", 0, NO_ARG, flag(eval_by_group, eval_by_group_set),
              false, "evaluate by group rather than by example" },
            Last_Option
        };

        static const Option options[] = {
            group("Classifier options", classifier_options),
            group("Data options", data_options),
            group("Output options",   output_options),
            Help_Options,
            Last_Option };

        Command_Line_Parser parser("boosting_testing_tool", argc, argv,
                                   options);
        
        bool res = parser.parse();
        if (res == false) exit(1);
        
        extra.insert(extra.end(), parser.extra_begin(), parser.extra_end());
    }

    if (extra.empty()) {
        cerr << "error: need to specify testing data" << endl;
        exit(1);
    }

    vector<std::shared_ptr<Training_Data> > data(extra.size());
    ssize_t var_count = -1;
    
    std::shared_ptr<Dense_Feature_Space> dense_feature_space;
    std::shared_ptr<Sparse_Feature_Space> sparse_feature_space;
    std::shared_ptr<Mutable_Feature_Space> feature_space;

    if (sparse_data) {
        sparse_feature_space.reset(new Sparse_Feature_Space());
        feature_space = sparse_feature_space;

        for (unsigned i = 0;  i < extra.size();  ++i) {
            std::shared_ptr<Sparse_Training_Data> dataset
                (new Sparse_Training_Data());
            dataset->init(extra[i], sparse_feature_space);
            data[i] = dataset;
            
            if (verbosity > 0)
                cerr << "dataset \'" << extra[i] << "\': "
                     << data[i]->all_features().size() << " vars, "
                     << data[i]->example_count() << " rows." << endl;
        }
    }
    else {
        dense_feature_space.reset(new Dense_Feature_Space());
        feature_space = dense_feature_space; 
        
        for (unsigned i = 0;  i < extra.size();  ++i) {
            std::shared_ptr<Dense_Training_Data> dataset
                (new Dense_Training_Data());
            dataset->init(extra[i], dense_feature_space);
            data[i] = dataset;
            
            /* Make sure that there are enough variables. */
            if (var_count == -1)
                var_count = dataset->variable_count();
            else if (var_count != dataset->variable_count())
                throw Exception(format("error: file \'%s\' has"
                                       " %zd variables; expected %zd",
                                       extra[i].c_str(),
                                       dataset->variable_count(), var_count));
            
            if (verbosity > 0)
                cerr << "dataset \'" << extra[i] << "\': "
                     << var_count << " vars, "
                     << dataset->example_count() << " rows." << endl;
        }
    }
    
    if (verbosity > 0) {
        unsigned rows = 0;
        for (unsigned i = 0;  i < data.size();  ++i)
            rows += data[i]->example_count();
        
        cerr << "data size: " << rows << " rows." << endl;
    }
    
    /* Now fix up the feature space. */
    bool use_existing = true;
    guess_all_info(*data[0], *feature_space, use_existing);


    /* Load the classifier. */
    if (classifier_file == "")
        throw Exception("Need to specify a classifier to test.");
    Classifier classifier;
    classifier.load(classifier_file, feature_space);

    vector<Feature> features;
    map<string, Feature> feature_index;

    cerr << "doing features..." << endl;

    do_features(*data[0], classifier.feature_space(), features, feature_index);

    cerr << "done." << endl;

    cerr << "predicted by classifier = " << classifier.predicted() << endl;
    cerr << "classifier predicted name = "
         << classifier.feature_space()->print(classifier.predicted())
         << endl;
    cerr << "dataset predicted name = "
         << feature_space->print(classifier.predicted()) << endl;

    /* Test all of the testing datasets. */
    for (unsigned i = 0;  i < data.size();  ++i) {
        cerr << "Stats over dataset " << i << ":" << endl;

        Null_Decoder decoder;

        calc_stats(*classifier.impl, decoder, *data[i], draw_graphs,
                   false, 1, false, MISSING_FEATURE);
        
        if (trace_features)
            trace_output(classifier, *data[i], feature_tracing, verbosity);
    }
}
catch (const std::exception & exc) {
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
