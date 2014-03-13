/* boosting_tool_common.cc
   Jeremy Barnes, 6 June 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of our common code for the boosting tools.
*/

#include "jml/boosting/config_impl.h"
#include "boosting_tool_common.h"
#include "jml/utils/file_functions.h"
#include "jml/utils/parse_context.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/classifier.h"
#include "jml/boosting/probabilizer.h"
#include "jml/utils/vector_utils.h"
#include <boost/timer.hpp>
#include <boost/regex.hpp>
#include "jml/boosting/null_classifier.h"
#include "jml/boosting/null_feature_space.h"
#include "jml/boosting/evaluation.h"
#include "jml/boosting/feature_info.h"
#include "jml/boosting/training_index.h"
#include "jml/boosting/data_aliases.h"
#include "jml/boosting/weighted_training.h"
#include "jml/math/xdiv.h"
#include "jml/stats/distribution_ops.h"


using namespace std;

namespace ML {

namespace {

/** Graphic metrics, that don't put the extremities like the others. */

struct Graphic_Metrics {

    Graphic_Metrics()
        : counts(10)
    {
    }

    void record(float value, float weight = 1.0)
    {
        if (value < 0.0 || value > 1.0) {
            cerr << format("value = %.12f", value);
            throw Exception("Graphic_Metrics::record(): value out of range");
        }
        int bucket = (int)std::floor(value * 10.0);
        if (bucket == 10) bucket = 9;
        counts[bucket] += weight;
    }

    void histogram(const std::string & title,
                   std::vector<std::string> & lines,
                   int max_width,
                   std::string count_fmt = "detect") const
    {
        double max = counts.max();

        if (count_fmt == "detect") {
            bool all_integral = true;
            for (unsigned i = 0;  i < counts.size() && all_integral;  ++i)
                if (((int)counts[i]) != counts[i]) all_integral = false;
            count_fmt = (all_integral ? "%.0f" : "%0.1f");
        }

        /* Make sure we have enough lines. */
        if (lines.size() < 11) lines.resize(11);

        /* How wide is the widest line? */
        size_t max_line_width = 0;
        for (unsigned i = 0;  i < lines.size();  ++i)
            max_line_width = std::max(max_line_width, lines[i].size());

        /* Make them all this length. */
        for (unsigned i = 0;  i < lines.size();  ++i)
            lines[i].resize(max_line_width, ' ');

        /* Add the title */
        lines[0] += title;

        int stars = 15;  // how many stars we want

        /* Work out the width of the counts. */
        distribution<int> widths
            = round(counts / (max ? max : 1) * stars).cast<int>();

        /* Add the distributions. */
        for (unsigned i = 0;  i < 10;  ++i) {
            string & line = lines[i + 1];
            line += format("%d: ", i);
            line += string(widths[i], '*');
            line += ' ';
            line += format(count_fmt.c_str(), counts[i]);
            if (line.size() < max_width)
                line += string(max_width - line.size(), ' ');
        }
    }
    
    distribution<double> counts;
};

template<class It>
int idxmax(It begin, It end)
{
    It el = std::max_element(begin, end);
    if (el == end) return -1;
    return el - begin;
}

template<class It>
std::pair<int, int>
idxmax2(It begin, It end)
{
    typedef typename std::iterator_traits<It>::value_type value_type;
    if (begin == end) return make_pair(-1, -1);
    value_type best = *begin++, second = best;
    int ibest = 0, isecond = -1, i = 1;

    for (; begin != end;  ++i, ++begin) {
        value_type val = *begin;

        if (val > best) {
            isecond = ibest;
            second = best;
            ibest = i;
            best = val;
        }
        else if (val > second || isecond == -1) {
            isecond = i;
            second = val;
        }
    }

    return make_pair(ibest, isecond);
}

} // file scope

void calc_stats(const Classifier_Impl & current,
                const Optimization_Info & opt_info,
                const Decoder_Impl & prob,
                const Training_Data & data,
                //const distribution<float> & weights,
                int draw_graphs,
                bool dump_testing,
                int dump_confusion,
                bool by_group,
                const Feature & group_feature)
{
    cerr << "dump_testing = " << dump_testing << endl;
    cerr << "by_group = " << by_group << endl;

    size_t nl = current.label_count();
    size_t nx = data.example_count();

    distribution<float> weights;  // todo: pass this in

    if (weights.empty()) weights.resize(nx, 1.0);

    //cerr << "nl = " << nl << "  nx = " << nx << endl;

    //cerr << "draw_graphs = " << draw_graphs << "  dump_confusion = "
    //     << dump_confusion << endl;

    Feature predicted = current.predicted();

    if (nx == 0) { cout << "0 examples" << endl;  return;  } // nothing to do

    if (nl == 1) {
        /* Regression problem gets treated specially. */
        calc_stats_regression(current, opt_info, prob,
                              data, draw_graphs, dump_testing,
                              dump_confusion);
        return;
    }

    /* Test it out. */
    vector<distribution<float> > avg_label(nl, distribution<float>(nl, 0.0));
    distribution<float> avg_corr(nl, 0.0);
    distribution<float> avg_inc(nl, 0.0);
    double correct = 0.0, total = 0.0;
    vector<unsigned> num_label(nl);

    Graphic_Metrics metrics[2][nl];
    Graphic_Metrics margins[nl];
    Graphic_Metrics rankings[nl];
    Graphic_Metrics correct_graph, incorrect_graph;
    Graphic_Metrics ranking_graph, margin_graph;

    if (dump_testing) {
        cout << "examp lbl";
        for (unsigned i = 0;  i < current.label_count();  ++i)
            cout << format("%6s", format("in%d", i).c_str())
                 << format("%6s", format("out%d", i).c_str());
        cout << "  poss  corr  marg" << endl;
    }

    const vector<Label> & labels = data.index().labels(predicted);

    //cerr << "by_group = " << by_group << " group_feature = "
    //     << data.group_feature_ << "  nl = " << nl << endl;

    by_group = by_group && group_feature != MISSING_FEATURE && nl == 2;
    
    if (!by_group) {
        //boost::progress_timer timer;
        for (unsigned x = 0;  x < nx;  ++x) {
            distribution<float> input = current.predict(data[x], opt_info);
            
            distribution<float> result = prob.apply(input);
            //distribution<float> result = input;
            
            int highest, highest_2;
            boost::tie(highest, highest_2)
                = idxmax2(result.begin(), result.end());
            //int highest_np = idxmax(input.begin(), input.end());

            double w = weights[x];

            Correctness c = correctness(result, predicted, data[x]);
            correct += w * c.possible * c.correct;
            total += w * c.possible;

            int label = labels[x];  // todo: make it work again...

            // Value to record.  Add a tiny bias to make exact zero margin
            // sit on the positive side.
            float margin_to_record
                = std::max<float>
                (std::min<float>(1.0f, c.margin / 2.0f + 0.50000001f),
                 0.0f);

            if (c.margin >= -1.0 && c.margin <= 1 && isfinite(c.margin))
                margins[label].record(margin_to_record);
            
            distribution<float> result2 = result;
            int ranking = 1;
            while (result2.max() != result[label] && ranking < 10) {
                result2[idxmax(result2.begin(), result2.end())] = 0.0;
                ++ranking;
            }
            rankings[label].record(std::min<float>(ranking * 0.1f, 1.0f));
            
            ranking_graph.record(std::min<float>(ranking * 0.1f, 1.0f));

            if (c.margin >= -1.0 && c.margin <= 1.0 && isfinite(c.margin))
                margin_graph.record(margin_to_record);
            
            avg_label[label] += result;
            num_label[label] += 1;
            
            for (unsigned l = 0;  l < nl;  ++l) {
                if (result[l] >= 0.0 && result[l] <= 1.0 && isfinite(result[l])) {
                    metrics[label == l][l].record(result[l]);
                    
                    if (label == l)
                        correct_graph.record(result[l]);
                    else incorrect_graph.record(result[l]);
                }
                
                if (label == l) avg_corr[l] += result[l];
                else avg_inc[l] += result[l];
            }
            
            if (dump_testing) {
                cout << format("%5d %3d", x, label);
                for (unsigned i = 0;  i < result.size();  ++i)
                    cout << format("%6.2f ", input[i])
                         << format("%5.2f", result[i]);
                cout <<  format(" %5.2f %5.2f %5.2f", c.possible, c.correct,
                                c.margin);
                //cout << data.row_comment(x);
                cout << endl;
            }
        }

        float recog_rate = xdiv(correct, total);
        
        distribution<float> label_freq = data.index().category_freqs(predicted);

        cerr << "label freq = " << label_freq << endl;
        
        cerr << format("recog rate = %9.1f/%7.0f = %8.3f%%",
                       correct, total,
                       recog_rate * 100.0) << endl;
        cerr << format("baseline   = %9.1f/%7.0f = %8.3f%%",
                       label_freq.max(), label_freq.total(),
                       xdiv<float>(label_freq.max(),
                                         label_freq.total()) * 100.0) << endl;
        
        float acc, rmse;
        boost::tie(acc, rmse) = current.accuracy(data);

        cerr << "non-prob accuracy              = "
             << format("%8.3f%% %8.3f%%",
                       acc * 100.0, rmse * 100.0) << endl;
    }
    else {
        /* Evaluate a group at a time */

        /* Get a list of groups. */
        Joint_Index group_index
            = data.index().dist(group_feature, BY_VALUE, IC_LABEL | IC_EXAMPLE);
        
        vector<float> groups;
        vector<vector<int> > group_examples;
        vector<int> group_numbers(nx, -1);

        float last_val = 0.0;
        for (Index_Iterator it = group_index.begin();
             it != group_index.end();  ++it) {
            float val = it->value();
            if (it == group_index.begin() || val != last_val) {
                groups.push_back(val);
                group_examples.push_back(vector<int>());
                last_val = val;
            }
            group_examples.back().push_back(it->example());
            group_numbers[it->example()] = group_examples.size() - 1;
        }
        
        int group_count = group_examples.size();
        double baseline = 0.0;

        //cerr << group_count << " groups" << endl;
        //cerr << nx << " examples" << endl;
        //cerr << "groups = " << groups << endl;

        /* Loop through a group at a time... */
        for (unsigned g = 0;  g < group_examples.size();  ++g) {
            int nxg = group_examples[g].size();

            distribution<float> correct_dist(nxg);
            distribution<float> values_dist(nxg);
            distribution<float> weights_dist(nxg);

            for (unsigned i = 0;  i < nxg;  ++i) {
                int x = group_examples[g][i];
                distribution<float> input = current.predict(data[x], opt_info);
                
                distribution<float> result = prob.apply(input);

                int label              = labels[x]; 
                correct_dist[i]        = label;
                values_dist[i]         = result[1];
                weights_dist[i]        = weights[x];
                avg_label[label] += result;
                num_label[label] += 1;
            }

            weights_dist /= weights_dist.size();
            double total_weight = weights_dist.total();

            float correct_value = 0.0;
            if (correct_dist.total() == 0)
                cerr << "warning: group " << groups[g] << " has no correct label"
                     << endl;
            else {
                correct_value = (values_dist * correct_dist).total()
                    / correct_dist.total();
                //if (correct_dist.total() > 1.0)
                //    cerr << "warning: group " << groups[g] << " has "
                //         << correct_dist.total() << " correct labels" << endl;
            }

            baseline += total_weight
                * (correct_dist.total() / correct_dist.size());

            int highest, highest_2;
            boost::tie(highest, highest_2)
                = idxmax2(values_dist.begin(), values_dist.end());
            
            bool is_correct = correct_dist[highest] != 0.0;
            
            for (unsigned i = 0;  i < nxg;  ++i) {
                if (values_dist[i] >= 0.0 && values_dist[i] <= 1.0 && isfinite(values_dist[i])) {
                    if (correct_dist[i] > 0.0)
                        correct_graph.record(values_dist[i], weights_dist[i]);
                    else incorrect_graph.record(values_dist[i], weights_dist[i]);
                }
            }
            
            float margin = (is_correct ? values_dist[highest]  - values_dist[highest_2]
                            : correct_value - values_dist[highest]);
            
            distribution<float> values_dist2 = values_dist;
            int ranking = 1;
            while (correct_dist[idxmax(values_dist2.begin(), values_dist2.end())] == 0.0
                   && ranking < 10) {
                values_dist2[idxmax(values_dist2.begin(), values_dist2.end())] = 0.0;
                ++ranking;
            }

            if (is_correct) correct += weights_dist.total();
            total += weights_dist.total();
            
            ranking_graph.record(ranking * 0.1);
            if (margin >= -1.0 && margin <= 1 && isfinite(margin))
                margin_graph.record((margin / 2) + 0.50000001, total_weight);
        }

        float recog_rate = xdiv<float>(correct, total);
        
        cerr << "recog rate = " << correct << "/" << total
             << " = " << recog_rate * 100.0 << "%" << endl;

        cerr << "baseline   = " << baseline << "/"
             << group_count << " = "
             << xdiv<float>(baseline, group_count) * 100.0
             << "%" << endl;
    }
    if (dump_confusion) {
        for (unsigned l = 0;  l < nl;  ++l) {
            avg_label[l] /= num_label[l];
            avg_corr[l] = xdiv(avg_corr[l], num_label[l]);
            avg_inc[l] = xdiv(avg_inc[l], nx - num_label[l]);
        }
        
        if (dump_confusion > 1) {
            cerr << "      ";
            for (unsigned lout = 0;  lout < nl;  ++lout)
                cerr << "  out  ";
            cerr << endl;
            
            cerr << " corr ";
            for (unsigned lout = 0;  lout < nl;  ++lout)
                cerr << format(" %3d   ", lout);
            cerr << endl;
            
            for (unsigned lcorr = 0;  lcorr < nl;  ++lcorr) {
                cerr << format(" %3d  ", lcorr);
                for (unsigned lout = 0;  lout < nl;  ++lout)
                    cerr << format("%6.3f ", avg_label[lcorr][lout]);
                cerr << endl;
            }
        
            cerr << "      ";
            for (unsigned l = 0;  l < nl;  ++l)
                cerr << "       ";
            cerr << endl;
        }
        
        cerr << " corr ";
        for (unsigned l = 0;  l < nl;  ++l)
            cerr << format("%6.3f ", avg_corr[l]);
        cerr << endl;
        
        cerr << " inc  ";
        for (unsigned l = 0;  l < nl;  ++l)
            cerr << format("%6.3f ", avg_inc[l]);
        cerr << endl;
        
        cerr << " gap  ";
        for (unsigned l = 0;  l < nl;  ++l)
            cerr << format("%6.3f ", avg_corr[l] - avg_inc[l]);
        cerr << endl;
        cerr << endl;
    }

    if (draw_graphs) {
        if (draw_graphs > 1 && !by_group) {
            for (unsigned l = 0;  l < nl;  ++l) {
                vector<string> lines(11);
                cerr << "graphs for label " << l << ":" << endl;
                metrics[1][l].histogram("correct:", lines, 20);
                metrics[0][l].histogram("incorrect:", lines, 40);
                margins[l].histogram("margins:", lines, 60);
                rankings[l].histogram("rankings:", lines, 80);
                std::copy(lines.begin(), lines.end(),
                          ostream_iterator<string>(cerr, "\n"));
                cerr << endl;
            }
        }
        
        {
            vector<string> lines(11);
            cerr << "overall graphs:" << endl;
            correct_graph.histogram("correct:", lines, 20);
            incorrect_graph.histogram("incorrect:", lines, 40);
            margin_graph.histogram("margins:", lines, 60);
            ranking_graph.histogram("rankings:", lines, 80);
            std::copy(lines.begin(), lines.end(),
                      ostream_iterator<string>(cerr, "\n"));
            cerr << endl;
        }
    }
}

void calc_stats_regression(const Classifier_Impl & current,
                           const Optimization_Info & opt_info,
                           const Decoder_Impl & prob,
                           const Training_Data & data, int draw_graphs,
                           bool dump_testing, int dump_confusion)
{
    size_t nl = current.label_count();
    size_t nx = data.example_count();

    Feature predicted = current.predicted();

    if (nl != 1)
        throw Exception("calc_stats_regression() called on classification "
                        "dataset");
    if (nl != current.label_count())
        throw Exception("calc_stats_regression(): nl doesn't match");

    /* Test it out. */
    if (dump_testing)
        cout << "examp       lbl     in     out     error";

    const vector<Label> & labels = data.index().labels(predicted);

    //double sum_r_squared = 0.0;

    /* Calculate the r-squared. */
    double sum_vsq = 0.0, sum_v = 0.0, sum_lsq = 0.0, sum_l = 0.0;
    double sum_vl = 0.0, n = nx;
    
    for (unsigned x = 0;  x < nx;  ++x) {
        distribution<float> input = current.predict(data[x], opt_info);
        
        distribution<float> result = prob.apply(input);
        
        float label = labels[x].value();

        float residual = label - result[0];

        double v = result[0], l = label;
        sum_vsq += v*v;  sum_v += v;
        sum_lsq += l*l;  sum_l += l;
        sum_vl  += v*l;
        
        if (dump_testing) {
            cout << format("%5d %10.4g %6.2f %6.2f %6.2f", x, label, input[0],
                           result[0], residual) << endl;
        }
        
        //sum_r_squared += residual * residual;
    }

    //cerr << "sum_r_squared = " << sum_r_squared / nx << endl;

    double svl = n * sum_vl - sum_v * sum_l;
    double svv = n * sum_vsq - sum_v * sum_v;
    double sll = n * sum_lsq - sum_l * sum_l;
    
    double r_squared = svl*svl / (svv * sll);
    double b = svl / svv;
    double bd = svl / sll;

#if 0
    cerr << "n = " << n << " sum_vsq = " << sum_vsq << " sum_v = "
         << sum_v << " sum_lsq = " << sum_lsq << " sum_l = " << sum_l
         << "  sum_vl = " << sum_vl << endl;

    cerr << "svl = " << svl << "  svv = " << svv << "  sll = " << sll << endl;
#endif

    cerr << "r^2 = " << r_squared << endl;
    cerr << "b = " << b << endl;
    cerr << "b' = " << bd << endl;
}

void remove_aliased_examples(Training_Data & data, const Feature & predicted,
                             int verbosity, bool profile)
{
    /** Clean the dataset by removing those examples which are aliased (where
        the features are the same). */
    boost::timer timer;
    
    /* Check for aliased rows in the training data. */
    vector<Alias> aliases = remove_aliases(data, predicted);
    int inhomogenous = 0;
    int aliased = 0;
    for (unsigned i = 0;  i < aliases.size();  ++i) {
        if (!aliases[i].homogenous) {
            aliased += aliases[i].examples.size();
            inhomogenous += 1;
        }
    }
    
    if (aliased > 0 && verbosity > 2) {
        cerr << aliased << " aliased examples in " << inhomogenous
             << " inhomogenous groups" << endl;
        
        if (verbosity > 4) {
            for (unsigned i = 0;  i < aliases.size();  ++i) {
                if (aliases[i].homogenous) continue;
                cerr << i << ": examples ";
                copy(aliases[i].examples.begin(),
                     aliases[i].examples.end(),
                     ostream_iterator<int>(cerr, " "));
                if (aliases[i].homogenous) cerr << "(homogenous)" << endl;
                else cerr << "(inhomogenous)" << endl;
            }
        }
    }
    
    if (profile)
        cerr << "[aliased: " << timer.elapsed() << "s]" << endl;
}

void do_features(const Training_Data & data,
                 std::shared_ptr<Mutable_Feature_Space> feature_space,
                 const string & predicted_name,
                 vector<string> ignore_features,
                 vector<string> optional_features,
                 int min_feature_count, int verbosity,
                 std::vector<Feature> & features,
                 Feature & predicted,
                 std::map<std::string, Feature> & feature_index,
                 vector<string> type_overrides)
{
    features.clear();
    vector<string> feature_names;

    features = data.all_features();
    std::sort(features.begin(), features.end());
    
    int total_features = features.size();
    
    for (unsigned i = 0;  i < features.size();  ++i) {
        feature_names.push_back(feature_space->print(features[i]));
    }

    /* Get an index of all of our variables. */
    feature_index.clear();
    for (unsigned i = 0;  i < features.size();  ++i)
        feature_index[feature_space->print(features[i])] = features[i];

    /* Find the predicted feature. */
    if (!feature_index.count(predicted_name))
        throw Exception("predicted variable " + predicted_name
                        + " not found in dataset");
    predicted = feature_index[predicted_name];

    Feature_Info predicted_info = feature_space->info(predicted);

    /* Ignore variables, if asked. */
    int ignored_features = 0;
    if (ignore_features.size()) {
        /* Get the names of all that are ignored.  These might be regexes, so
           we also have to search through with these regexes. */
        for (unsigned i = 0;  i < ignore_features.size();  ++i) {
            if (ignore_features[i].size() == 0) continue;  // blank line...
            else if (ignore_features[i][0] == '@') {
                /* If we get '@filename', we load them from a file. */
                string filename(ignore_features[i], 1);  // remove first char
                ifstream stream(filename.c_str());
                stream.exceptions(ios::badbit | ios::failbit);

                while (!stream.eof()) {
                    string s;
                    getline(stream, s);
                    ignore_features.push_back(s);
                }
            }
            else {
                string regex_str = ignore_features[i];
                bool negative = false;
                if (regex_str.length() && regex_str[0] == '!') {
                    regex_str = string(regex_str, 1);
                    negative = true;
                }

                /* Add it as a regular expression. */
                boost::regex regex(regex_str);

                /* Check out all of the variable names. */
                int matched = 0;
                for (unsigned j = 0;  j < features.size();  ++j) {
                    const string & name = feature_names[j];

                    bool match = boost::regex_match(name, regex);
                    if ((match && !negative) || (!match && negative)) {
                        std::swap(features[j], features.back());
                        std::swap(feature_names[j], feature_names.back());
                        features.pop_back();
                        feature_names.pop_back();
                        --j;
                        ++matched;
                    }
                }
                if (verbosity > 2 && matched > 0)
                    cerr << "regex '" << ignore_features[i] << "' matched "
                         << matched << " features" << endl;
                else if (verbosity > 0 && matched == 0)
                    cerr << "warning: regex '" << ignore_features[i]
                         << "' matched no features" << endl;
                ignored_features += matched;
            }
        }
    }

    /* Optional features, if asked. */
    if (optional_features.size()) {
        /* Get the names of all that are optional.  These might be regexes, so
           we also have to search through with these regexes. */
        for (unsigned i = 0;  i < optional_features.size();  ++i) {
            if (optional_features[i].size() == 0) continue;  // blank line...
            else if (optional_features[i][0] == '@') {
                /* If we get '@filename', we load them from a file. */
                string filename(optional_features[i], 1);  // remove first char
                ifstream stream(filename.c_str());
                stream.exceptions(ios::badbit | ios::failbit);

                while (!stream.eof()) {
                    string s;
                    getline(stream, s);
                    optional_features.push_back(s);
                }
            }
            else {
                /* Add it as a regular expression. */
                boost::regex regex(optional_features[i]);

                /* Check out all of the variable names. */
                int matched = 0;
                for (unsigned j = 0;  j < features.size();  ++j) {
                    const string & name = feature_names[j];

                    if (boost::regex_match(name, regex)) {
                        Mutable_Feature_Info info = feature_space->info(features[j]);
                        info.set_optional(true);
                        feature_space->set_info(features[j], info);
                        ++matched;
                    }
                }
                if (verbosity > 2 && matched > 0)
                    cerr << "regex '" << optional_features[i] << "' matched "
                         << matched << " features" << endl;
                else if (verbosity > 0 && matched == 0)
                    cerr << "warning: regex '" << optional_features[i]
                         << "' matched no features" << endl;
            }
        }
    }


    /* Ignore features with not enough examples. */
    int no_data_features = 0;
    for (unsigned i = 0;  i < features.size();  ++i) {
        int num_examples = data.index().count(features[i]);
        
        if (num_examples < min_feature_count) {
            std::swap(features[i], features.back());
            std::swap(feature_names[i], feature_names.back());
            features.pop_back();
            feature_names.pop_back();
            --i;
            no_data_features += 1;
        }
    }

    cerr << format("%d/%d/%d/%zd total/ignored/< %d examples/remaining "
                   "features",
                   total_features, ignored_features, no_data_features,
                   features.size(), min_feature_count) << endl;

    guess_all_info(data, *feature_space, true);

    /* Take any overrides in the feature info. */
    for (unsigned i = 0;  i < type_overrides.size();  ++i) {
        if (type_overrides[i].size() == 0) continue;  // blank line...
        else if (type_overrides[i][0] == '@') {
            /* If we get '@filename', we load them from a file. */
            string filename(type_overrides[i], 1);  // remove first char
            ifstream stream(filename.c_str());
            stream.exceptions(ios::badbit | ios::failbit);
            
            while (!stream.eof()) {
                string s;
                getline(stream, s);
                type_overrides.push_back(s);
            }
        }
        else {
            string::size_type pos = type_overrides[i].rfind('=');
            if (pos == string::npos)
                throw Exception("couldn't parse type override "
                                + type_overrides[i]
                                + ": format is REGEX=VALUE");
            
            string match(type_overrides[i], 0, pos);
            string result(type_overrides[i], pos + 1);

            /* Add it as a regular expression. */
            boost::regex regex(match);
            
            /* Check out all of the variable names. */
            int matched = 0;
            for (unsigned j = 0;  j < features.size();  ++j) {
                const string & name = feature_names[j];
                
                if (boost::regex_match(name, regex)) {
                    Mutable_Feature_Info info = feature_space->info(features[j]);
                    
                    if (result == "CATEGORICAL")
                        info = data.index()
                            .guess_info_categorical(features[j]);
                    else if (result == "REAL")
                        info.set_type(REAL);
                    else throw Exception("unknown or unimplemented type "
                                         "override: " + result);
                    
                    feature_space->set_info(features[j], info);
                    ++matched;
                }
            }
            if (verbosity > 2 && matched > 0)
                cerr << "regex '" << type_overrides[i] << "' matched "
                     << matched << " features" << endl;
            else if (verbosity > 0 && matched == 0)
                cerr << "warning: regex '" << type_overrides[i]
                     << "' matched no features" << endl;
        }
    }

    /* Print feature info if verbosity is high enough. */
    if (verbosity > 5) {
        cerr << "Feature info: " << endl;
        for (unsigned i = 0;  i < features.size();  ++i) {
            cerr << format("%-30s %s\n", feature_names[i].c_str(),
                           feature_space->info(features[i]).print().c_str());
        }
    }
}

void write_null_classifier(const std::string & filename,
                           const Feature & predicted,
                           std::shared_ptr<const Feature_Space> feature_space,
                           int verbosity)
{
    if (verbosity > 0)
        cerr << "warning: no data or less than 2 labels; can't do any "
             << "training" << endl;
    if (filename != "") {
        if (verbosity > 0)
            cerr << "writing null classifier to \'" << filename
                 << "\'... ";
        Classifier cl(Null_Classifier(feature_space, predicted));
        cl.save(filename);
        if (verbosity > 0) cerr << "done." << endl;
    }
}

void print_data_stats(const Training_Data & data)
{
}

void print_weight_spec(const std::vector<Weight_Spec> & weight_spec,
                       std::shared_ptr<Feature_Space> feature_space)
{
    for (unsigned i = 0;  i < weight_spec.size();  ++i) {
        const Feature & feature = weight_spec[i].feature;
        cerr << "weight spec " << i << ":" << endl;
        cerr << "  feature = " << feature_space->print(feature)
             << endl;
        if (weight_spec[i].weights.size() < 20) {
            for (map<float, float>::const_iterator it
                     = weight_spec[i].weights.begin();
                 it != weight_spec[i].weights.end();  ++it)
                cerr << "    " << feature_space->print(feature, it->first)
                     << " / " << it->second << endl;
        }
        else {
            cerr << "  " << weight_spec[i].weights.size() << " entries."
                 << endl;
        }
    }
}

} // namespace ML

