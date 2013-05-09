/* training_data_tool.cc
   Jeremy Barnes, 4 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   A tool to analyze training data sets.
*/

#include "jml/boosting/training_data.h"
#include "jml/boosting/training_index.h"
#include "jml/boosting/dense_features.h"
#include "boosting_tool_common.h"
#include "jml/utils/file_functions.h"
#include "jml/utils/parse_context.h"
#include "jml/stats/moments.h"
#include "jml/utils/pair_utils.h"
#include "jml/stats/sparse_distribution.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/floating_point.h"

#include <iterator>
#include <iostream>
#include <set>
#include <cmath>

#include <boost/math/distributions/normal.hpp>

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;

using namespace ML;

/** Return the iterator before.  Requires a bidirectional iterator. */
template<class Iterator>
Iterator pred(Iterator it)
{
    it--;
    return it;
}

/** Return the iterator after.  Requires a forward iterator. */
template<class Iterator>
Iterator succ(Iterator it)
{
    it++;
    return it;
}

/** Structure to contain stats on a variable. */
struct Variable_Stats {
    Variable_Stats()
    {
        clear();
    }

    void clear()
    {
        counts.clear();
        total_count = missing = denorm = 0;
        min = max = range = mean = mode = median = stddev = NAN;
        label_totals.clear();
        label_min.clear();
        label_max.clear();
        label_counts.clear();
    }

    /** Count of unique values of the variable. */
    typedef map<double, int> counts_type;
    counts_type counts;
    
    /** Total count of all variables. */
    size_t total_count;

    /** Number which were missing (NaN) */
    size_t missing;

    /** Number which were denormalized (usually infinite) */
    size_t denorm;

    /** Statistics. */
    double min;
    double max;
    double range;
    double mode;
    double mean;
    double median;
    double stddev;
    double r_squared;
    double b;

    // Only for when we have two labels
    distribution<double> label_totals;
    distribution<double> label_min;
    distribution<double> label_max;
    distribution<double> label_counts;
    
    /** Add an instance to the stats. */
    void add(double val)
    {
        counts[val] += 1;
    }

    /** Calculate the stats from the values accumulated. */
    void calc_from_counts()
    {
        /* Min and max are very easy. */
        if (counts.empty()) { min = NAN;  max = NAN;  range = NAN; }
        if (counts.size() > 0) {
            min = counts.begin()->first;
            max = pred(counts.end())->first;
            range = max - min;
        }

        /* Scan through and find the median and mode. */
        size_t pos = 0;
        size_t mode_count = 0;
        //double prev = NAN;
        vector<double> data;
        for (counts_type::const_iterator it = counts.begin();
             it != counts.end();  ++it) {
            double val = it->first;
            size_t count = it->second;

            if (count > mode_count) {
                mode = val;
                mode_count = count;
            }

            if (pos <= total_count / 2 && pos + count > total_count / 2)
                median = val;

            for (unsigned i = 0;  i < count;  ++i) data.push_back(val);
        }

        /* Scan through and calculate the mean and standard deviation. */
        mean = ML::mean(data.begin(), data.end());
        stddev = std_dev(data.begin(), data.end(), mean);
    }

    /** Calculate from a vector of values. */
    void calc(vector<float> values, const vector<double> & labels, int nl)
    {
        clear();

        if (nl == 2) {
            label_totals.resize(nl);
            label_min.resize(nl, INFINITY);
            label_max.resize(nl, -INFINITY);
            label_counts.resize(nl);
        }

        /* Calculate the r-squared. */
        double sum_vsq = 0.0, sum_v = 0.0, sum_lsq = 0.0, sum_l = 0.0;
        double sum_vl = 0.0, n = 0.0;
        for (unsigned i = 0;  i < values.size();  ++i) {
            double v = values[i], l = labels[i];
            if (!finite(v)) continue;
            sum_vsq += v*v;  sum_v += v;
            sum_lsq += l*l;  sum_l += l;
            sum_vl  += v*l;
            n += 1.0;

            if (nl == 2) {
                int label = labels[i];
                if (label < 0 || label > 1) continue;
                label_totals[label] += v;
                label_min[label] = std::min(label_min[label], v);
                label_max[label] = std::max(label_max[label], v);
                label_counts[label] += 1;
            }
        }
        
        double svl = n * sum_vl - sum_v * sum_l;
        double svv = n * sum_vsq - sum_v * sum_v;
        double sll = n * sum_lsq - sum_l * sum_l;

        r_squared = svl*svl / (svv * sll);
        b = svl / svv;

        //cerr << "n = " << n << " sum_vsq = " << sum_vsq << " sum_v = "
        //     << sum_v << " sum_lsq = " << sum_lsq << " sum_l = " << sum_l
        //     << "  sum_vl = " << sum_vl << endl;
        
        /* Sort them to improve the efficiency of the counting. */
        std::sort(values.begin(), values.end());

        vector<pair<double, int> > pairs;

        double last_value = 0.0;
        for (unsigned i = 0;  i < values.size();  ++i) {
            double value = values[i];
            if (std::isnan(value)) ++missing;
            else if (!isfinite(value)) ++denorm;
            else if (i != 0 && last_value == value) {
                assert(!pairs.empty());
                pairs.back().second += 1;
            }
            else pairs.push_back(make_pair(value, 1));

            last_value = value;
        }
        
        counts.insert(pairs.begin(), pairs.end());
        total_count = values.size();
        
        calc_from_counts();
    }
};

double t_test(const Variable_Stats & s1, const Variable_Stats & s2)
{
    return (s1.mean - s2.mean)
        / sqrt(sqr(s1.stddev)/(double)s1.total_count
               + sqr(s2.stddev)/(double)s2.total_count);
}

int main(int argc, char ** argv)
try
{
    ios::sync_with_stdio(false);

    bool variable_header    = false;
    string variable_name_file;
    string predicted_name   = "LABEL";

    int verbosity           = 1;
    bool is_regression      = false;   // Set to be a regression?
    bool by_label           = false;

    vector<string> dataset_files;

    namespace opt = boost::program_options;

    opt::options_description data_options("Data options");
    opt::options_description output_options("Output options");
    {
        using namespace boost::program_options;

        data_options.add_options()
            ( "variable-names,n", value<string>(&variable_name_file),
              "read variables names from FILE")
            ( "variable-header,d", value<bool>(&variable_header)->zero_tokens(),
              "use first line of file as a variable header")
            ( "regression,N", value<bool>(&is_regression),
              "force dataset to be a regression problem" )
            ( "predict-feature,L", value<string>(&predicted_name),
              "train classifier to predict FEATURE" )
            ( "dataset", value<vector<string> >(&dataset_files),
              "datasets to process" )
            ( "by-label", value<bool>(&by_label)->zero_tokens(),
              "further break down stats by label");

        output_options.add_options()
            ( "verbosity,v", value(&verbosity),
              "set verbosity to LEVEL [0-3]" );

        positional_options_description p;
        p.add("dataset", -1);

        options_description all_opt;
        all_opt
            .add(data_options)
            .add(output_options);

        all_opt.add_options()
            ("help,h", "print this message");
        
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

    if (dataset_files.empty())
        throw Exception("error: need to specify at least one data set");

    /* Variables for holding our datasets and feature spaces. */
    vector<std::shared_ptr<Dense_Training_Data> > data(dataset_files.size());
    vector<std::shared_ptr<Dense_Feature_Space> > fs(dataset_files.size());
    vector<Feature> predicted(dataset_files.size(), MISSING_FEATURE);

    vector<string> feature_names;
    map<string, vector<int> > feature_name_map;

    /* First, read in all of the data. */
    for (unsigned i = 0;  i < dataset_files.size();  ++i) {
        fs[i].reset(new Dense_Feature_Space());
        data[i].reset(new Dense_Training_Data());
        data[i]->init(dataset_files[i], fs[i]);

        if (verbosity > 0)
            cout << "dataset \'" << dataset_files[i] << "\': "
                 << data[i]->all_features().size() << " features, "
                 << data[i]->example_count() << " rows." << endl;

        vector<string> data_feature_names = fs[i]->feature_names();
        
        for (unsigned j = 0;  j < data_feature_names.size();  ++j) {
            string feat = data_feature_names[j];
            if (feature_name_map.count(feat) == 0) {
                feature_names.push_back(feat);
                feature_name_map[feat] = vector<int>(dataset_files.size(), -1);
            }
            feature_name_map[feat][i] = j;
        }

        /* Work out what feature we're predicting. */
        if (predicted_name != "")
            fs[i]->parse(predicted_name, predicted[i]);

        /** Work out what kind of feature it is */
        //data[i]->preindex_features();

        if (fs[i]->info(predicted[i]).type() == UNKNOWN) {
            fs[i]->set_info(predicted[i], guess_info(*data[i], predicted[i]));
        }

    }
    
    /* Work out our features. */
    int nf = feature_names.size();

    cerr << "overall: " << nf << " features." << endl;

    const Feature NONE(-1);

    /* Calculate labels to be used in calculation of correlations. */
    vector<distribution<double> > label_dists;
    for (unsigned d = 0;  d < data.size();  ++d) {
        label_dists.push_back(distribution<double>(data[d]->example_count()));
        for (unsigned x = 0;  x < data[d]->example_count();  ++x) {
            label_dists.back()[x] = (*data[d])[x][predicted[d]];
        }
    }
    
    /* Generate the stats. */
    vector<double> significance(nf, 0.0);
    vector<double> t_values(nf, 0.0);
    vector<int> contained(nf, 0);

    for (unsigned f = 0;  f < nf;  ++f) {
        string feat = feature_names[f];
        const vector<int> & ids = feature_name_map[feat];

        cout << feat << ":" << endl;
        vector<Variable_Stats> stats(data.size());

        cout << " set   values      min      max     mean      std     mode"
             << "    uniq      r^2      int"
             << endl;
        /* Go over each dataset. */
        for (unsigned d = 0;  d < data.size();  ++d) {
            if (ids[d] == -1) continue;

            int nl = data[d]->label_count(predicted[d]);

            //cerr << "nl = " << nl << " by_label = " << by_label << endl;
            //cerr << "predicted[d] = " << fs[d]->print(predicted[d]) << endl;
            //cerr << "info = " << fs[d]->info(predicted[d]) << endl;

            Feature feature(ids[d]);

            ++contained[f];

            vector<double> vec;

            // TODO: fill in
            stats[d].calc(data[d]->index().values(feature), label_dists[d], nl);

            cout << format("  %2d %8zd %8.3f %8.3f %8.3f %8.3f %8.3f "
                           "%7zd %8.6f %8.3g\n",
                           d,
                           stats[d].total_count - stats[d].missing
                               - stats[d].denorm,
                           stats[d].min, stats[d].max,
                           stats[d].mean,
                           stats[d].stddev, stats[d].mode,
                           stats[d].counts.size(),
                           stats[d].r_squared, stats[d].b);

            if (nl == 2 && by_label) {
                for (unsigned i = 0;  i < nl;  ++i) {
                    cout
                        << format("        label %d: min %8.3f max: %8.3f avg: %8.3f count: %7.0f",
                                  i, stats[d].label_min[i],
                                  stats[d].label_max[i],
                                  stats[d].label_totals[i] / stats[d].label_counts[i],
                                  stats[d].label_counts[i])
                        << endl;

                }
            }
        }
        
        //cerr << "contained[" << f << "] = " << contained[f] << endl;

        if (data.size() == 2 && contained[f] == 2) {
            cout << endl;
            double t = t_test(stats[0], stats[1]);

            if (!isfinite(t) && stats[0].stddev == 0.0 && stats[1].stddev == 0)
                t = 0.0;

            

            double sig = (boost::math::cdf(boost::math::normal_distribution<double>(), abs(t)) - 0.5) * 2.0;

            cout << format("   t    %6.3f\n   sig %6.2f%%\n",
                           t, sig * 100.0);

            t_values[f] = t;
            significance[f] = sig;
        }
        cout << endl;
    }
    
    if (data.size() == 2) {
        cout << "features sorted by abs(t) score..." << endl;

        vector<pair<double, pair<int, double> > > sorted(nf);

        for (unsigned f = 0;  f < nf;   ++f) {
            sorted[f].second.first = f;
            if (!isfinite(t_values[f])) {
                sorted[f].first = INFINITY;
                sorted[f].second.second = INFINITY;
            }
            else {
                sorted[f].first = abs(t_values[f]);
                sorted[f].second.second = significance[f];
            }
        }

        sort_on_first_descending(sorted);

        cout << "   abs(t)      sig  variable" << endl;
        for (unsigned f = 0;  f < nf;  ++f) {
            cout << format("%9.3f %8.2f  ",
                           sorted[f].first, sorted[f].second.second * 100.0)
                 << feature_names[sorted[f].second.first] << endl;
        }
    }
}
catch (const std::exception & exc) {
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
