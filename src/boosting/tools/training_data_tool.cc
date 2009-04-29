/* training_data_tool.cc
   Jeremy Barnes, 4 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   A tool to analyze training data sets.
*/

#include "boosting/training_data.h"
#include "boosting/training_index.h"
#include "boosting/dense_features.h"
#include "boosting/boosting_tool_common.h"
#include "utils/command_line.h"
#include "utils/file_functions.h"
#include "utils/parse_context.h"
#include "stats/moments.h"
#include "utils/pair_utils.h"
#include "stats/sparse_distribution.h"
#include "utils/vector_utils.h"

#include <iterator>
#include <iostream>
#include <set>
#include <cmath>

#include <boost/math/distributions/normal.hpp>

using namespace std;

using namespace ML;
using namespace Math;
using namespace Stats;

/* Hash function for a float.   Just converts it bitwise to an int. */
struct float_hasher {
    int operator () (float x) const
    {
        return *(int *)(&x);
    }
};

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
        double prev = NAN;
        vector<double> data;
        for (counts_type::const_iterator it = counts.begin();
             it != counts.end();  ++it, prev = it->first) {
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
        mean = Stats::mean(data.begin(), data.end(), 0.0);
        stddev = Stats::std_dev(data.begin(), data.end(), mean);
    }

    /** Calculate from a vector of values. */
    void calc(vector<float> values, const vector<double> & labels)
    {
        clear();

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
            if (isnan(value)) ++missing;
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

double sqr(double x) { return x * x; }

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
    bool is_regression = false;   // Set to be a regression?
    bool is_regression_set = false;  // Value of is_regression set?

    vector<string> extra;
    {
        using namespace CmdLine;

        static const Option data_options[] = {
            { "variable-names", 'n', variable_name_file, variable_name_file,
              false, "read variables names from FILE", "FILE" },
            { "variable-header", 'd', NO_ARG, flag(variable_header),
              false, "use first line of file as a variable header" },
            { "regression", 'N', NO_ARG, flag(is_regression, is_regression_set),
              false, "force dataset to be a regression problem" },
            { "predict-feature", 'L', predicted_name, predicted_name,
              true, "train classifier to predict FEATURE", "FEATURE" },
            Last_Option
        };
        
        static const Option output_options[] = {
            { "quiet", 'q', NO_ARG, assign(verbosity, 0),
              false, "don't write any non-fatal output" },
            { "verbosity", 'v', optional(2), verbosity,
              false, "set verbosity to LEVEL (0-3)", "LEVEL" },
            Last_Option
        };

        static const Option options[] = {
            group("Data options", data_options),
            group("Output options",   output_options),
            Help_Options,
            Last_Option };

        Command_Line_Parser parser("training_data_tool", argc, argv, options);
        
        bool res = parser.parse();
        if (res == false) exit(1);
        
        extra.insert(extra.end(), parser.extra_begin(), parser.extra_end());
    }

    if (extra.empty())
        throw Exception("error: need to specify at least one data set");

    /* Variables for holding our datasets and feature spaces. */
    vector<boost::shared_ptr<Dense_Training_Data> > data(extra.size());
    vector<boost::shared_ptr<Dense_Feature_Space> > fs(extra.size());
    vector<Feature> predicted(extra.size(), MISSING_FEATURE);

    vector<string> feature_names;
    map<string, vector<int> > feature_name_map;

    /* First, read in all of the data. */
    for (unsigned i = 0;  i < extra.size();  ++i) {
        fs[i].reset(new Dense_Feature_Space());
        data[i].reset(new Dense_Training_Data());
        data[i]->init(extra[i], fs[i]);

        if (verbosity > 0)
            cout << "dataset \'" << extra[i] << "\': "
                 << data[i]->all_features().size() << " features, "
                 << data[i]->example_count() << " rows." << endl;

        vector<string> data_feature_names = fs[i]->feature_names();
        
        for (unsigned j = 0;  j < data_feature_names.size();  ++j) {
            string feat = data_feature_names[j];
            if (feature_name_map.count(feat) == 0) {
                feature_names.push_back(feat);
                feature_name_map[feat] = vector<int>(extra.size(), -1);
            }
            feature_name_map[feat][i] = j;
        }

        /* Work out what feature we're predicting. */
        if (predicted_name != "")
            fs[i]->parse(predicted_name, predicted[i]);
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
             << "   uniq      r^2    int"
             << endl;
        /* Go over each dataset. */
        for (unsigned d = 0;  d < data.size();  ++d) {
            if (ids[d] == -1) continue;

            Feature feature(ids[d]);

            ++contained[f];

            vector<double> vec;

            // TODO: fill in
            stats[d].calc(data[d]->index().values(feature), label_dists[d]);

            cout << format("  %2d %8zd %8.3f %8.3f %8.3f %8.3f %8.3f "
                           "%7zd %5.3f %8.3g\n",
                           d,
                           stats[d].total_count - stats[d].missing
                               - stats[d].denorm,
                           stats[d].min, stats[d].max,
                           stats[d].mean,
                           stats[d].stddev, stats[d].mode,
                           stats[d].counts.size(),
                           stats[d].r_squared, stats[d].b);
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
