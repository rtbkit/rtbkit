/* boosting_training_tool.cc                                       -*- C++ -*-
   Jeremy Barnes, 12 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Tool to use to train boosting with.  Very similar in spirit to the MLP
   training tool.
*/

#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/sparse_features.h"
#include "jml/boosting/classifier.h"
#include "jml/utils/command_line.h"
#include "jml/utils/file_functions.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/parse_context.h"
#include <boost/multi_array.hpp>
#include "jml/utils/sgi_numeric.h"
#include "jml/utils/vector_utils.h"
#include "jml/stats/distribution.h"
#include "jml/stats/sparse_distribution.h"
#include "jml/sections/sink_stream.h"
#include "jml/stats/distribution_ops.h"

#include <iterator>
#include <iostream>
#include <set>


using namespace std;

using namespace ML;
using namespace Math;
using namespace Stats;

/** Return the number of the maximum label in the file. */
sparse_distribution<int, size_t>
file_label_counts(const string & filename)
{
    File_Read_Buffer file(filename);
    Parse_Context context(filename, file.start(), file.end());

    sparse_distribution<int, size_t> result;

    context.skip_line();  // skip header line
    while (context) {
        int label = context.expect_unsigned();
        result[label] += 1;
        context.skip_line();
    }
    
    cerr << context.get_line() << " lines scanned." << endl;
    cerr << context.get_pos() - context.get_start() << " bytes total"
         << endl;

    return result;
}

/** Return the maximum number of labels in all of the files. */
sparse_distribution<int, size_t>
get_label_counts(const vector<string> & filenames)
{
    sparse_distribution<int, size_t> result;
    for (unsigned i = 0;  i < filenames.size();  ++i)
        result += file_label_counts(filenames[i]);
    return result;
}

int main(int argc, char ** argv)
try
{
    ios::sync_with_stdio(false);

    bool data_is_sparse     = false;
    int verbosity           = 1;
    string classifier_file;
    string output_filename;

    vector<string> extra;
    {
        using namespace CmdLine;

        static const Option classifier_options[] = {
            { "classifier-file", 'C', classifier_file, classifier_file,
              false, "load classifier from FILE", "FILE" },
            Last_Option
        };

        static const Option data_options[] = {
            { "sparse-data", 'S', NO_ARG, flag(data_is_sparse),
              false, "dataset is in sparse format" },
            Last_Option
        };
        
        static const Option output_options[] = {
            { "quiet", 'q', NO_ARG, assign(verbosity, 0),
              false, "don't write any non-fatal output" },
            { "verbosity", 'v', optional(2), verbosity,
              false, "set verbosity to LEVEL (0-3)", "LEVEL" },
            { "output-file", 'o', output_filename, output_filename,
              false, "write output to FILE", "FILE" },
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

    /* The rest of the options are training sets.  If we have specified a
       training split, then we will discard some of it. */

    if (extra.empty()) {
        cerr << "error: need to specify (at least) training data" << endl;
        exit(1);
    }

    /* Get the label counts by scanning the datasets. */
    sparse_distribution<int, size_t> labels = get_label_counts(extra);
    cerr << "labels[0] = " << labels[0] << "  labels[1] = " << labels[1]
         << endl;

    if (verbosity > 0) {
        cerr << labels.size() << " unique labels: ";
        size_t total = labels.total();
        for (map<int, size_t>::const_iterator it = labels.begin();
             it != labels.end();  ++it) {
            cerr << format("%d (%zd/%.1f%%) ", it->first, it->second,
                           (double)it->second / (double)total * 100.0);
        }
        cerr << endl;
    }
    
    size_t label_count = 0;
    if (!labels.empty()) {
        sparse_distribution<int, size_t>::const_iterator it = labels.end();
        --it;
        label_count = it->first + 1;
    }
    else throw Exception("No training data, therefore no labels");

    /* Raw data. */
    std::shared_ptr<Feature_Space> feature_space;
    vector<std::shared_ptr<Training_Data> > data(extra.size());

    /* Broken down and split up datasets.  First one is training, second is
       validation, any after that are testing. */
    vector<std::shared_ptr<Training_Data> > testing;

    /* Variables for when we are sparse. */
    std::shared_ptr<Sparse_Feature_Space> sparse_feature_space;
    vector<std::shared_ptr<Sparse_Training_Data> > sparse_data(extra.size());

    /* Variables for when we are dense. */
    std::shared_ptr<Dense_Feature_Space> dense_feature_space;
    vector<std::shared_ptr<Dense_Training_Data> > dense_data(extra.size());

    /* Set up the feature space. */
    if (data_is_sparse) {
        sparse_feature_space.reset(new Sparse_Feature_Space());
        feature_space = sparse_feature_space;
    }
    else {
        dense_feature_space.reset(new Dense_Feature_Space());
        feature_space = dense_feature_space;
    }
    
    /* First, read in all of the data. */
    for (unsigned i = 0;  i < extra.size();  ++i) {
        //cerr << "i = " << i << "  extra[i] = " << extra[i] << endl;

        if (data_is_sparse) {
            sparse_data[i]
                .reset(new Sparse_Training_Data(extra[i], sparse_feature_space,
                                                label_count));
            data[i] = sparse_data[i];
        }
        else {
            dense_data[i]
                .reset(new Dense_Training_Data(extra[i], dense_feature_space,
                                               label_count));
            data[i] = dense_data[i];
        }
        
        if (verbosity > 0)
            cerr << "dataset \'" << extra[i] << "\': "
                 << data[i]->all_features().size() << " features, "
                 << data[i]->example_count() << " rows." << endl;
        cerr << "label freq: " << data[i]->label_freq << endl;
    }

    if (verbosity > 0) {
        unsigned testing_rows = 0;
        for (unsigned i = 0;  i < testing.size();  ++i) {
            testing_rows += testing[i]->example_count();
        }
        
        cerr << "data size: " << testing_rows << endl;
    }
    
    /* Load the classifier. */
    if (classifier_file == "")
        throw Exception("Need to specify a classifier to test.");
    Classifier classifier;
    classifier.load(classifier_file, feature_space);

    if (labels.size() > classifier.label_count())
        throw Exception(format("%zd labels in classifier but %zd found",
                               classifier.label_count(), labels.size()));

    filter_ostream out(output_filename);
    out << feature_space->print() << "\n";

    /* Test all of the testing datasets. */
    for (unsigned i = 0;  i < data.size();  ++i) {
        for (unsigned x = 0;  x < data[i]->example_count();  ++x) {
            distribution<float> result
                = classifier.predict(*(*data[i])[x].features);
            cerr << "result = " << result << endl;
            out << (*data[i])[x].label << " "
                << feature_space->print(*(*data[i])[x].features);
            if (feature_space->type() == Feature_Space::SPARSE) {
                for (unsigned i = 0;  i < result.size();  ++i)
                    out << format(" inp%d:%f", i, result[i]);
            }
            out << "\n";
        }
    }
}
catch (const std::exception & exc) {
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
