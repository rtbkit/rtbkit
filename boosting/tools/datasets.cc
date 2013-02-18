/* datasets.cc                                                     -*- C++ -*-
   Jeremy Barnes, 28 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of datasets structure.
*/

#include "datasets.h"
#include "jml/utils/parse_context.h"
#include "jml/utils/file_functions.h"
#include "jml/utils/filter_streams.h"
#include "jml/boosting/sparse_features.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/feature_transformer.h"
#include <boost/timer.hpp>
#include "jml/boosting/feature_set_filter.h"

using namespace std;


namespace ML {

std::ostream & operator << (std::ostream & stream, Disposition d)
{
    switch (d) {
    case DSP_TRAIN: return stream << "TRAIN";
    case DSP_VALIDATE: return stream << "VALIDATE";
    case DSP_TEST: return stream << "TEST";
    case DSP_UNKNOWN: return stream << "UNKNOWN";
    default: return stream << format("Disposition(%d)", d);
    }
}


/*****************************************************************************/
/* DATASETS                                                                  */
/*****************************************************************************/

Datasets::
Datasets()
    : group_feature(MISSING_FEATURE)
{
}

void
Datasets::
init(const std::vector<std::string> & files, int verbosity, bool profile)
{
    this->verbosity = verbosity;
    this->profile = profile;

    /* Find the index of the first real file (not a TRAIN, VALIDATE or
       TEST indicator) so that we can test for sparseness. */
    int first_real_file = 0;
    while (first_real_file < files.size()
           && (files[first_real_file] == "TRAIN"
               || files[first_real_file] == "VALIDATE"
               || files[first_real_file] == "TEST")) ++first_real_file;

    if (first_real_file == files.size())
        throw Exception("Datasets::init(): no real files");
    

    /* Find if the dataset is sparse. */
    bool data_is_sparse = detect_sparseness(files[first_real_file]);

    /* Set up the feature space. */
    if (data_is_sparse) {
        sparse_feature_space.reset(new Sparse_Feature_Space());
        feature_space = sparse_feature_space;
    }
    else {
        dense_feature_space.reset(new Dense_Feature_Space());
        feature_space = dense_feature_space;
    }

    Disposition d = DSP_UNKNOWN;
    
    /* First, read in all of the data. */
    for (unsigned i = 0;  i < files.size();  ++i) {
        boost::timer timer;

        /* Look for a marker to change the disposition. */
        if (files[i] == "TRAIN") {
            d = DSP_TRAIN;
            continue;
        }
        else if (files[i] == "VALIDATE") {
            d = DSP_VALIDATE;
            continue;
        }
        else if (files[i] == "TEST") {
            d = DSP_TEST;
            continue;
        }

        //cerr << "i = " << i << "  files[i] = " << files[i] << endl;

        if (data_is_sparse) {
            data.push_back
                (make_sp
                 (new Sparse_Training_Data(files[i], sparse_feature_space)));
        }
        else {
            data.push_back
                (make_sp
                 (new Dense_Training_Data(files[i], dense_feature_space)));
        }

        dispositions.push_back(d);
        
        if (verbosity > 0) {
            cerr << "dataset \'" << files[i] << "\': ";
            if (d != DSP_UNKNOWN)
                cerr << "(" << d << "): ";
            cerr << data.back()->all_features().size() << " features, "
                 << data.back()->example_count() << " rows." << endl;
            //cerr << "label freq: " << data[i]->label_freq << endl;
        }
        
        if (profile)
            cerr << "[load dataset '" << files[i] << "': "
                 << timer.elapsed() << "s]" << endl;
    }
}

void
Datasets::
fixup_grouping(const std::vector<Feature> & groups)
{
    vector<float> offsets(groups.size(), 0.0);

    for (unsigned i = 0;  i < data.size();  ++i)
        data[i]->fixup_grouping_features(groups, offsets);
}

void
Datasets::
split(float training_split, float validation_split, float testing_split,
      bool randomize_order, const Feature & group_feature,
      const std::string & testing_filter)
{
    this->group_feature = group_feature;

    splits.resize(3);

    testing.clear();

    /* Get all datasets with an unknown disposition, which we then split
       up automatically. */
    vector<std::shared_ptr<Training_Data> > unknowns;

    for (unsigned i = 0;  i < data.size();  ++i) {
        switch (dispositions[i]) {
        case DSP_UNKNOWN:
            unknowns.push_back(data[i]);
            break;
        case DSP_TRAIN:
            if (!training)
                training = data[i];
            else training->add(*data[i]);
            break;
        case DSP_VALIDATE:
            if (!validation)
                validation = data[i];
            else validation->add(*data[i]);
            break;
            
        case DSP_TEST:
            if (testing.empty())
                testing.push_back(data[i]);
            else testing[0]->add(*data[i]);
            break;
        default:
            throw Exception("Datasets::split(): unknown dataset disposition");
        }
    }

    if (unknowns.empty()) {
        /* No "unknowns", so we can populate our training, testing and
           validation datasets from what was already there. */
        if (!training)
            throw Exception("No training datasets found anywhere");
        unknowns.push_back(training);
        if (validation)
            unknowns.push_back(validation);
    }

    //cerr << "training_split = " << training_split << "  validation_split = "
    //     << validation_split << "  testing_split = " << testing_split
    //     << endl;
    //cerr << "data.size() = " << data.size() << endl;

    if (unknowns.size() == 1 || validation_split > 0.0 || testing_split > 0.0) {
        splits[0] = training_split;
        splits[1] = validation_split;
        splits[2] = testing_split;

        cerr << "splits = " << splits << endl;

        vector<std::shared_ptr<Training_Data> > train_val_test
            = unknowns[0]->partition(splits, randomize_order, group_feature);
        
        training = train_val_test[0];
        validation = train_val_test[1];
        if (train_val_test[2]->example_count() > 0)
            testing.push_back(train_val_test[2]);

        copy(unknowns.begin() + 1, unknowns.end(), back_inserter(testing));
    }
    else {
        training = unknowns[0];
        if (unknowns.size() == 2) {
            /* Two datasets: training testing */
            validation = unknowns[0];
            testing.push_back(unknowns[1]);
        }
        else if (unknowns.size() >= 3) {
            /* Three datasets: train val test+ */
            validation = unknowns[1];
            copy(unknowns.begin() + 2, unknowns.end(), back_inserter(testing));
        }
    }

    
    if (testing_filter != "" && dense_feature_space) {
        /* Make a new testing set which is the other ones, filtered.  Used to
           see how we tested on an interesting subset of the data. */

        Feature_Set_Filter filter;
        filter.parse(testing_filter, *dense_feature_space);

        std::shared_ptr<Training_Data> new_data
            (new Training_Data(dense_feature_space));

        for (unsigned i = 0;  i < testing.size();  ++i) {
            const Training_Data & dataset = *testing[i];
            for (unsigned x = 0;  x < dataset.example_count();  ++x) {
                if (filter(dataset[x]))
                    new_data->add_example(dataset.share(x));
            }
        }

        new_data->index();

        testing.push_back(new_data);
    }

#if 0
    cerr << "datasets: " << endl;
    cerr << "TRAINING:" << endl;
    training->dump(cerr);
    if (validation) {
        cerr << endl << "VALIDATION:" << endl;
        validation->dump(cerr);
    }
    cerr << endl << "TESTING:" << endl;
    for (unsigned i = 0;  i < testing.size();  ++i) {
        cerr << "SET " << i << ": " << endl;
        testing[i]->dump(cerr);
        cerr << endl;
    }
#endif

    if (verbosity > 0) print_sizes();
}

void
Datasets::
reshuffle()
{
    bool randomize_partition = true;
    vector<std::shared_ptr<Training_Data> > train_val_test
        = data[0]->partition(splits, randomize_partition, group_feature);
    
    training   = train_val_test[0];
    validation = train_val_test[1];
    if (splits[2] > 0 && testing.size()) testing[0] = train_val_test[2];
}

void
Datasets::
print_sizes() const
{
    unsigned testing_rows = 0;
    for (unsigned i = 0;  i < testing.size();  ++i) {
        testing_rows += testing[i]->example_count();
    }
    
    cerr << "data sizes: training " << training->example_count()
         << "  validation: " << (validation ? validation->example_count() : 0UL)
         << "  testing: " << testing_rows << endl;
}

bool
Datasets::
detect_sparseness(const std::string & filename)
{
    filter_istream stream(filename);

    string s;
    stream >> s;

    return (stream && s.find("SPARSE") == 0);

#if 0
    const char * start_pos = context.get_pos();
    context.skip_line();

    string header(start_pos, context.get_pos() - 1);
    int num_header_vars = split_on_char(header, ' ').size();

    cerr << "header = " << header << endl;

    while (context && (!context.match_eol())) {
        if (context.match_literal('#')) {
            context.skip_line();
            continue;
        }
        else if (context.match_eol()) {
            continue;
        }
        else break;
    }
    
    


    start_pos = context.get_pos();
    context.skip_line();

    string line(start_pos, context.get_pos() - 1);

    cerr << "line = " << line << endl;

    int num_line_vars = split_on_char(line, ' ').size();

    return (num_header_vars != num_line_vars);
#endif
}

void
transform_dataset(std::shared_ptr<Training_Data> training,
                  const Feature_Transformer & transformer)
{
    if (!training) return;

    Training_Data new_data(transformer.output_fs());
    size_t nx = training->example_count();

    for (unsigned x = 0;  x < nx;  ++x)
        new_data.add_example(transformer.transform((*training)[x]));
    
    training->swap(new_data);
}

void
Datasets::
transform(const Feature_Transformer & transformer)
{
    transform_dataset(training, transformer);
    transform_dataset(validation, transformer);
    for (unsigned i = 0;  i < testing.size();  ++i)
        transform_dataset(testing[i], transformer);
}

} // namespace ML
