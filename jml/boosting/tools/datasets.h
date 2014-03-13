/* datasets.h                                                      -*- C++ -*-
   Jeremy Barnes, 28 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Structure to deal with loading and partitioning datasets for the various
   boosting tools.
*/

#ifndef __boosting_tools__datasets_h__
#define __boosting_tools__datasets_h__

#include "jml/boosting/config.h"
#include <vector>
#include "jml/boosting/training_data.h"
#include "jml/boosting/feature_space.h"
#include "jml/stats/distribution.h"


namespace ML {

class Feature_Transformer;
class Sparse_Feature_Space;
class Dense_Feature_Space;

/** Tells us about what the user wanted to use the dataset for. */
enum Disposition {
    DSP_TRAIN,      ///< Dataset is a training set
    DSP_VALIDATE,   ///< Dataset is a validation set
    DSP_TEST,       ///< Dataset is a testing set
    DSP_UNKNOWN     ///< No disposition was given
};

std::ostream & operator << (std::ostream & stream, Disposition d);    


/*****************************************************************************/
/* DATASETS                                                                  */
/*****************************************************************************/

class Datasets {
public:
    Datasets();

    void init(const std::vector<std::string> & files,
              int verbosity, bool profile);

    void fixup_grouping(const std::vector<Feature> & groups);

    void split(float training_split, float validation_split,
               float testing_split, bool randomize_order,
               const Feature & group_feature,
               const std::string & testing_filter = "");

    void reshuffle();

    void print_sizes() const;

    /** Apply the given feature transformer to each of the datasets.
        TODO: pass in a way to know over which of the datasets the
        feature transformer was trained...
    */
    void transform(const Feature_Transformer & transformer);
    
    /** Overall feature space. */
    std::shared_ptr<Mutable_Feature_Space> feature_space;
    std::shared_ptr<Sparse_Feature_Space> sparse_feature_space;
    std::shared_ptr<Dense_Feature_Space> dense_feature_space;
    std::vector<std::shared_ptr<Training_Data> > data;
    std::vector<Disposition> dispositions;
    
    /* Broken down and split up datasets.  First one is training, second is
       validation, any after that are testing. */
    std::shared_ptr<Training_Data> training;
    std::shared_ptr<Training_Data> validation;
    std::vector<std::shared_ptr<Training_Data> > testing;

    Feature group_feature;

    distribution<float> splits;

    int verbosity;
    bool profile;

    static bool detect_sparseness(const std::string & filename);

};




} // namespace ML



#endif /* __boosting_tools__datasets_h__ */
