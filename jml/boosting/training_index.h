/* training_index.h                                                -*- C++ -*-
   Jeremy Barnes, 16 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Index of training data.
*/

#ifndef __boosting__training_index_h__
#define __boosting__training_index_h__


#include "config.h"
#include "jml/db/persistent.h"
#include "feature_set.h"
#include "feature_info.h"
#include "jml/utils/hash_map.h"
#include <boost/shared_ptr.hpp>
#include "training_index_iterators.h"
#include "buckets.h"

namespace ML {


/** This is the default number of buckets to use.  This tells us how many
    distinct buckets to turn a continuous variable into when doing the
    bucketizing part of finish().
    
    It is used for real variables only, not for other types.
*/
static const int DEFAULT_NUM_BUCKETS = 512;


/*****************************************************************************/
/* DATASET_INDEX                                                             */
/*****************************************************************************/

/** Enum describing in which order the index iterator will iterate. */
enum Sort_By {
    BY_EXAMPLE,    ///< Example number is increasing
    BY_VALUE       ///< Value of the feature is increasing
};

/** Enum describing what we want in the index */
enum Index_Contents {
    IC_VALUE   = 1 << 0,  //< Feature values are in index
    IC_BUCKET  = 1 << 1,  //< Bucket numbers are in index
    IC_LABEL   = 1 << 2,  //< Labels are in index
    IC_EXAMPLE = 1 << 3,  //< Example numbers are in index
    IC_COUNT   = 1 << 4,  //< Example counts are in index
    IC_DIVISOR = 1 << 5,  //< Divisors (1 / count) are in index

    IC_STRICT  = 1 << 31  //< Strict (don't put anything not needed)
};

/** An index that tells us which features are found in which example. */

class Dataset_Index {
public:
    /** Initialise the index from a dataset.  This will create the bare
        minimum index as fast as possible, by scanning each of the feature
        sets one time.  The rest of the indexes (which are more expensive)
        are created on demand from this information.
    */
    void init(const Training_Data & data,
              const std::vector<Feature> & features
                  = std::vector<Feature>());

    /** Initialise the index from a dataset, but only for the given features.
        An attempt to access any other feature will fail.
    */
    void init(const Training_Data & data,
              const Feature & label,
              const std::vector<Feature> & features);

    /** The type of the frequency distribution. */
    typedef sparse_distribution<float, float,
                                sorted_vector<float, float> > Freqs;

    /** Returns the frequency distribution for the given feature. */
    const Freqs & freqs(const Feature & feature) const;

    /** Type of a category frequency distribution. */
    typedef distribution<float> Category_Freqs;

    /** For a category-based feature (CATEGORICAL or BOOLEAN), gives the
        frequency for each value. */
    const Category_Freqs & category_freqs(const Feature & feature) const;

    /** Returns the label distribution for the given feature.  Note that this
        is in example order, and will contain exactly one entry for each
        example.
    */
    const std::vector<Label> & labels(const Feature & feature) const;

    /** Return the values, in example order. */
    const std::vector<float> & values(const Feature & feature) const;
    
    /** Returns an index of the joint distribution between any two given
        features.

        \param target       The target feature (dependent).  This will take
                            the place of the label() field.
        \param independent  The independent feature.  This takes the place of
                            the value() field.
        \param sort_by      Controls in which order the index is iterated.
                            This can either be to go by example number, or to
                            go by the value of the bucket.
        \param buckets      If non-zero, then the iterator will be over the
                            specified number of buckets, and the bucket()
                            (instead of value()) field can be used.

        \returns            An Index_Iterator which will enumerate the values
                            of the joint distribution according to the
                            specification given.
    */
    Joint_Index
    joint(const Feature & target,
          const Feature & independent,
          Sort_By sort_by,
          unsigned contents,
          size_t buckets = 0) const;

    /** Returns an index between a feature and itself.  This allows us to
        do things based upon a feature's distribution. */
    Joint_Index
    dist(const Feature & feature,
         Sort_By sort_by,
         unsigned contents,
         size_t buckets = 0) const;

    /** Guesses the feature information given the given feature.  Tries not to
        generate more of an index than is necessary to identify the feature.

        \param feature      The feature for which we are trying to guess the
                            info.

        \returns            The best guess at the Feature_Info for the object.
    */
    Feature_Info guess_info(const Feature & feat) const;

    Feature_Info guess_info_categorical(const Feature & feat) const;

    /** Tells us the density of a given feature. */
    double density(const Feature & feat) const;

    /** Tells us if the feature occurs exactly once in each example. */
    bool exactly_one(const Feature & feat) const;

    /* If true, was found once or more in every example. */
    bool dense(const Feature & feat) const;

    /* If true, was found zero or one in each example. */
    bool only_one(const Feature & feat) const;

    /** Tells us in how many examples this feature was seen. */
    size_t count(const Feature & feat) const;

    /** Tells us the range (minimum and maximum values) of the feature. */
    std::pair<float, float> range(const Feature & feature) const;

    /** Tells us if the feature is constant over the whole dataset */
    bool constant(const Feature & feat) const;

    /** Tells us if the feature is integral (has only integral values). */
    bool integral(const Feature & feature) const;

    /** Return a list of all features found in the dataset. */
    const std::vector<Feature> & all_features() const;

private:
    struct Itl;
    struct Index_Entry;
    std::shared_ptr<Itl> itl;
};


} // namespace ML

#endif /* __boosting__training_index_h__ */

