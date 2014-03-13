/* training_index_entry.h                                          -*- C++ -*-
   Jeremy Barnes, 23 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   An entry to the training data index.
*/

#ifndef __boosting__training_index_entry_h__
#define __boosting__training_index_entry_h__


#include "config.h"
#include "training_index.h"
#include "feature_map.h"
#include "jml/arch/threads.h"
#include "jml/math/xdiv.h"
#include <boost/utility.hpp>


namespace ML {


/*****************************************************************************/
/* DATASET_INDEX::INDEX_ENTRY                                                */
/*****************************************************************************/

/** This structure is a single entry of the index.  It contains all of the
    information and indexes for a given feature.
*/

struct Dataset_Index::Index_Entry {
    Index_Entry();

    Lock lock;  // for when we store things that aren't there

    bool used;  ///< True if we use this entry
    bool initialized;

    Feature feature;
    std::shared_ptr<const Feature_Space> feature_space;

    unsigned example_count;     ///< Number of examples in the feature set
    unsigned seen;              ///< Number of times this feature seen
    unsigned found_in;          ///< Number of examples feature is found
    unsigned missing_from;      ///< Number of examples feature is missing 
    unsigned found_twice;       ///< Number of examples more than one instance
    unsigned zeros;             ///< Number of times value zero found
    unsigned ones;              ///< Number of times value one found
    unsigned non_integral;      ///< Number of non-integral values found
    float  max_value;           ///< Maximum non-missing value found
    float  min_value;           ///< Minimum non-missing value found
    unsigned last_example;      ///< Last example number we were found in
    unsigned in_this_ex;        ///< Number of times in this example

    /** Print a string containing the information above. */
    std::string print_info() const;
    
    /** Check that this feature is used before we access it. */
    void check_used() const;

    /// If true, was found once or more in every examp.
    bool dense() const
    {
        return missing_from == 0;
    }      

    /// If true, was found zero or one in each examp.
    bool only_one() const
    {
        return found_twice == 0;
    }    

    /// If true, was found exactly once each examp.
    bool exactly_one() const
    {
        return seen == example_count && found_in == example_count
            && dense() && only_one();
    }

    /// Return the average number of times this feature occurs per example
    double density() const
    {
        return xdiv<double>(seen, example_count);
    }
    
    /** Contains a list of each of the examples that the feature was
        found in.  Will be empty if dense and exactly_one are both true,
        since it can be produced with the iota function.
    */
    std::vector<unsigned> examples;

    /** Ditto, but sorted by value. */
    bool has_examples_sorted;
    std::vector<unsigned> examples_sorted;
    
    /** Contains the value each time it was found.  Sorted by example number. */
    std::vector<float> values;
    
    /** Contains the same values as values, but sorted by the value itself. */
    bool has_values_sorted;
    std::vector<float> values_sorted;

    /** Counts per example. */
    bool has_counts;
    std::vector<unsigned> counts;

    /** Counts per example, sorted by the value. */
    bool has_counts_sorted;
    std::vector<unsigned> counts_sorted;

    /** Divisors per example. */
    bool has_divisors;
    std::vector<float> divisors;

    /** Divisors per example, sorted by the value. */
    bool has_divisors_sorted;
    std::vector<float> divisors_sorted;

    /** Contains this feature mapped as a label.  Sorted by example number. */
    bool has_labels;
    std::vector<Label> labels;
    
    /** Ditto, but sorted by the value. */
    bool has_labels_sorted;
    std::vector<Label> labels_sorted;
    
    /** Contains the frequency distribution of values, if it has been
        requested. */
    bool has_freqs;
    Freqs freqs;

    /** Contains the categorical frequency distribution. */
    bool has_category_freqs;
    Category_Freqs category_freqs;

    struct Mapped_Labels_Entry : public std::vector<Label> {
        bool initialized;
    };

    /** Labels for something else, mapped onto our example distribution. */
    Feature_Map<Mapped_Labels_Entry> mapped_labels;

    /** Ditto, but sorted by example. */
    Feature_Map<Mapped_Labels_Entry> mapped_labels_sorted;

    /** Buckets, one entry for each total number of buckets (cached). */
    map<unsigned, Bucket_Info> bucket_info;


    /*************************************************************************/
    /* INITIALIZATION                                                        */
    /*************************************************************************/

    /** Add the given value and example number to the feature index.  The
        example_count and sparse fields are used to improve memory management.
    */
    void insert(float value, unsigned example, unsigned example_count,
                bool sparse, const Feature_Set & fset);

    /** Copy the data structures to allow unused space on the end of vectors
        to be reclaimed. */
    void finalize(unsigned example_count, const Feature & feature,
                  std::shared_ptr<const Feature_Space> feature_space);


    /*************************************************************************/
    /* INDEX GENERATION                                                      */
    /*************************************************************************/

    /** Return the values, sorted as specified. */
    const vector<float> & get_values(Sort_By sort_by);

    /** Return the example numbers, sorted as specified.  If the vector is
        empty, then the examples count implicitly from one to the highest
        value. */
    const vector<unsigned> & get_examples(Sort_By sort_by);

    /** Get the example counts.  If the vector is empty, then the counts are
        implicitly one every time. */
    const vector<unsigned> & get_counts(Sort_By sort_by);

    /** Get the divisors.  If the vector is empty, then the divisor is
        implicitly one every time. */
    const vector<float> & get_divisors(Sort_By sort_by);

    /** Get the frequencies.  These are extracted from the values array. */
    const Freqs & get_freqs();

    /** Get the category frequencies.  These are extracted from Freqs. */
    const Category_Freqs & get_category_freqs(size_t num_categories);

    /** Get the labels: one per example.  The distribution will have the same
        length as example_count. */
    const vector<Label> & get_labels();

    /** Map the labels from something else onto our examples.  The distribution
        will have the same length as examples.size().  The labels are sorted in
        the manner specified.
    */
    const vector<Label> &
    get_mapped_labels(const vector<Label> & labels, const Feature & feature,
                      Sort_By sort_by);


    /*************************************************************************/
    /* BUCKETING                                                             */
    /*************************************************************************/

    /* Bucketing algorithm.

       This code deals with the creation of "buckets" in the training
       data.  There are two purposes of buckets:

       1.  They reduce the tendancy to overfit.  Without buckets, a
           real variable over a 100,000 example dataset could have
           100,000 different values.  Each of these gets tested
           individually, leading to a propensity to fine-tune the
           split values very closely (which can lead to overfitting).
           With buckets, these 100,000 values will be put into 1000
           (or another number of) buckets, and only 1000 split points
           will be tested.  As these split points are pre-computed,
           they will not be optimised to the point of overfitting.  Small
           improvements in accuracy over a very large dataset have been
           noted using buckets.

           Note that this doesn't hold for when there are less distinct
           values of the variable than buckets (for example, in boolean
           variables).  However, they will still be put in buckets as
           point 2 below still holds.

       2.  They speed up the training, by allowing the weights array to
           be traversed in-order.  Without buckets, the examples are seen
           in order of the value of the variable, and thus the weights
           array is accessed in a random order.  For a very large dataset,
           the weights array does not fit in cache, which leads to a
           main memory access for each of the examples for each split
           point for each variable.  When buckets are used, the weights
           array is traversed in-order, and can thus be streamed from
           memory (using prefetching, etc as appropriate).  For very
           large datasets, speedups of 5-10x have been noted.
                  
           The n buckets are created by dividing the range of the given
           variable into n disjoint regions.  The boundary values between
           the bucket numbers are kept.

           The algorithm for choosing the disjoint regions tries to put the
           same number of examples into each bucket if possible, so that
           regions of the range where there are lots of examples tend to have
           buckets closer together, and outliers tend to be clustered together
           (rather than be in buckets all by themselves).
    */

    const Bucket_Info & create_buckets(size_t num_buckets);

    const Bucket_Info & buckets(size_t num_buckets);
};

} // namespace ML


#endif /* __boosting__training_index_entry_h__ */


