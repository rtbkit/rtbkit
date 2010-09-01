/* feature_set.i                                                   -*- C++ -*-
   Jeremy Barnes, 24 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   SWIG wrapper for the Training_Data class.
*/

%module jml 
%{
#include "jml/boosting/training_data.h"
%}

%include "std_vector.i"


namespace ML {


class Dataset_Index;


/*****************************************************************************/
/* TRAINING_DATA                                                             */
/*****************************************************************************/

/** This is a class that rearranges a training set to be usable by the word
    expert. */

class Training_Data {
public:
    Training_Data();
    //Training_Data(const Training_Data & other);
    Training_Data(boost::shared_ptr<const Feature_Space> feature_space);
    virtual ~Training_Data();

    Training_Data & operator = (const Training_Data & other)
    {
        if (&other == this) return *this;
        Training_Data result(other);
        swap(result);
        return *this;
    }
    
    /** Initialise with the given feature space and number of labels. */
    void init(boost::shared_ptr<const Feature_Space> feature_space);
    
    /** Clear the training data completely. */
    void clear();
    
    /** Swap with another Training_Data object.  Guaranteed not to throw. */
    void swap(Training_Data & other);
    
    /** Return a list of all features that are defined anywhere within the
        space.  Can be very long.  Is not cached; thus it may take a while
        to calculate.
    */
    std::vector<Feature> all_features() const;
    
    size_t example_count() const { return data_.size(); }

    bool empty() const { return example_count() == 0; }
    
    /** What is the feature space? */
    boost::shared_ptr<const Feature_Space> feature_space() const
    {
        return feature_space_;
    }
    
    /** Dump the training data to a text file.  Exactly what is the format is
        determined by the feature space, but there will be the following
        general format:
        1 line: FS information
        the rest: feature sets (label, then the result of the print routine
        for each training_set)
    */
    void dump(const std::string & filename) const;
    
    void dump(std::ostream & stream) const;

    /** Serialize to the given store. */
    void serialize(DB::Store_Writer & store) const;

    /** Save to the given filename. */
    void save(const std::string & filename) const;
    
    /** Reconstitute from the given store. */
    void reconstitute(DB::Store_Reader & store);
    
    /** Load from the given (binary) filename */
    void load(const std::string & filename);

    /** Access the Example_Data for a single example. */
    const Feature_Set & operator [] (int example) const
    {
        return *data_.at(example);
    }
    
    /** Access the pointer to a given example data.  Used when we need to
        get a new reference to the data. */
    boost::shared_ptr<const Feature_Set> get(int example) const
    {
        return data_.at(example);
    }

    /** Access the pointer to a given example data.  Used when we need to
        get a new reference to the data to share between two datasets.
    */
    boost::shared_ptr<Feature_Set> share(int example) const
    {
        return data_.at(example);
    }

    /** Access the pointer to a given example data.  Used when we want to
        modify the data.  It causes all indexes to be released.
    */
    boost::shared_ptr<Feature_Set> & modify(int example);

    /** Partition this into a set of disjoint training data objects, with
        relative amounts given by the vector of sizes.
        
        \param sizes        a vector of relative sizes.  These control which
                            proportion of data goes in which output training
                            data object.
        \param random       if \c true, then the dataset is assigned in random
                            order.  If false, then it will be split up so that
                            the combined outputs have the same order as the
                            input dataset.
        \returns            a vector of datasets (formed by make_empty())
                            containing the data.
        \pre                All elements of sizes are >= 0 and not Inf or NaN
    */
    virtual std::vector<boost::shared_ptr<Training_Data> >
    partition(const std::vector<float> & sizes, bool random = true,
              const Feature & group_feature = MISSING_FEATURE) const;

    /** Add the given Training_Data object onto the end of this one.
        \param merge_index  If true, we also merge their indexes so that the
                            new data contains the same indexes as the old.
    */
    void add(const Training_Data & other, bool merge_index = false);

    /** Given a pointer to an Example_Data
        object (or a derivation of one), this method will add it to all of
        the indexes, etc.

        Returns the example number of this example.
    */
    int add_example(const boost::shared_ptr<Feature_Set> & example);

    /** Fix up any grouping features.  We ensure that they are strictly
        increasing.  Makes sure that they all exceed the given offset. */
    virtual void
    fixup_grouping_features(const std::vector<Feature> & group_features,
                            std::vector<float> & offsets);

    /** Polymorphic copy. */
    virtual Training_Data * make_copy() const;

    /** Polymorphic construct.  Makes another object of the same type, but
        doesn't populate it. */
    virtual Training_Data * make_type() const;

    /** Return the group counts. */
    //const sparse_distribution<float, float> group_counts() const;

    /** Return the label count (according to the feature space) to predict the
        given feature. */
    size_t label_count(const Feature & predicted) const;

    /** Return the offset from the start of the data file for this row.
        Default throws an exception. */
    virtual size_t row_offset(size_t row) const;

    /** Return any text comment attached to this row.
        Default throws an exception. */
    virtual std::string row_comment(size_t row) const;

    /** Modify the value of the given feature in the given example number
        to the new value given.  Returns the old value. */
    virtual float modify_feature(int example_number,
                                 const Feature & feature,
                                 float new_value);


    /*************************************************************************/
    /* INDEXES                                                               */
    /*************************************************************************/

    /** Pre-index the training data to predict the given feature using the
        given set of features. */
    void preindex(const Feature & label, const std::vector<Feature> & features);

    /** Pre-index the training data to predict the given feature using all
        features except for the label feature itself. */
    void preindex(const Feature & label);

    /** Pre-index the features in the dataset. */
    void preindex_features();
    
    /** Access the index for the given feature. */
    const Dataset_Index & index() const
    {
        //Guard guard(index_lock);
        if (!dirty_ && index_) return *index_;
        return generate_index();
    }

protected:
    /** Data for all our examples. */
    typedef std::vector<boost::shared_ptr<Feature_Set> > data_type;
    data_type data_;

    mutable Lock index_lock;
    mutable boost::shared_ptr<Dataset_Index> index_;

    boost::shared_ptr<const Feature_Space> feature_space_;

    /** Are the indexes dirty?  Occurs when we could have mutated one of the
        entries...
    */
    mutable bool dirty_;

    /** Notify that the given feature needs to be reindexed */
    void notify_needs_reindex(const Feature & feature)
    {
        dirty_ = true;
    }

    const Dataset_Index & generate_index() const;
};

} // namespace ML

