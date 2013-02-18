/* dense_feature_space.h                                           -*- C++ -*-
   Jeremy Barnes, 12 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   A feature space based upon dense features.
*/

#ifndef __boosting__dense_features_h__
#define __boosting__dense_features_h__


#include "config.h"
#include "feature_set.h"
#include "feature_space.h"
#include "feature_info.h"
#include "training_data.h"
#include <boost/multi_array.hpp>
#include <map>


namespace ML {

/*****************************************************************************/
/* DENSE_FEATURE_MAPPING                                                     */
/*****************************************************************************/

/** Holds a mapping between two dense feature spaces.  This includes how
    to map variables onto each other, and how to map the values of
    categorical variables onto each other.
*/
struct Dense_Feature_Mapping {
    Dense_Feature_Mapping() : initialized_(false) {}
    std::vector<int> vars;
    bool initialized_;
    int num_vars_expected_;
    std::vector<std::shared_ptr<Categorical_Mapping> > categories;
    bool initialized() const;
    void clear();
};


        
/*****************************************************************************/
/* DENSE_FEATURE_SPACE                                                       */
/*****************************************************************************/

/** This is a feature space that has a fixed number of dense variables, and
    always returns a vector of this number of real features for each feature
    set.  It is essentially an adaptor around the "file-o-data" feature
    representation, used by the more classical algorithms.
*/

class Dense_Feature_Space : public Mutable_Feature_Space {
public:
    Dense_Feature_Space();
    Dense_Feature_Space(DB::Store_Reader & store);
    Dense_Feature_Space(size_t feature_count);
    Dense_Feature_Space(const std::vector<std::string> & feature_names);
    virtual ~Dense_Feature_Space();

    /** Initialise, given the array of feature names.  Default info gives
        a REAL variable.  This should work fine, except for categorical
        variables. */
    void init(const std::vector<std::string> & feature_names,
              Feature_Type type = REAL);

    /** Initialise, given the array of feature names and the associated
        info. */
    void init(const std::vector<std::string> & feature_names,
              const std::vector<Mutable_Feature_Info> & feature_info);

    /** Encode the given parameter vector into a feature set. */
    std::shared_ptr<Mutable_Feature_Set>
    encode(const std::vector<float> & variables) const;

    /** Encode the given parameter vector into a feature set for another
        dense feature space.  Takes care of mapping the variables onto each
        other via their variable names. */
    std::shared_ptr<Mutable_Feature_Set>
    encode(const std::vector<float> & variables,
           const Dense_Feature_Space & other) const;

    typedef Dense_Feature_Mapping Mapping;

    /** Create a mapping to go from the other feature space to this feature
        space.  This can be used to encode a feature vector (created in a
        problem feature space) into a feature vector for the classifier
        feature space.  The classifier feature space may be different from
        the problem feature space in the following circumstances:

        * When there were extra features added for the training of the
          classifier.  For example, if there were label, weighting or grouping
          features added for the training (these features don't make sense
          when running, as opposed to training, the classifier).
        * When extra features have been added to the problem feature space.
          For example, if new features were added but we still want to run an
          old classifier that was trained without these features for
          comparison.

        Note that it is easy to confuse the two classifiers.

        For example:

        Classifier classifier;
        classifier.load("classifier.cls");
        std::shared_ptr<const Dense_Feature_Space>
            problem_fs = feature_space();
        std::shared_ptr<const Dense_Feature_Space>
            classifier_fs = classifier.feature_space<ML::Dense_Feature_Space>();

        classifier_fs->create_mapping(*problem_fs, mapping);

        distribution<float> features = ...;

        std::shared_ptr<Mutable_Feature_Set> encoded
            = classifier_fs->encode(features, *problem_fs, mapping);

        float score = classifier.predict(1, *encoded);
    */
    void create_mapping(const Dense_Feature_Space & other,
                        Mapping & mapping) const;

    /** Encode the given parameter vector (in the other feature space) into
        a feature set suitable for this classifier.

        Creates the mapping if necessary (see create_mapping).

        The extra map argument is a map which will be used to cache the
        important information (which should make it substantially faster to
        perform the mapping).  It will be filled in the first time called.
        It is the caller's responsibility to clear the map each time that
        either of the Dense_Feature_Space objects change.
    */
    std::shared_ptr<Mutable_Feature_Set>
    encode(const std::vector<float> & variables,
           const Dense_Feature_Space & other, Mapping & mapping) const;

    /** Encode the given parameter vector (in the other feature space) into
        a feature set suitable for this classifier.

        The mapping argument must have already been initialized with
        create_mapping.  Make sure that the parameters are in the right
        order (see create_mapping).
    */
    std::shared_ptr<Mutable_Feature_Set>
    encode(const std::vector<float> & variables,
           const Dense_Feature_Space & other, const Mapping & mapping) const;

    /** Encode in an optimized way such that memory isn't allocated.  The
        mapping and the info should have been already allocated.  The features
        are assumed to contain the correct number of variables for the
        mapping (mapping.num_vars_expected_), and the output is expected to
        contain at least n values.  It will leave output in a state that will
        allow Classifier_Impl::optimized_predict_impl to be called with the
        array.
    */
    void
    encode(const float * features,
           float * output,
           const Dense_Feature_Space & other,
           const Mapping & mapping) const;
    
    /** Decode the results of an encode step. */
    distribution<float> decode(const Feature_Set & feature_set) const;

    using Feature_Space::serialize;
    using Feature_Space::reconstitute;
    using Feature_Space::print;
    using Feature_Space::parse;

    virtual Feature_Info info(const Feature & feature) const;
    virtual void set_info(const Feature & feature, const Feature_Info & info);
    virtual std::string print(const Feature & feature) const;
    virtual std::string print(const Feature & feature, float value) const;
    virtual void parse(const std::string & name, Feature & feature) const;
    virtual bool parse(Parse_Context & context, Feature & feature) const;
    virtual void expect(Parse_Context & context, Feature & feature) const;
    virtual void serialize(DB::Store_Writer & store,
                           const Feature & feature) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              Feature & feature) const;

    /** Methods to deal with a feature set. */
    virtual std::string print(const Feature_Set & fs) const;
    virtual void serialize(DB::Store_Writer & store,
                           const Feature_Set & fs) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              std::shared_ptr<Feature_Set> & fs) const;

    /* Methods to deal with the feature space as a whole. */
    virtual Dense_Feature_Space * make_copy() const;
    virtual std::string print() const;

    /** Serialization and reconstitution. */
    virtual std::string class_id() const;
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);
    virtual void reconstitute(DB::Store_Reader & store,
                              const std::shared_ptr<const Feature_Space>
                                  & feature_space);

    /** Return the total number of variables in the dense feature array. */
    size_t variable_count() const { return info_array.size(); }

    /** Return the list of features. */
    const std::vector<Feature> & features() const { return features_; }

    const std::vector<Feature> & dense_features() const { return features_; }
    
    /** Return an array with the names of each feature. */
    const std::vector<std::string> & feature_names() const;

    /** Set the names of each feature. */
    void set_feature_names(const std::vector<std::string> & names);

    /** Find the index of a feature given its name.  A result of -1 means
        that it was not found. */
    int feature_index(const std::string & name) const;

    /** Add the given feature to the feature space.  Returns its index. */
    int add_feature(const std::string & name, const Feature_Info & info);

    /** Add the entire feature space */
    void add(const Dense_Feature_Space & other_fs,
             const std::string & name_prefix = "");

    virtual Type type() const { return DENSE; }

    /** Turn the feature info for the given feature into a mutable categorical
        version. */
    std::shared_ptr<Mutable_Categorical_Info>
    make_categorical(Feature feature);

    /** Create a new feature with the given name and the given feature info.
        If one with the name already exists, it is returned instead and the
        info isn't set.
    */
    virtual Feature
    make_feature(const std::string & name,
                 const Feature_Info & info = UNKNOWN);

    /** Return the feature with the given name.  Throws if the name is
        unknown. */
    virtual Feature get_feature(const std::string & name) const;

    virtual void freeze();

protected:    
    std::vector<std::string> names_fwd;
    std::map<std::string, int> names_bwd;
    std::vector<Mutable_Feature_Info> info_array;
    std::vector<Feature> features_;

    friend class Dense_Training_Data;
};


/*****************************************************************************/
/* DENSE_TRAINING_DATA                                                       */
/*****************************************************************************/

/** The training data component.  Allows us to train our boosting algorithm
    on the same data that all the other algorithms get.
*/

class Dense_Training_Data : public Training_Data {
public:
    /** Default do-nothing constructor. */
    Dense_Training_Data();

    /** Initialise from a filename.  Loads it into a dataset. */
    Dense_Training_Data(const std::string & filename);

    /** Initialise from a filename.  Loads it into a dataset.  The dataset
        must already know how to modify the variables.
    */
    Dense_Training_Data(const std::string & filename,
                        std::shared_ptr<const Feature_Space> feature_space);

    /** Initialise from a filename.  This one will modify the feature space
        to make its data also fit the feature space, or throw an exception if
        it cannot.  This is primarily useful when loading a series of
        datasets. */
    Dense_Training_Data(const std::string & filename,
                        std::shared_ptr<Dense_Feature_Space> feature_space);

    virtual ~Dense_Training_Data();


    /** Initialise from a data file. */
    void init(const std::string & filename);

    /** Initialize from the text of a training data file */
    void init(const char * data,
              const char * data_end);

    /** Initialise from the data file, using the given feature space. */
    void init(const std::string & filename,
              std::shared_ptr<const Feature_Space> feature_space);

    /** Initialise from a filename.  This one will modify the feature space
        to make its data also fit the feature space, or throw an exception if
        it cannot.  This is primarily useful when loading a series of
        datasets. */
    void init(const std::string & filename,
              std::shared_ptr<Dense_Feature_Space> feature_space);

    /** Initialise from a set of filenames.  This loads each of the files in
        turn and initialises from the sum of all of the files. */
    void init(const std::vector<std::string> & filenames,
              std::shared_ptr<Dense_Feature_Space> feature_space);

    /** Initialize from the text of a training data file */
    void init(const char * data,
              const char * data_end,
              std::shared_ptr<Dense_Feature_Space> feature_space);

private:
    struct Data_Source;

    /** Initialise from a set of filenames.  This loads each of the files in
        turn and initialises from the sum of all of the files. */
    void init(const std::vector<Data_Source> & data_sources,
              std::shared_ptr<Dense_Feature_Space> feature_space);

public:
    /** Polymorphic copy. */
    virtual Dense_Training_Data * make_copy() const;

    /** Polymorphic construct.  Makes another object of the same type, but
        doesn't populate it. */
    virtual Dense_Training_Data * make_type() const;

    size_t variable_count() const { return dataset.shape()[1]; }

    virtual size_t row_offset(size_t row) const;

    virtual std::string row_comment(size_t row) const;

    /** Modify the value of the given feature in the given example number
        to the new value given.  Returns the old value. */
    virtual float modify_feature(int example_number,
                                 const Feature & feature,
                                 float new_value);

protected:
    /** Dense version of the dataset. */
    boost::multi_array<float, 2> dataset;

    /** The comment attached to each of the examples. */
    std::vector<std::string> row_comments;

    /** The offset from the start of the file for the start of the line for
        each of the examples in the file. */
    std::vector<size_t> row_offsets;
    
    /** Add the data from the dataset to the Training_Data structures so they
        can be indexed.  Usually called after all files have been read. */
    void add_data();
};


/*****************************************************************************/
/* DENSE_FEATURE_SET                                                         */
/*****************************************************************************/

/** This is a specialization of a feature set that stores itself as a dense
    array.
*/

class Dense_Feature_Set : public Feature_Set {
public:
    Dense_Feature_Set(std::shared_ptr<const std::vector<Feature> > features,
                      const float * values)
    : features(features), values(values)
    {
    }

    virtual ~Dense_Feature_Set() {}

    virtual boost::tuple<const Feature *, const float *, int, int, size_t>
    get_data(bool need_sorted = false) const
    {
        return boost::make_tuple
            (&(*features)[0], values, sizeof(Feature), sizeof(float), 
             features->size());
    }

    virtual void sort()
    {
    }

    std::shared_ptr<const std::vector<Feature> > features;
    const float * values;

    virtual Dense_Feature_Set * make_copy() const;
};


} // namespace ML



#endif /* __boosting__dense_features_h__ */
