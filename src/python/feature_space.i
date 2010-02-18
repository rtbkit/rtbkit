/* feature_space.i                                                 -*- C++ -*-
   Jeremy Barnes, 24 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   SWIG wrapper for the Feature_Space class.
*/

%module jml 

%{
#include "jml/boosting/feature_space.h"
#include "jml/boosting/dense_features.h"
#include "jml/stats/distribution.h"

// Avoid compilation errors due to SWIG being confused by the using statement
// in distribution.h
using ML::Stats::distribution;
%}

%include "std_vector.i"
%include "shared_ptr.i"

%template(svector) std::vector<std::string>;
%template(ivector) std::vector<int>;
%template(Mutable_Feature_Info_Vector) std::vector<ML::Mutable_Feature_Info>;
%template(Feature_Vector) std::vector<ML::Feature>;


namespace ML {
namespace Stats {

template<typename F, class Underlying= std::vector<F> >
class distribution;

} // namespace Stats

using Stats::distribution;


enum Feature_Space_Type {
    DENSE,   ///< Dense feature space
    SPARSE   ///< Sparse feature space
};


/*****************************************************************************/
/* FEATURE_SPACE                                                             */
/*****************************************************************************/

/** This is a class that provides information on a space of features. */

class Feature_Space {
public:
    virtual ~Feature_Space();

    virtual Feature_Info info(const Feature & feature) const = 0;
    virtual std::string print(const Feature & feature) const;
    virtual std::string print(const Feature & feature, float value) const;
    virtual bool parse(Parse_Context & context, Feature & feature) const;
    virtual void parse(const std::string & name, Feature & feature) const;
    virtual void expect(Parse_Context & context, Feature & feature) const;

    virtual void serialize(DB::Store_Writer & store,
                           const Feature & feature) const;

    virtual void reconstitute(DB::Store_Reader & store,
                              Feature & feature) const;
    virtual void serialize(DB::Store_Writer & store,
                           const Feature & feature,
                           float value) const;

    virtual void reconstitute(DB::Store_Reader & store,
                              const Feature & feature,
                              float & value) const;


    virtual std::string print(const Feature_Set & fs) const;
    virtual void serialize(DB::Store_Writer & store,
                           const Feature_Set & fs) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              boost::shared_ptr<Feature_Set> & fs) const;


    virtual std::string class_id() const = 0;

    virtual Feature_Space_Type type() const = 0;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const boost::shared_ptr<const Feature_Space> & fs);
    
    virtual Feature_Space * make_copy() const = 0;

    virtual std::string print() const;

    virtual boost::shared_ptr<Training_Data>
    training_data(const boost::shared_ptr<const Feature_Space> & fs) const;

    virtual void freeze();
};


/*****************************************************************************/
/* MUTABLE_FEATURE_SPACE                                                     */
/*****************************************************************************/

/** A feature space that can be modified. */

class Mutable_Feature_Space : public Feature_Space {
public:
    
    virtual ~Mutable_Feature_Space();

    /** Set the information for how to use a feature.
        \param feature       The feature for which we are setting the
                             information.
        \param info          The new information.

        Note that feature spaces are not required to implement this method.
        The default implementation will throw an exception; any classes
        which do implement it (at least Dense_Feature_Space and
        Sparse_Feature_Space) need to override it.

        Throws an exception if the feature was not found.
    */
    virtual void set_info(const Feature & feature, const Feature_Info & info) = 0;

    /** Create a new feature with the given name and the given feature info.
        If one with the name already exists, it is returned instead and the
        info isn't set.
    */
    virtual Feature
    make_feature(const std::string & name,
                 const Feature_Info & info = UNKNOWN) = 0;

    /** Return the feature with the given name.  Throws if the name is
        unknown. */
    virtual Feature get_feature(const std::string & name) const = 0;

    /** Import another feature space.  This is capable of converting between
        different types of feature spaces; for example of converting sparse
        into dense or vice versa.
    */
    virtual void import(const Feature_Space & from);

    virtual Mutable_Feature_Space * make_copy() const = 0;
};

/*****************************************************************************/
/* DENSE_FEATURE_MAPPING                                                     */
/*****************************************************************************/

/** Holds a mapping between two dense feature spaces.  This includes how
    to map variables onto each other, and how to map the values of
    categorical variables onto each other.
*/
struct Dense_Feature_Mapping {
    Dense_Feature_Mapping();
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
    //Dense_Feature_Space(DB::Store_Reader & store);
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
    boost::shared_ptr<Mutable_Feature_Set>
    encode(const std::vector<float> & variables) const;

    /** Encode the given parameter vector into a feature set for another
        dense feature space.  Takes care of mapping the variables onto each
        other via their variable names. */
    boost::shared_ptr<Mutable_Feature_Set>
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
        boost::shared_ptr<const Dense_Feature_Space>
            problem_fs = feature_space();
        boost::shared_ptr<const Dense_Feature_Space>
            classifier_fs = classifier.feature_space<ML::Dense_Feature_Space>();

        classifier_fs->create_mapping(*problem_fs, mapping);

        distribution<float> features = ...;

        boost::shared_ptr<Mutable_Feature_Set> encoded
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
    boost::shared_ptr<Mutable_Feature_Set>
    encode(const std::vector<float> & variables,
           const Dense_Feature_Space & other, Mapping & mapping) const;

    /** Encode the given parameter vector (in the other feature space) into
        a feature set suitable for this classifier.

        The mapping argument must have already been initialized with
        create_mapping.  Make sure that the parameters are in the right
        order (see create_mapping).
    */
    boost::shared_ptr<Mutable_Feature_Set>
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
                              boost::shared_ptr<Feature_Set> & fs) const;

    /* Methods to deal with the feature space as a whole. */
    virtual Dense_Feature_Space * make_copy() const;
    virtual std::string print() const;

    /** Serialization and reconstitution. */
    virtual std::string class_id() const;
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);
    virtual void reconstitute(DB::Store_Reader & store,
                              const boost::shared_ptr<const Feature_Space>
                                  & feature_space);

    /** Return the total number of variables in the dense feature array. */
    size_t variable_count() const { return info_array.size(); }

    /** Return the list of features. */
    const std::vector<Feature> & features() const { return features_; }
    
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

    virtual Feature_Space_Type type() const { return DENSE; }

    /** Turn the feature info for the given feature into a mutable categorical
        version. */
    boost::shared_ptr<Mutable_Categorical_Info>
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


} // namespace ML
