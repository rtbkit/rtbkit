/* classifier.i                                                    -*- C++ -*-
   Jeremy Barnes, 24 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   SWIG wrapper for the classifier class.
*/

%module jml 
%{
#include "jml/boosting/classifier.h"
%}

namespace ML {

/*****************************************************************************/
/* OUTPUT_ENCODING                                                           */
/*****************************************************************************/

/** This enumeration gives the output encoding for a classifier.   There are
    currently three different possibilities.
*/

enum Output_Encoding {
    OE_PROB,     ///< Output encoding is a probability between 0 and 1
    OE_PM_INF,   ///< Output encoding is a value between -inf and inf
    OE_PM_ONE    ///< Output encoding is a value between -1 and 1
};


class Classifier_Impl;

typedef distribution<float> Label_Dist;


/*****************************************************************************/
/* CLASSIFIER                                                                */
/*****************************************************************************/

class Classifier {
public:
    /** Default constructor.  Must be initialised or assigned to before
        anything useful can be done. */
    Classifier();

    /** Construct from given implementation. */
    Classifier(const boost::shared_ptr<Classifier_Impl> & impl);

    /** Construct from implementation pointer. */
    Classifier(Classifier_Impl * impl, bool take_copy = false);

    /** Construct by copying an existing implementation. */
    explicit Classifier(const Classifier_Impl & impl);

    /** Construct by instantiating a new object of the given type. */
    //template<class Impl>
    //Classifier(const boost::shared_ptr<const Feature_Space> & feature_space)
    //    : impl(new Impl(feature_space)) {}

    /** Allow the existence to be queried. */
    operator bool () const { return impl; }

    /** Construct by instantiating a new object of the given registered
        name. */
    Classifier(const std::string & name,
               const boost::shared_ptr<const Feature_Space> & feature_space);

    /** Construct by reconstitution. */
    Classifier(const boost::shared_ptr<const Feature_Space> & feature_space,
               DB::Store_Reader & store);

    /** Copy constructor. */
    Classifier(const Classifier & other);

    /** Assignment operator. */
    Classifier & operator = (const Classifier & other);
    
    /** Allow efficient swapping. */
    void swap(Classifier & other);

    /** Returns the number of classes that this classifier is trying to
        classify into.  */
    size_t label_count() const;
    
    /** Returns the feature space. */
    boost::shared_ptr<const Feature_Space> feature_space() const;
    
    /** Returns the feature space, dynamic cast as specified. */
    //template<class Target_FS>
    //boost::shared_ptr<const Target_FS> feature_space() const;
    
    /** Predict the score for a single class. */
    float predict(int label, const Feature_Set & features) const;

    /** Predict the highest class. */
    int predict_highest(const Feature_Set & features) const;

    /** Predict the score for all classes. */
    Label_Dist
    predict(const Feature_Set & features) const;

    /** Calculate the prediction accuracy over a training set. */
    std::pair<float, float>
    accuracy(const Training_Data & data,
             const distribution<float> & example_weights
                  = UNIFORM_WEIGHTS) const;

    /** Dump it to a string. */
    std::string print() const;

    /** Provides a list of all features used by this classifier. */
    std::vector<Feature> all_features() const;

    /** Return the feature that this classifier is predicting. */
    const Feature & predicted() const;

    /** Serialization and reconstitution. */
    void serialize(DB::Store_Writer & store, bool write_fs = true) const;

    /** Reconstitute into the given feature space. */
    void reconstitute(DB::Store_Reader & store,
                      const boost::shared_ptr<const Feature_Space>
                          & feature_space);

    /** Reconstitute. */
    void reconstitute(DB::Store_Reader & store);

    /** Return the class ID of the given classifier. */
    std::string class_id() const;

    Output_Encoding output_encoding() const;

    void load(const std::string & filename);
    void load(const std::string & filename,
              boost::shared_ptr<const Feature_Space> fs);
    void save(const std::string & filename, bool write_fs = true) const;
};

} // namespace ML

