/* classifier.h                                                    -*- C++ -*-
   Jeremy Barnes, 6 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Generic interface to a classifier.
*/

#ifndef __boosting__classifier_h__
#define __boosting__classifier_h__


#include "feature_set.h"
#include "training_data.h"
#include "stats/distribution.h"
#include "arch/demangle.h"
//#include "utils/compact_vector.h"
#include <map>
#include <boost/any.hpp>
#include <boost/multi_array.hpp>
#include <boost/function.hpp>
#include <string>

namespace ML {

extern const distribution<float> UNIFORM_WEIGHTS;


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

BYTE_PERSISTENT_ENUM_DECL(Output_Encoding);

typedef distribution<float> Label_Dist;
//typedef distribution<float, compact_vector<float, 3> > Label_Dist;


/*****************************************************************************/
/* CLASSIFIER_IMPL                                                           */
/*****************************************************************************/

/** The base class upon which classifiers are built.

    This class provides the base framework upon which all of the classifier
    classes are built.  It is designed to be general enough to allow for most
    operations, including boosting, to be performed without requiring that the
    exact underlying class be known.  It also provides the framework necessary
    for persistence.

    Note that user code will often use the Classifier class (which is always
    concrete) than the Classifier_Impl.  It is mainly implementors of
    classifiers, or algorithms which need special treatment, that will
    concern themselves with this class.

    Note that the word "label" is used where the word "class" would normally
    be used in the literature to describe the range of the classifier (the
    discreet outputs).  This is due to a name clash with the C++ reserved
    word "class".
*/

class Classifier_Impl {
public:
    /** Standard virtual destructor.  Does nothing as all variables are smart
        pointers. */
    virtual ~Classifier_Impl();

    /** \name Domain and range
        These functions allow the domain and range of the classifier to be
        queried.  They describe the arity and interpretation of its input
        and output spaces.
        @{
    */

    /** Returns the number of classes that this classifier is trying to
        classify into.  Non-virtual: simply reads the protected variable
        label_count_.

        \returns                  the number of labels in the range of the
                                  classifier.
    */
    size_t label_count() const { return label_count_; }
    ///@}

    /** Returns the feature space being used.  This describes the range of
        the classifier, and also the interpretation of the features in its
        input.
    
        \returns                  a shared pointer to the feature space
                                  being used.

        Note that this method is non-virtual and simply returns the value
        of the protected variables feature_space_.  Thus, it cannot be
        overridden to change the behaviour; the correct feature space will
        have to be passed in at construction.
    */
    boost::shared_ptr<const Feature_Space> feature_space() const
    {
        return feature_space_;
    }

    /** Change the feature space.  This method should be used with caution,
        as it is quite possible that there will have been initialisation done
        assuming the old feature space.

        \param new_feature_space  the new feature space to use
    */
    void set_feature_space(const boost::shared_ptr<const Feature_Space> &
                               new_feature_space)
    {
        feature_space_ = new_feature_space;
    }
    ///@}

    /** \name Prediction
        These methods are used when applying an already learned classifier.
        They attempt to make a classification based upon a feature set.
        @{
    */

    /** Predict the score for a single label.
        \param label              label to predict.
        \param features           features to use to make the prediction.
        \returns                  the prediction for the given label, based
                                  upon the features given.

        \pre label >= 0 && label < label_count()
        \post result == predict(features).at(label)

        Throws an out of range exception if label was not in range, or
        an Exception if there is a problem in performing the
        classification.

        The default implementation is the following:
        \code
        float Classifier_Impl::
        predict(int label, const Feature_Set & features) const
        {
            return predict(features).at(label);
        }
        \endcode
        
        The only reason that this method is provided is that it may be
        quicker (more efficient) to calculate a score for one feature only
        rather than all at once.
     */
    virtual float predict(int label, const Feature_Set & features) const;

    /** Predict the highest class.  This method returns the label which has
        the highest class.  Ties may be handled in any way.  Normally, it
        is better to use the predict method than this one.

        \param features           the features to use to make the prediction.
        \returns                  an integer 0 <= result < label_count() with
                                  the number of the highest label.
                                  
        The default implementation simply finds the maximum value in the
        output of \c predict(features), and returns the index of the first
        one which is equal to this value.

        Throws an exception if label_count() == 0, or if there was a problem
        performing the classification.
    */
    virtual float predict_highest(const Feature_Set & features) const;

    /** Predict the score for all labels.  This method needs to be overridden
        at the very minimum.
        
        \param features           the features used to make the prediction.
        \returns                  a distribution over all labels of the
                                  classifier output.
        \post                     result.size() == label_count()
        
        If there is an error in classification, the implementation should throw
        an exception.
     */
    virtual Label_Dist
    predict(const Feature_Set & features) const = 0;

    /** Run the classifier over the entire dataset, calling the predict
        function on each of them.  The output function will be called once
        for each example number (from 0 to data.example_count() - 1) pointing
        to a vector of floats giving the number of labels.
    */
    typedef boost::function<void (int example_num, const float * predictions)>
        Predict_All_Output_Func;
    virtual void predict(const Training_Data & data,
                         Predict_All_Output_Func output) const;

    /** Run the classifier over the entire dataset, calling the predict
        function on each of them, to predict a single label.  The output
        function will be called once for each example number (from 0 to
        data.example_count() - 1) with the value of the prediction.
    */
    typedef boost::function<void (int example_num, float prediction)>
        Predict_One_Output_Func;
    virtual void predict(const Training_Data & data,
                         int label,
                         Predict_One_Output_Func output) const;

    ///@}

    /** \name Accuracy
        These methods are all ways of returning the accuracy of the classifier
        over a set of training data.  They vary in how the output of the
        classifier is represented.
     */

    /** Calculate the prediction accuracy over a training set, using the
        predict() to determine the predictions.

        \param data               the training data used to calculate the
                                  accuracy over.
        \param example_weights    a weighting of the examples.  The examples
                                  will count as if they were duplicated this
                                  number of times.  An empty distribution is
                                  the same as all ones.
        \returns                  a number between 0 and 1, giving the
                                  proportion of the examples in data that were
                                  correct.

        Note that this method cannot be overridden.
    */
    virtual float accuracy(const Training_Data & data,
                           const distribution<float> & example_weights
                               = UNIFORM_WEIGHTS) const;

    ///@}
    
    /** Provides a list of all features used by this classifier. */
    virtual std::vector<Feature> all_features() const = 0;

    /** Dump a representation of the classifier to a string. */
    virtual std::string print() const = 0;

    /** Dump a short compact representation.  Default is just the class
        name. */
    virtual std::string summary() const;

    /** \name Persistence
        These functions are concerned with allowing serialization and
        reconstitution without needing to know the details of what is the
        exact type of classifier that is being used.
        @{
    */

    /** Perform polymorphic reconstitution.  Can't be done with the normal
        reconstitution operator since the feature space needs to be passed
        in to it.
    */
    static boost::shared_ptr<Classifier_Impl>
    poly_reconstitute(DB::Store_Reader & store,
                      const boost::shared_ptr<const Feature_Space> & features);

    /** Perform polymorphic reconstitution, including the feature space. */
    static boost::shared_ptr<Classifier_Impl>
    poly_reconstitute(DB::Store_Reader & store);

    /** Perform polymorphic serialization.  This is different to the virtual
        methods that are overridden elsewhere, as this one also writes the
        class metadata to the stream.
    */
    void poly_serialize(DB::Store_Writer & store, bool write_fs = true) const;

    /** Return the ID of this class.  This needs to be unique for each
        class in order to allow the persistence framework to function.
    */
    virtual std::string class_id() const = 0;

    /** Serialization and reconstitution. */
    virtual void serialize(DB::Store_Writer & store) const = 0;
    virtual void reconstitute(DB::Store_Reader & store,
                              const boost::shared_ptr<const Feature_Space>
                                  & feature_space) = 0;
    
    /** Allow polymorphic copying. */
    virtual Classifier_Impl * make_copy() const = 0;
    ///@}

    /** Return the feature that this classifier is predicting. */
    const Feature & predicted() const { return predicted_; }

    virtual bool merge_into(const Classifier_Impl & other, float weight = 1.0);

    virtual Classifier_Impl *
    merge(const Classifier_Impl & other, float weight = 1.0) const;

    virtual Output_Encoding output_encoding() const = 0;
    
protected:
    /** Default constructor.  Must be initialised or assigned to before
        anything useful can be done. */
    Classifier_Impl();

    /** Construct for the given feature space, to classify into the given
        number of labels.
        
        \param fs                the feature space.  This specifies the
                                 domain of the classifier (what it takes in
                                 as its input).
        \param predicted         the feature that the classifier is trying
                                 to predict.
    */
    Classifier_Impl(const boost::shared_ptr<const Feature_Space> & fs,
                    const Feature & predicted);

    /** Construct for the given feature space, to classify into the given
        number of labels.
        
        \param fs                the feature space.  This specifies the
                                 domain of the classifier (what it takes in
                                 as its input).
        \param predicted         the feature that the classifier is trying
                                 to predict.
        \param label_count       the number of labels (classes) that the
                                 classifier will predict into.
    */
    Classifier_Impl(const boost::shared_ptr<const Feature_Space> & fs,
                    const Feature & predicted, size_t label_count);

    /** Two-stage initialisation.  Fills in the feature space and the label
        count. */
    void init(const boost::shared_ptr<const Feature_Space> & feature_space,
              const Feature & predicted);

    /** Two-stage initialisation.  Fills in the feature space and the label
        count. */
    void init(const boost::shared_ptr<const Feature_Space> & feature_space,
              const Feature & predicted, size_t label_count);

    /** Swap the class with another one.  This method is protected as it is
        dangerous: if it were public it might get called on two classifiers
        of a derived class, which would then swap only the Classifier_Impl
        members and not the derived class's members.

        Normally, you will want to swap two pointers that point to something
        derived from a Classifier_Impl rather then the objects themselves.
    */
    void swap(Classifier_Impl & other)
    {
        std::swap(label_count_, other.label_count_);
        feature_space_.swap(other.feature_space_);
        std::swap(predicted_, other.predicted_);
    }

protected:
    /** The feature space to use.  Be careful when changing it that something
        doesn't already depend upon the other value. */
    boost::shared_ptr<const Feature_Space> feature_space_;

    /** The feature that we are trying to predict. */
    Feature predicted_;

    /** The number of labels.  Be careful when changing it that something
        doesn't already depend upon the other value. */
    size_t label_count_;
};


/*****************************************************************************/
/* POLYMORPHIA                                                               */
/*****************************************************************************/

class FS_Context {
public:
    FS_Context(const boost::shared_ptr<const Feature_Space> & feature_space);
    ~FS_Context();
    
    static const boost::shared_ptr<const Feature_Space> & inner();
};

DB::Store_Writer &
operator << (DB::Store_Writer & store,
             const boost::shared_ptr<const Classifier_Impl> & classifier);

DB::Store_Reader &
operator >> (DB::Store_Reader & store,
             boost::shared_ptr<Classifier_Impl> & classifier);


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
    template<class Impl>
    Classifier(const boost::shared_ptr<const Feature_Space> & feature_space)
        : impl(new Impl(feature_space)) {}

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
    void swap(Classifier & other)
    {
        impl.swap(other.impl);
    }

    /** Returns the number of classes that this classifier is trying to
        classify into.  */
    size_t label_count() const
    {
        return impl->label_count();
    }
    
    /** Returns the feature space. */
    boost::shared_ptr<const Feature_Space> feature_space() const
    {
        return impl->feature_space();
    }
    
    /** Returns the feature space, dynamic cast as specified. */
    template<class Target_FS>
    boost::shared_ptr<const Target_FS> feature_space() const
    {
        boost::shared_ptr<const Target_FS> result
            = boost::dynamic_pointer_cast<const Target_FS>(feature_space());
        if (!result)
            throw Exception("Couldn't cast feature space of type "
                            + demangle(typeid(*feature_space()).name())
                            + " to "
                            + demangle(typeid(Target_FS).name()));
        return result;
    }
    
    /** Predict the score for a single class. */
    float predict(int label, const Feature_Set & features) const
    {
        return impl->predict(label, features);
    }

    /** Predict the highest class. */
    float predict_highest(const Feature_Set & features) const
    {
        return impl->predict_highest(features);
    }

    /** Predict the score for all classes. */
    Label_Dist
    predict(const Feature_Set & features) const
    {
        return impl->predict(features);
    }

    /** Calculate the prediction accuracy over a training set. */
    float accuracy(const Training_Data & data,
                   const distribution<float> & example_weights
                       = UNIFORM_WEIGHTS) const
    {
        return impl->accuracy(data);
    }

    /** Dump it to a string. */
    std::string print() const
    {
        return impl->print();
    }

    /** Provides a list of all features used by this classifier. */
    std::vector<Feature> all_features() const
    {
        return impl->all_features();
    }

    /** Return the feature that this classifier is predicting. */
    const Feature & predicted() const
    {
        return impl->predicted();
    }

    /** Serialization and reconstitution. */
    void serialize(DB::Store_Writer & store, bool write_fs = true) const
    {
        impl->poly_serialize(store, write_fs);
    }

    /** Reconstitute into the given feature space. */
    void reconstitute(DB::Store_Reader & store,
                      const boost::shared_ptr<const Feature_Space>
                              & feature_space)
    {
        impl = Classifier_Impl::poly_reconstitute(store, feature_space);
    }

    /** Reconstitute. */
    void reconstitute(DB::Store_Reader & store)
    {
        impl = Classifier_Impl::poly_reconstitute(store);
    }

    /** Return the class ID of the given classifier. */
    std::string class_id() const
    {
        if (!impl) return "NONE";
        else return impl->class_id();
    }

    Output_Encoding output_encoding() const
    {
        return impl->output_encoding();
    }

    void load(const std::string & filename);
    void load(const std::string & filename,
              boost::shared_ptr<const Feature_Space> fs);
    void save(const std::string & filename, bool write_fs = true) const;

    boost::shared_ptr<Classifier_Impl> impl;
};


DB::Store_Writer &
operator << (DB::Store_Writer & store, const Classifier & classifier);

DB::Store_Reader &
operator >> (DB::Store_Reader & store, Classifier & classifier);

} // namespace ML




#endif /* __boosting__classifier_h__ */
