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
#include "jml/stats/distribution.h"
#include "jml/arch/demangle.h"
//#include "jml/utils/compact_vector.h"
#include <map>
#include <boost/any.hpp>
#include <boost/multi_array.hpp>
#include <boost/function.hpp>
#include <string>
#include "jml/utils/unnamed_bool.h"

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

std::string print(Output_Encoding encoding);

inline std::ostream & operator << (std::ostream & stream, Output_Encoding e)
{
    return stream << print(e);
}

typedef distribution<float> Label_Dist;
//typedef distribution<float, compact_vector<float, 3> > Label_Dist;


/*****************************************************************************/
/* OPTIMIZATION_INFO                                                         */
/*****************************************************************************/

/** A structure that provides information on optimization to a classifier. */

struct Optimization_Info {
    Optimization_Info() : initialized(false) {}

    std::vector<Feature> from_features;
    std::vector<Feature> to_features;
    std::vector<int> indexes;
    bool initialized;

    std::map<Feature, int> feature_to_optimized_index;

    int features_in() const
    {
        if (!initialized)
            throw Exception("using uninitialized feature map");
        return from_features.size();
    }

    int features_out() const
    {
        if (!initialized)
            throw Exception("using uninitialized feature map");
        return to_features.size();
    }

    void apply(const Feature_Set & fset, float * output) const;
    void apply(const std::vector<float> & fset, float * output) const;
    void apply(const float * fset, float * output) const;
    

    /** Given a feature, return the index in the dense feature vector that
        it corresponds to.  If there is none, an exception will be thrown. */
    int get_optimized_index(const Feature & feature) const;

    JML_IMPLEMENT_OPERATOR_BOOL(initialized);
};


/*****************************************************************************/
/* EXPLANATION                                                               */
/*****************************************************************************/

/** This structure is used to accumulate the effect of various features on
    the prediction. */

struct Explanation {
    Explanation()
    {
    }

    Explanation(std::shared_ptr<const Feature_Space> fspace,
                double weight);

    /** Add the weights from another explanation. */
    void add(const Explanation & other, double weight = 1.0);

    /** Explain how a single prediction was made. */
    std::string explain(int nfeatures,
                        const Feature_Set & fset,
                        int = 0/* ununsed */) const;
    
    /** Explain how the whole set of predictions were made. */
    std::string explain(int nfeatures = -1) const;

    /** Raw explain; the list is a ranked list of (feature, type) pairs.
        MISSING_FEATURE deals with the bias.
    */
    std::vector<std::pair<Feature, float> >
    explainRaw(int nfeatures = -1, bool includeBias = true) const;
    
    /** Divide all of the feature weights by the weight so that the effect
        of feature set size is removed.
    */
    void normalize();

    double bias;
    double weight;

    typedef std::map<Feature, double> Feature_Weights;
    Feature_Weights feature_weights;
    std::shared_ptr<const Feature_Space> fspace;
};


/*****************************************************************************/
/* PREDICTION CONTEXT                                                        */
/*****************************************************************************/

/** This structure provides context for a prediction.  It's designed to
    provide extra information that a classifier may need that doesn't neatly
    fit within its arguments.
*/

struct PredictionContext {
    boost::any seed;  ///< Seed data for the prediction
    std::map<std::string, boost::any> entries;
};


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
    std::shared_ptr<const Feature_Space> feature_space() const
    {
        return feature_space_;
    }

    /** Returns the feature space, dynamic cast as specified. */
    template<class Target_FS>
    std::shared_ptr<const Target_FS> feature_space() const
    {
        std::shared_ptr<const Target_FS> result
            = std::dynamic_pointer_cast<const Target_FS>(feature_space());
        if (!result)
            throw Exception("Couldn't cast feature space of type "
                            + demangle(typeid(*feature_space()).name())
                            + " to "
                            + demangle(typeid(Target_FS).name()));
        return result;
    }
    
    /** Change the feature space.  This method should be used with caution,
        as it is quite possible that there will have been initialisation done
        assuming the old feature space.

        \param new_feature_space  the new feature space to use
    */
    void set_feature_space(const std::shared_ptr<const Feature_Space> &
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
    virtual float predict(int label, const Feature_Set & features,
                          PredictionContext * context = 0) const;

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
    virtual int predict_highest(const Feature_Set & features,
                                PredictionContext * context = 0) const;

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
    predict(const Feature_Set & features,
            PredictionContext * context = 0) const = 0;

    /** Optimize the classifier to be called optimally with the given list of
        features.  The Optimization_Info structure can then be passed to the
        optimized predict method.  Note that the feature vector passed to the
        optimized predict method MUST have EXACTLY the features indicated.
    */
    virtual Optimization_Info optimize(const std::vector<Feature> & features);
    virtual Optimization_Info optimize(const Feature_Set & features);

    /** Is optimization supported by the classifier? */
    virtual bool optimization_supported() const;

    /** Is predict optimized?  Default returns false; those classifiers which
        a) support optimized predict and b) have had optimize_predict() called
        will override to return true in this case.
    */
    virtual bool predict_is_optimized() const;
    

    /** Methods to call for the optimized predict.  Will check if
        predict_is_optimized() and if true, will call the optimized methods.
        Otherwise, they fall back to the non-optimized versions. */
    virtual Label_Dist predict(const Feature_Set & features,
                               const Optimization_Info & info,
                               PredictionContext * context = 0) const;
    virtual Label_Dist predict(const std::vector<float> & features,
                               const Optimization_Info & info,
                               PredictionContext * context = 0) const;
    virtual Label_Dist predict(const float * features,
                               const Optimization_Info & info,
                               PredictionContext * context = 0) const;
    
    virtual float predict(int label,
                          const Feature_Set & features,
                          const Optimization_Info & info,
                          PredictionContext * context = 0) const;
    virtual float predict(int label,
                          const std::vector<float> & features,
                          const Optimization_Info & info,
                          PredictionContext * context = 0) const;
    virtual float predict(int label,
                          const float * features,
                          const Optimization_Info & info,
                          PredictionContext * context = 0) const;

    //protected:

    /** Function to override to perform the optimization.  Default will
        simply modify the optimization info to indicate that optimization
        had failed.
    */
    virtual bool
    optimize_impl(Optimization_Info & info);

    /** Optimized predict for a dense feature vector.
        This is the worker function that all classifiers that implement the
        optimized predict should override.  The default implementation will
        convert to a Feature_Set and will call the non-optimized predict.
    */
    virtual Label_Dist
    optimized_predict_impl(const float * features,
                           const Optimization_Info & info,
                           PredictionContext * context = 0) const;
    
    virtual void
    optimized_predict_impl(const float * features,
                           const Optimization_Info & info,
                           double * accum,
                           double weight = 1.0,
                           PredictionContext * context = 0) const;
    
    virtual float
    optimized_predict_impl(int label,
                           const float * features,
                           const Optimization_Info & info,
                           PredictionContext * context = 0) const;
    
public:
    /** Run the classifier over the entire dataset, calling the predict
        function on each of them.  The output function will be called once
        for each example number (from 0 to data.example_count() - 1) pointing
        to a vector of floats giving the number of labels.
    */
    typedef boost::function<void (int example_num, const float * predictions)>
        Predict_All_Output_Func;
    virtual void predict(const Training_Data & data,
                         Predict_All_Output_Func output,
                         const Optimization_Info * opt_info = 0) const;

    /** Run the classifier over the entire dataset, calling the predict
        function on each of them, to predict a single label.  The output
        function will be called once for each example number (from 0 to
        data.example_count() - 1) with the value of the prediction.
    */
    typedef boost::function<void (int example_num, float prediction)>
        Predict_One_Output_Func;
    virtual void predict(const Training_Data & data,
                         int label,
                         Predict_One_Output_Func output,
                         const Optimization_Info * opt_info = 0) const;

    ///@}

    /** Perform a prediction and explain why the prediction was done in the
        given way.  If not implemented, an exception will be thrown.
    */
    virtual Explanation explain(const Feature_Set & feature_set,
                                int label,
                                double weight = 1.0,
                                PredictionContext * context = 0) const;

    virtual Explanation explainContext(const Feature_Set & feature_set,
                                       int label,
                                       const PredictionContext & context,
                                       double weight = 1.0) const
    {
        return explain(feature_set, label, weight);
    }

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
        \returns                  first: a number between 0 and 1, giving the
                                  proportion of the examples in data that were
                                  correct (for categorical classifiers), or the
                                  RMSE (for regression).
                                  second: the weighted average margin (for
                                  classification) or the RMSE (for regression).

        Note that this method cannot be overridden.
    */
    virtual std::pair<float, float>
    accuracy(const Training_Data & data,
             const distribution<float> & example_weights
                 = UNIFORM_WEIGHTS,
             const Optimization_Info * opt_info = 0) const;

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
    static std::shared_ptr<Classifier_Impl>
    poly_reconstitute(DB::Store_Reader & store,
                      const std::shared_ptr<const Feature_Space> & features);

    /** Perform polymorphic reconstitution, including the feature space. */
    static std::shared_ptr<Classifier_Impl>
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
                              const std::shared_ptr<const Feature_Space>
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
    Classifier_Impl(const std::shared_ptr<const Feature_Space> & fs,
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
    Classifier_Impl(const std::shared_ptr<const Feature_Space> & fs,
                    const Feature & predicted, size_t label_count);

    /** Two-stage initialisation.  Fills in the feature space and the label
        count. */
    void init(const std::shared_ptr<const Feature_Space> & feature_space,
              const Feature & predicted);

    /** Two-stage initialisation.  Fills in the feature space and the label
        count. */
    void init(const std::shared_ptr<const Feature_Space> & feature_space,
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
    std::shared_ptr<const Feature_Space> feature_space_;

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
    FS_Context(const std::shared_ptr<const Feature_Space> & feature_space);
    ~FS_Context();
    
    static const std::shared_ptr<const Feature_Space> & inner();
};

DB::Store_Writer &
operator << (DB::Store_Writer & store,
             const std::shared_ptr<const Classifier_Impl> & classifier);

DB::Store_Reader &
operator >> (DB::Store_Reader & store,
             std::shared_ptr<Classifier_Impl> & classifier);


/*****************************************************************************/
/* CLASSIFIER                                                                */
/*****************************************************************************/

class Classifier {
public:
    /** Default constructor.  Must be initialised or assigned to before
        anything useful can be done. */
    Classifier();

    /** Construct from given implementation. */
    Classifier(const std::shared_ptr<Classifier_Impl> & impl);

    /** Construct from implementation pointer. */
    Classifier(Classifier_Impl * impl, bool take_copy = false);

    /** Construct by copying an existing implementation. */
    explicit Classifier(const Classifier_Impl & impl);

    /** Construct by instantiating a new object of the given type. */
    template<class Impl>
    Classifier(const std::shared_ptr<const Feature_Space> & feature_space)
        : impl(new Impl(feature_space)) {}

    /** Allow the existence to be queried. */
    operator bool () const { return !!impl; }

    /** Construct by instantiating a new object of the given registered
        name. */
    Classifier(const std::string & name,
               const std::shared_ptr<const Feature_Space> & feature_space);

    /** Construct by reconstitution. */
    Classifier(const std::shared_ptr<const Feature_Space> & feature_space,
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
    std::shared_ptr<const Feature_Space> feature_space() const
    {
        return impl->feature_space();
    }
    
    /** Returns the feature space, dynamic cast as specified. */
    template<class Target_FS>
    std::shared_ptr<const Target_FS> feature_space() const
    {
        std::shared_ptr<const Target_FS> result
            = std::dynamic_pointer_cast<const Target_FS>(feature_space());
        if (!result)
            throw Exception("Couldn't cast feature space of type "
                            + demangle(typeid(*feature_space()).name())
                            + " to "
                            + demangle(typeid(Target_FS).name()));
        return result;
    }
    
    /** Predict the score for a single class. */
    float predict(int label, const Feature_Set & features,
                  PredictionContext * context = 0) const
    {
        return impl->predict(label, features, context);
    }

    /** Predict the highest class. */
    int predict_highest(const Feature_Set & features,
                        PredictionContext * context = 0) const
    {
        return impl->predict_highest(features, context);
    }

    /** Predict the score for all classes. */
    Label_Dist
    predict(const Feature_Set & features,
            PredictionContext * context = 0) const
    {
        return impl->predict(features, context);
    }

    /** Calculate the prediction accuracy over a training set. */
    std::pair<float, float>
    accuracy(const Training_Data & data,
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
                      const std::shared_ptr<const Feature_Space>
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
              std::shared_ptr<const Feature_Space> fs);
    void save(const std::string & filename, bool write_fs = true) const;

    std::shared_ptr<Classifier_Impl> impl;
};


DB::Store_Writer &
operator << (DB::Store_Writer & store, const Classifier & classifier);

DB::Store_Reader &
operator >> (DB::Store_Reader & store, Classifier & classifier);

} // namespace ML




#endif /* __boosting__classifier_h__ */
