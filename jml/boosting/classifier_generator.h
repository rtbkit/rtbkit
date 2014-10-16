/* classifier_generator.h                                          -*- C++ -*-
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Generic classifier generator.  Can be used by classifier training
   tools.
*/

#ifndef __boosting__classifier_generator_h__
#define __boosting__classifier_generator_h__

#include "config.h"
#include "config_options.h"
#include "feature_space.h"
#include "training_data.h"
#include "jml/utils/configuration.h"
#include "classifier.h"
#include "thread_context.h"


namespace ML {


/*****************************************************************************/
/* CLASSIFIER_GENERATOR                                                      */
/*****************************************************************************/

/** Class to generate a classifier.  The meta-algorithms (bagging, boosting,
    etc) can use this algorithm to generate weak-learners.
*/

class Classifier_Generator {
public:
    virtual ~Classifier_Generator();

    /** Configure the generator with its parameters. */
    virtual void
    configure(const Configuration & config);
    
    /** Return to the default configuration. */
    virtual void defaults();

    /** Return possible configuration options. */
    virtual Config_Options options() const;

    /** Initialize the generator, given the feature space to be used for
        generation. */
    virtual void init(std::shared_ptr<const Feature_Space> fs,
                      Feature predicted);

    /** Generate a classifier from two training sets.  Default will
        call the next method with uniform weights. */
    virtual std::shared_ptr<Classifier_Impl>
    generate(Thread_Context & context,
             const Training_Data & training_data,
             const Training_Data & validation_data,
             const std::vector<Feature> & features,
             int recursion = 0) const;

    /** Generate a classifier from two training sets with data weighted
        by example.  Default will call the next method with the training
        data only.
    */
    virtual std::shared_ptr<Classifier_Impl>
    generate(Thread_Context & context,
             const Training_Data & training_data,
             const Training_Data & validation_data,
             const distribution<float> & training_weights,
             const distribution<float> & validation_weights,
             const std::vector<Feature> & features,
             int recursion = 0) const;

    /** Generate a classifier from a training set with data weighted
        by example.  Default expands the weights and calls the other
        generate method.  Note that either this method or the one
        after it must be overridden. */
    virtual std::shared_ptr<Classifier_Impl>
    generate(Thread_Context & context,
             const Training_Data & training_data,
             const distribution<float> & weights,
             const std::vector<Feature> & features,
             int recursion = 0) const;
    
    /** Generate a classifier from a training set with data weighted
        by label.  Default will contract the weights and call the
        other generate method.  Note that either this method or the
        one before it must be overridden.  This method is used by
        the boosting algorithm.

        The Z parameter is used by boosting to get an idea of how good
        the classifier is.  Those that don't calulate it should set it
        to 0.0.
    */
    virtual std::shared_ptr<Classifier_Impl>
    generate(Thread_Context & context,
             const Training_Data & training_data,
             const boost::multi_array<float, 2> & weights,
             const std::vector<Feature> & features,
             float & Z,
             int recursion = 0) const;

    /** What type of generator is it? */
    virtual std::string type() const;

    /* Enough for the moment... */

    /** Log a message for the given module at the given debug level. */
    std::ostream & log(const std::string & module, int level) const;

    /** Current verbosity level. */
    int verbosity;

    /** Are we profiling? */
    bool profile;

    /** Do we perform validation as we go? */
    bool validate;

    /** Feature space we are using. */
    std::shared_ptr<const Feature_Space> feature_space;

    /** Feature we are predicting. */
    Feature predicted;

    /** Number of labels. */
    size_t nl;
};


/*****************************************************************************/
/* FACTORIES                                                                 */
/*****************************************************************************/

/** Generate a trainer for the classifier with the given name. */
std::shared_ptr<Classifier_Generator>
get_trainer(const std::string & name, const Configuration & config);

template<class Base> class Factory_Base;
template<class Base, class Derived> class Object_Factory;

template<>
class Factory_Base<Classifier_Generator> {
public:
    virtual ~Factory_Base() {}
    virtual std::shared_ptr<Classifier_Generator> create() const = 0;
};

template<class Derived>
class Object_Factory<Classifier_Generator, Derived>
    : public Factory_Base<Classifier_Generator> {
public:
    virtual ~Object_Factory() {}
    virtual std::shared_ptr<Classifier_Generator> create() const
    {
        return std::shared_ptr<Classifier_Generator>(new Derived());
    }
};


} // namespace ML

#endif /* __boosting__classifier_generator_h__ */
