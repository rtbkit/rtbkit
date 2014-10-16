/* null_classifier_generator.h                                          -*- C++ -*-
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Generator for a null_classifier.
*/

#ifndef __boosting__null_classifier_generator_h__
#define __boosting__null_classifier_generator_h__


#include "classifier_generator.h"
#include "null_classifier.h"


namespace ML {


/*****************************************************************************/
/* NULL_CLASSIFIER_GENERATOR                                                 */
/*****************************************************************************/

/** Class to generate a classifier.  The meta-algorithms (bagging, boosting,
    etc) can use this algorithm to generate weak-learners.
*/

class Null_Classifier_Generator : public Classifier_Generator {
public:
    Null_Classifier_Generator();

    virtual ~Null_Classifier_Generator();

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

    using Classifier_Generator::generate;

    /** Generate a classifier from one training set. */
    virtual std::shared_ptr<Classifier_Impl>
    generate(Thread_Context & context,
             const Training_Data & training_data,
             const Training_Data & validation_data,
             const distribution<float> & training_weights,
             const distribution<float> & validation_weights,
             const std::vector<Feature> & features,
             int recursion) const;

    /* Once init has been called, we clone our potential models from this
       one. */
    Null_Classifier model;
};


} // namespace ML


#endif /* __boosting__null_classifier_generator_h__ */
