/* bagging_generator.h                                          -*- C++ -*-
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Generator for a bagging.
*/

#ifndef __boosting__bagging_generator_h__
#define __boosting__bagging_generator_h__


#include "early_stopping_generator.h"


namespace ML {


/*****************************************************************************/
/* BAGGING_GENERATOR                                                         */
/*****************************************************************************/

/** Class to generate a classifier.  The meta-algorithms (bagging, boosting,
    etc) can use this algorithm to generate weak-learners.
*/

class Bagging_Generator : public Early_Stopping_Generator {
public:
    Bagging_Generator();

    virtual ~Bagging_Generator();

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

    using Early_Stopping_Generator::generate;

    /** Generate a classifier from one training set. */
    virtual std::shared_ptr<Classifier_Impl>
    generate(Thread_Context & context,
             const Training_Data & training_data,
             const Training_Data & validation_data,
             const distribution<float> & training_weights,
             const distribution<float> & validation_weights,
             const std::vector<Feature> & features,
             int) const;

    std::shared_ptr<Classifier_Generator> weak_learner;

    int num_bags;
    float validation_split;
    float testing_split;
};


} // namespace ML


#endif /* __boosting__bagging_generator_h__ */
