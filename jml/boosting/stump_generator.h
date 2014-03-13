/* stump_generator.h                                               -*- C++ -*-
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Generator for a stump.
*/

#ifndef __boosting__stump_generator_h__
#define __boosting__stump_generator_h__


#include "classifier_generator.h"
#include "stump.h"


namespace ML {


/*****************************************************************************/
/* STUMP_GENERATOR                                                           */
/*****************************************************************************/

/** Class to generate a classifier.  The meta-algorithms (bagging, boosting,
    etc) can use this algorithm to generate weak-learners.
*/

class Stump_Generator : public Classifier_Generator {
public:
    Stump_Generator();

    virtual ~Stump_Generator();

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
             int) const;

    /** Generate a classifier for boosting. */
    virtual std::shared_ptr<Classifier_Impl>
    generate(Thread_Context & context,
             const Training_Data & training_data,
             const boost::multi_array<float, 2> & weights,
             const std::vector<Feature> & features,
             float & Z,
             int recursion = 0) const;

    int trace;
    float ignore_highest;
    int committee_size;
    Stump::Update update_alg;
    float feature_prop;

    /* Once init has been called, we clone our potential models from this
       one. */
    Stump model;


    /*************************************************************************/
    /* STUMP_GENERATOR                                                       */
    /*************************************************************************/

    /** Training */
    Stump train_weighted(Thread_Context & context,
                         const Training_Data & data,
                         const boost::multi_array<float, 2> & weights,
                         const std::vector<Feature> & features) const;

    /** Find the best committee_size stumps. */
    std::vector<Stump>
    train_all(Thread_Context & context,
              const Training_Data & data,
              const boost::multi_array<float, 2> & weights,
              const std::vector<Feature> & features) const;

    /** Get the bias for this stump.  This is the weight that would be
        assigned to the missing class when all of the examples are missing
        a given feature. */
    static Stump
    get_bias(const Training_Data & data,
             const boost::multi_array<float, 2> & weights,
             const Feature & predicted,
             int trace, Stump::Update update_alg);
};


} // namespace ML


#endif /* __boosting__stump_generator_h__ */

