/* glz_classifier_generator.h                                          -*- C++ -*-
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Generator for a glz_classifier.
*/

#ifndef __boosting__glz_classifier_generator_h__
#define __boosting__glz_classifier_generator_h__


#include "classifier_generator.h"
#include "glz_classifier.h"


namespace ML {


/*****************************************************************************/
/* GLZ_CLASSIFIER_GENERATOR                                                  */
/*****************************************************************************/

/** Class to generate a classifier.  The meta-algorithms (bagging, boosting,
    etc) can use this algorithm to generate weak-learners.
*/

class GLZ_Classifier_Generator : public Classifier_Generator {
public:
    GLZ_Classifier_Generator();

    virtual ~GLZ_Classifier_Generator();

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
             const boost::multi_array<float, 2> & weights,
             const std::vector<Feature> & features,
             float & Z,
             int) const;

    bool add_bias;
    bool do_decode;
    bool normalize;
    bool ridge_regression;
    Link_Function link_function;
    float feature_proportion;

    /* Once init has been called, we clone our potential models from this
       one. */
    GLZ_Classifier model;


    float train_weighted(Thread_Context & thread_context,
                         const Training_Data & data,
                         const boost::multi_array<float, 2> & weights,
                         const std::vector<Feature> & features,
                         GLZ_Classifier & result) const;
};


} // namespace ML


#endif /* __boosting__glz_classifier_generator_h__ */
