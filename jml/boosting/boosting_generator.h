/* boosting_generator.h                                            -*- C++ -*-
   Jeremy Barnes, 16 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Generic boosting training on top of an arbitrary weak learner.
*/

#ifndef __boosting__boosting_generator_h__
#define __boosting__boosting_generator_h__


#include "weight_updating_generator.h"
#include "boosting_training.h"


namespace ML {


/*****************************************************************************/
/* BOOSTING_GENERATOR                                                        */
/*****************************************************************************/

/** Class to generate a classifier.  The meta-algorithms (boosting, boosting,
    etc) can use this algorithm to generate weak-learners.
*/

class Boosting_Generator : public Weight_Updating_Generator {
public:
    Boosting_Generator();

    virtual ~Boosting_Generator();

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

    using Weight_Updating_Generator::generate;

    /** Generate a classifier from one training set. */
    virtual std::shared_ptr<Classifier_Impl>
    generate(Thread_Context & context,
             const Training_Data & training_data,
             const Training_Data & validation_data,
             const distribution<float> & training_weights,
             const distribution<float> & validation_weights,
             const std::vector<Feature> & features, int) const;

    /** Generate a classifier, with a given weights array as input. */
    virtual std::shared_ptr<Classifier_Impl>
    generate_and_update(Thread_Context & context,
                        const Training_Data & training_data,
                        boost::multi_array<float, 2> & weights,
                        const std::vector<Feature> & features) const;

    std::shared_ptr<Classifier_Generator> weak_learner;

    unsigned max_iter;
    unsigned min_iter;
    Cost_Function cost_function;
    float ignore_highest;
    Stump::Update update_alg;
    int short_circuit_window;
    bool trace_training_acc;

    std::shared_ptr<Classifier_Impl>
    train_iteration(Thread_Context & context,
                    const Training_Data & data,
                    boost::multi_array<float, 2> & weights,
                    std::vector<Feature> & features,
                    float & Z,
                    Optimization_Info & opt_info) const;

    std::shared_ptr<Classifier_Impl>
    train_iteration(Thread_Context & context,
                    const Training_Data & data,
                    boost::multi_array<float, 2> & weights,
                    std::vector<Feature> & features,
                    boost::multi_array<float, 2> & output,
                    const distribution<float> & ex_weights,
                    double & training_accuracy, float & Z,
                    Optimization_Info & opt_info) const;

    double
    update_accuracy(Thread_Context & context,
                    const Classifier_Impl & weak_classifier,
                    const Optimization_Info & opt_info,
                    const Training_Data & data,
                    const std::vector<Feature> & features,
                    boost::multi_array<float, 2> & output,
                    const distribution<float> & ex_weights) const;
};


} // namespace ML


#endif /* __boosting__boosting_generator_h__ */
