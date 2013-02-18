/* boosted_stumps_generator.h                                      -*- C++ -*-
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Generator for a boosted_stumps.
*/

#ifndef __boosting__boosted_stumps_generator_h__
#define __boosting__boosted_stumps_generator_h__


#include "weight_updating_generator.h"
#include "stump_generator.h"
#include "boosted_stumps.h"
#include "boosting_training.h"


namespace ML {


/*****************************************************************************/
/* BOOSTED_STUMPS_GENERATOR                                                  */
/*****************************************************************************/

/** Class to generate a classifier.  The meta-algorithms (bagging, boosting,
    etc) can use this algorithm to generate weak-learners.
*/

class Boosted_Stumps_Generator : public Weight_Updating_Generator {
public:
    Boosted_Stumps_Generator();

    virtual ~Boosted_Stumps_Generator();

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
             const std::vector<Feature> & features,
             int recursion) const;

    Boosted_Stumps
    generate_stumps(Thread_Context & context,
                    const Training_Data & training_data,
                    const Training_Data & validation_data,
                    const distribution<float> & training_weights,
                    const distribution<float> & validation_weights,
                    const std::vector<Feature> & features) const;

    /** Generate a classifier, with a given weights array as input. */
    virtual std::shared_ptr<Classifier_Impl>
    generate_and_update(Thread_Context & context,
                        const Training_Data & training_data,
                        boost::multi_array<float, 2> & weights,
                        const std::vector<Feature> & features) const;

    Stump_Generator weak_learner;
    
    unsigned max_iter;
    unsigned min_iter;
    bool true_only;
    Cost_Function cost_function;
    Boosted_Stumps::Output output_function;
    bool fair;
    int short_circuit_window;
    bool trace_training_acc;

    /* Once init has been called, we clone our potential models from this
       one. */
    Boosted_Stumps model;


    /*************************************************************************/
    /* TRAINING METHODS                                                      */
    /*************************************************************************/

    Stump train_iteration(Thread_Context & context,
                          const Training_Data & data,
                          boost::multi_array<float, 2> & weights,
                          std::vector<Feature> & features,
                          Boosted_Stumps & result,
                          Optimization_Info & opt_info) const;

    Stump
    train_iteration(Thread_Context & context,
                    const Training_Data & data,
                    boost::multi_array<float, 2> & weights,
                    std::vector<Feature> & features,
                    Boosted_Stumps & result,
                    boost::multi_array<float, 2> & output,
                    const distribution<float> & ex_weights,
                    double & training_accuracy,
                    Optimization_Info & opt_info) const;

    /** Train a committee of decision stumps.  This function will examine all
        of the features, and return a committee of decision stumps.  It is
        scrupulously fair in that if there is more than one possible stump with
        the same Z value, all of these will either be included or not included.

        This method is a generalization of the train_iteration method above.

        \param data         The training data to train with.
        \param weights      The sample weights.  These will be updated based
                            upon the committee returned.
        \param features     A ranked list of features to test.  They will be
                            tested in the given order.  This list will be
                            re-ranked at the end by the best (lowest) Z score
                            obtained for the given feature.
        \param num_stumps   The number of stumps to try to return in the
                            committee.  Note that this is a hint only: if there
                            are not enough features then less than this number
                            may be returned; if there are several stumps with
                            the same Z value, then all will be included.

        \returns            A list of the decision stumps in the committee.
                            This list will be empty only if there are no
                            features or one was found with a zero Z value.

        An exception will be thrown on an error.
    */
    std::vector<Stump>
    train_iteration_fair(Thread_Context & context,
                         const Training_Data & data,
                         boost::multi_array<float, 2> & weights,
                         std::vector<Feature> & features,
                         Boosted_Stumps & result,
                         std::vector<Optimization_Info> & opt_infos) const;

    double
    update_accuracy(Thread_Context & context,
                    const Stump & stump,
                    const Optimization_Info & opt_info,
                    const Training_Data & data,
                    const std::vector<Feature> & features,
                    boost::multi_array<float, 2> & output,
                    const distribution<float> & ex_weights) const;

};


} // namespace ML


#endif /* __boosting__boosted_stumps_generator_h__ */

