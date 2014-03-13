/* decision_tree_generator.h                                       -*- C++ -*-
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.

   Generator for a decision_tree.
*/

#ifndef __boosting__decision_tree_generator_h__
#define __boosting__decision_tree_generator_h__


#include "classifier_generator.h"
#include "decision_tree.h"
#include "stump.h"


namespace ML {


/*****************************************************************************/
/* DECISION_TREE_GENERATOR                                                  */
/*****************************************************************************/

/** Class to generate a classifier.  The meta-algorithms (bagging, boosting,
    etc) can use this algorithm to generate weak-learners.
*/

class Decision_Tree_Generator : public Classifier_Generator {
public:
    Decision_Tree_Generator();

    virtual ~Decision_Tree_Generator();

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
             const std::vector<Feature> & features, int recursion) const;

    /** Generate a classifier for boosting. */
    virtual std::shared_ptr<Classifier_Impl>
    generate(Thread_Context & context,
             const Training_Data & training_data,
             const boost::multi_array<float, 2> & weights,
             const std::vector<Feature> & features,
             float & Z,
             int recursion) const;

    int max_depth;
    int trace;
    Stump::Update update_alg;
    float random_feature_propn;

    /* Once init has been called, we clone our potential models from this
       one. */
    Decision_Tree model;

    Decision_Tree
    train_weighted(Thread_Context & context,
                   const Training_Data & data,
                   const boost::multi_array<float, 2> & weights,
                   const std::vector<Feature> & features,
                   int max_depth) const;
    
    Tree::Ptr
    train_recursive(Thread_Context & context,
                    const Training_Data & data,
                    const std::vector<const float *> & weights,
                    int advance,
                    const std::vector<Feature> & features,
                    const distribution<float> & in_class,
                    int depth, int max_depth, Tree & tree) const;

    Tree::Ptr
    train_recursive_regression(Thread_Context & context,
                               const Training_Data & data,
                               const std::vector<float> & weights,
                               const std::vector<Feature> & features_,
                               const distribution<float> & in_class,
                               int depth, int max_depth, Tree & tree) const;

    void do_branch(Tree::Ptr & ptr,
                   int & group_to_wait_on,
                   Thread_Context & context,
                   const Training_Data & data,
                   const std::vector<const float *> & weights,
                   int advance,
                   const std::vector<Feature> & features,
                   const distribution<float> & new_in_class,
                   double total_in_class,
                   int new_depth, int max_depth,
                   Tree & tree) const;
    
    struct Train_Recursive_Job;
};


} // namespace ML


#endif /* __boosting__decision_tree_generator_h__ */
