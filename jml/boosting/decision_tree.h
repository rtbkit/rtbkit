/* decision_tree.h                                                -*- C++ -*-
   Jeremy Barnes, 22 March 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Decision tree classifier.
*/

#ifndef __boosting__decision_tree_h__
#define __boosting__decision_tree_h__

#include "classifier.h"
#include "feature_set.h"
#include <boost/pool/object_pool.hpp>
#include "tree.h"
#include "boolean_expression.h"


namespace ML {


class Training_Data;


/*****************************************************************************/
/* DECISION_TREE                                                             */
/*****************************************************************************/

/** This is a tree of arbitrary depth as a classifier.  Based upon the CART
    algorithm.

    We are not yet doing any pruning.  As a result, capacity control needs
    to be done by using the max_depth property, rather than relying on held
    out data.  The correct depth can still be evaluated on this held out
    data, though.
*/

class Decision_Tree : public Classifier_Impl {
public:
    /** Default construct.  Must be initialised before use. */
    Decision_Tree();

    /** Construct it by reconstituting it from a store. */
    Decision_Tree(DB::Store_Reader & store,
                  const std::shared_ptr<const Feature_Space> & fs);
    
    /** Construct not filled in yet. */
    Decision_Tree(std::shared_ptr<const Feature_Space> feature_space,
                  const Feature & predicted);
    
    virtual ~Decision_Tree();
    
    void swap(Decision_Tree & other);

    Tree tree;                 ///< The tree we have learned
    Output_Encoding encoding;  ///< How the outputs are represented
    bool optimized_;           ///< Is predict() optimized?

    using Classifier_Impl::predict;

    virtual float predict(int label, const Feature_Set & features,
                          PredictionContext * context = 0) const;

    virtual distribution<float>
    predict(const Feature_Set & features,
            PredictionContext * context = 0) const;

    /** Is optimization supported by the classifier? */
    virtual bool optimization_supported() const;

    /** Is predict optimized?  Default returns false; those classifiers which
        a) support optimized predict and b) have had optimize_predict() called
        will override to return true in this case.
    */
    virtual bool predict_is_optimized() const;
    /** Function to override to perform the optimization.  Default will
        simply modify the optimization info to indicate that optimization
        had failed.
    */
    virtual bool
    optimize_impl(Optimization_Info & info);

    void optimize_recursive(Optimization_Info & info,
                            const Tree::Ptr & ptr);

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
                           double weight,
                           PredictionContext * context = 0) const;
    virtual float
    optimized_predict_impl(int label,
                           const float * features,
                           const Optimization_Info & info,
                           PredictionContext * context = 0) const;

    template<class GetFeatures, class Results>
    void predict_recursive_impl(const GetFeatures & get_features,
                                Results & results,
                                const Tree::Ptr & ptr,
                                double weight = 1.0) const;

    virtual Explanation explain(const Feature_Set & feature_set,
                                int label,
                                double weight = 1.0,
                                PredictionContext * context = 0) const;

    void explain_recursive(Explanation & explanation,
                           const Feature_Set & fset,
                           int label,
                           double weight,
                           const Tree::Ptr & ptr,
                           const Tree::Node * parent) const;

    /** Convert the decision tree to a disjuction of conjunctions form
        of boolean rules. */
    virtual Disjunction<Tree::Leaf> to_rules() const;

    void to_rules_recursive(Disjunction<Tree::Leaf> & result,
                            std::vector<std::shared_ptr<Predicate> > & path,
                            const Tree::Ptr & ptr) const;

    virtual std::string print() const;

    virtual std::string summary() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    std::string print_recursive(int level, const Tree::Ptr & ptr,
                                float total_weight) const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const std::shared_ptr<const Feature_Space>
                                  & feature_space);
    
    virtual std::string class_id() const;

    virtual Decision_Tree * make_copy() const;
};


} // namespace ML



#endif /* __boosting__decision_tree_h__ */
