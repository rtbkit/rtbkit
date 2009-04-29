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
                  const boost::shared_ptr<const Feature_Space> & fs);
    
    /** Construct not filled in yet. */
    Decision_Tree(boost::shared_ptr<const Feature_Space> feature_space,
                  const Feature & predicted);
    
    virtual ~Decision_Tree();
    
    void swap(Decision_Tree & other);

    Tree tree;                 ///< The tree we have learned
    Output_Encoding encoding;  ///< How the outputs are represented

    using Classifier_Impl::predict;

    virtual float predict(int label, const Feature_Set & features) const;

    virtual distribution<float>
    predict(const Feature_Set & features) const;

    distribution<float>
    predict_recursive(const Feature_Set & features,
                      const Tree::Ptr & ptr) const;

    virtual std::string print() const;

    virtual std::string summary() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    std::string print_recursive(int level, const Tree::Ptr & ptr,
                                float total_weight) const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const boost::shared_ptr<const Feature_Space>
                                  & feature_space);
    
    virtual std::string class_id() const;

    virtual Decision_Tree * make_copy() const;
};


} // namespace ML



#endif /* __boosting__decision_tree_h__ */
