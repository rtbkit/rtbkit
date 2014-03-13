/* split.h                                                         -*- C++ -*-
   Jeremy Barnes, 5 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Split class to divide a feature set up based upon the value of one feature.
*/

#ifndef __boosting__split_h__
#define __boosting__split_h__

#include "split_fwd.h"
#include "jml/utils/less.h"
#include "feature.h"
#include "jml/db/persistent_fwd.h"
#include <cmath>
#include "jml/arch/exception.h"
#include <iostream>
#include <stdint.h>

namespace ML {

class Feature_Space;
class Feature_Set;
class Feature_Info;
class Optimization_Info;

class Split {
public:
    enum Op {
        LESS,         ///< Test is val(feature) < value
        EQUAL,        ///< Test is val(feature) == value
        NOT_MISSING   ///< Test is always true if feature not missing
    };

    Split()
        : feature_(MISSING_FEATURE), split_val_(0.0f), op_(LESS),
          opt_(false), idx_(0)
    {
        validate();
    }

    Split(const Feature & feature, float split_val, Op op)
        : feature_(feature), split_val_(split_val), op_(op),
          opt_(false), idx_(0)
    {
        validate();
    }

    Split(const Feature & feature, float split_val, const Feature_Space & fs);

    void swap(Split & other)
    {
        std::swap(feature_, other.feature_);
        std::swap(split_val_, other.split_val_);
        std::swap(bits_, other.bits_);
    }

    // Will return true, false or MISSING
    JML_ALWAYS_INLINE int apply(float feature_val) const
    {
        // TODO: optimize to avoid conditionals
        if (isnanf(feature_val)) return MISSING;

        bool equal = feature_val == split_val_;
        bool less = feature_val < split_val_;

        if (JML_UNLIKELY(op_ > NOT_MISSING))
            throw_invalid_op_exception(op());

        int all = (less | (equal << 1) | 4); 

        return (all & (1 << op_)) != 0;
    };

    struct Weights {
        Weights()
        {
            val_[0] = val_[1] = val_[2] = 0.0f;
        }

        JML_ALWAYS_INLINE float & operator [] (int index) { return val_[index]; }
        JML_ALWAYS_INLINE const float & operator [] (int index) const { return val_[index]; }
        
    private:
        float val_[3];
    };

    // Apply to a feature set, returning the amount of weight on true, false
    // and missing
    JML_ALWAYS_INLINE Weights
    apply(const Feature_Set & fset, float weight = 1.0f) const
    {
        Weights result;
        apply(fset, result, weight);
        return result;
    }

    // Same interface, but optimized
    JML_ALWAYS_INLINE Weights
    apply(const float * fset, float weight = 1.0f) const
    {
        if (!opt_)
            throw Exception("Split::apply(): "
                            "wrong method for unoptimized split");
        
        Weights result;
        result[apply(fset[idx_])] = weight;
        return result;
    }

    // Update the given weights with the feature between the given range of
    // iterators
    // PRECONDITION: all iterator vales from first to last (except for last
    // itself) should point to our feature.
    template<class FeatureExPtrIter>
    JML_ALWAYS_INLINE void
    apply(FeatureExPtrIter first,
          const FeatureExPtrIter & last,
          Weights & weights,
          float weight = 1.0f) const
    {

        // Feature missing
        if (JML_UNLIKELY(first == last))
            weights[MISSING] += weight;

        // Feature occurs only once
        else if (JML_LIKELY(first + 1 == last))
            weights[apply(first.value())] += weight;

        // Feature occurs more than once
        else {
            float f = weight / (last - first);
            for (; first != last;  ++first)
                weights[apply(first.value())] += f;
        }
    }

    // Update the given weights with this split
    void apply(const Feature_Set & fset,
               Weights & weights,
               float weight = 1.0f) const;

    // Update the given weights with this split
    void apply(const float * fset,
               Weights & weights,
               float weight = 1.0f) const
    {
        if (!opt_)
            throw Exception("Split::apply(): "
                            "wrong method for unoptimized split");

        weights[apply(fset[idx_])] += weight;
    }

    /** Optimize for the given optimization info. */
    void optimize(const Optimization_Info & info);

    /** Given the feature info for a given feature, figure out which type
        of relationship it should have. */
    static Op get_op_from_feature(const Feature_Info & info);

    bool operator == (const Split & other) const
    {
        return split_val_ == other.split_val_
            && feature_ == other.feature_
            && op_ == other.op_;
    }
    
    bool operator != (const Split & other) const
    {
        return ! operator == (other);
    }

    bool operator < (const Split & other) const
    {
        return less_all(feature_, other.feature_,
                        split_val_, other.split_val_,
                        op_, other.op_);
    }
    
    const Feature & feature() const { return feature_; }
    float split_val() const { return split_val_; }
    Op op() const { return (Op)op_; }

    std::string print(const Feature_Space & fs, int branch = true) const;

    void serialize(DB::Store_Writer & store,
                   const Feature_Space & fs) const;

    void reconstitute(DB::Store_Reader & store,
                      const Feature_Space & fs);

    /** Check that everything is OK; throws an exception if not */
    void validate() const;

    /** Ditto but has access to the feature space. */
    void validate(const Feature_Space & fs) const;

private:
    Feature feature_;      ///< Feature to obtain
    float   split_val_;    ///< Value to test against; **must not be NaN**

    union {
        struct {
            uint32_t op_:8;    ///< What operation to apply?
            uint32_t opt_:1;   ///< Is this optimized for predict?
            uint32_t idx_:23;  ///< Index for an optimized predict
        };
        uint32_t bits_;
    };

    static void throw_invalid_op_exception(Op op) __attribute__((__noreturn__));
};

std::string print(Split::Op op);

std::ostream & operator << (std::ostream & stream, Split::Op op);


} // namespace ML


#endif /* __boosting__split_h__ */
