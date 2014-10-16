/* training_index_iterators.h                                      -*- C++ -*-
   Jeremy Barnes, 19 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Iterators over a training index.
*/

#ifndef __boosting__training_index_iterator_h__
#define __boosting__training_index_iterator_h__

#include "jml/compiler/compiler.h"
#include <vector>
#include <cmath>

#include <string>
#include <stdint.h>
#include "jml/db/persistent_fwd.h"
#include "jml/arch/exception.h"
#include "label.h"

namespace ML {

class Index_Iterator;


/*****************************************************************************/
/* JOINT_INDEX                                                               */
/*****************************************************************************/

class Joint_Index {
public:
    Joint_Index(const float * values, const uint16_t * buckets,
                const Label * labels, const unsigned * examples,
                const unsigned * counts, const float * divisors,
                unsigned size, const std::vector<float> * bucket_vals);

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator begin() const;

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator end() const;

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    size_t size() const { return size_; }

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    bool empty() const { return size_ == 0; }
    
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator front() const;

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator back() const;

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator operator [] (int idx) const;
    
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    size_t bucket_count() const { return bucket_vals_->size() + 1; }

    const std::vector<float> & bucket_vals() const { return *bucket_vals_; }

#ifndef JML_COMPILER_NVCC
    void dump(std::ostream & stream) const;
#endif // JML_COMPILER_NVCC

    /** Points to the array of values. */
    const float * values() const { return values_; }

    /** Points to the array of buckets. */
    const uint16_t * buckets() const { return buckets_; }

    /** Points to the array of labels (for the target feature) */
    const Label * labels() const { return labels_; }

    /** Points to the array of labels (for the target feature) */
    const int * labels_as_int() const { return (const int *)labels_; }

    /** Points to the array of example numbers.  If null, then the example
        numbers increment monotonically.  (This corresponds to the dense()
        and exactly_one() when ordered by example). */
    const uint32_t * examples() const { return examples_; }

    /** Points to the array of example counts.  If null, then this is always
        one.  This corresponds to the exactly_one case. */
    const uint32_t * counts() const { return counts_; }

    /** Points to the array of divisors. */
    const float * divisors() const { return divisors_; }

protected:
    Joint_Index();
    
private:
    /** Points to the array of values. */
    const float * values_;

    /** Points to the array of buckets. */
    const uint16_t * buckets_;

    /** Points to the array of labels (for the target feature) */
    const Label * labels_;

    /** Points to the array of example numbers.  If null, then the example
        numbers increment monotonically.  (This corresponds to the dense()
        and exactly_one() when ordered by example). */
    const uint32_t * examples_;

    /** Points to the array of example counts.  If null, then this is always
        one.  This corresponds to the exactly_one case. */
    const uint32_t * counts_;

    /** Points to the array of divisors. */
    const float * divisors_;

    /** The size of the array. */
    size_t size_;

    /** Pointer to the array of bucket values, if we used them. */
    const std::vector<float> * bucket_vals_;

    friend class Index_Iterator;
};


/*****************************************************************************/
/* INDEX_ITERATOR                                                            */
/*****************************************************************************/

/** Most of the algorithms operate over a (label, value, example, count) index
    sorted either by value or by example number.  This class is an abstraction
    which allows various index entries to be assembled together into this
    virtual structure.
*/

class Index_Iterator {
public:
    typedef int difference_type;
    typedef void value_type;
    typedef void * pointer;
    typedef void reference;
    typedef std::random_access_iterator_tag iterator_category;
    
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator() : index(0), n(0) {}

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator(const Joint_Index * index, unsigned n)
        : index(index), n(n)
    {
    }

    /* Allow for iterator-like semantics. */
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    bool operator == (const Index_Iterator & other) const
    {
        return n == other.n;
    }

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    bool operator != (const Index_Iterator & other) const
    {
        return n != other.n;
    }
    
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator & operator ++ () { ++n;  return *this; }

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator operator ++ (int)
    {
        Index_Iterator result(*this);  operator ++ ();  return result;
    }

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator & operator += (int i) { n += i;  return *this; }

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator & operator -= (int i) { n -= i;  return *this; }

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    ssize_t operator - (const Index_Iterator & other) const
    {
        return n - other.n;
    }

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator operator + (int i) const
    {
        Index_Iterator result = *this;
        result += i;
        return result;
    }

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Index_Iterator operator - (int i) const
    {
        Index_Iterator result = *this;
        result -= i;
        return result;
    }

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    const Index_Iterator & operator * () const
    {
        return *this;
    }

    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    const Index_Iterator * operator -> () const
    {
        return this;
    }

    /** The value of the variable. */
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    float value() const { return index->values_[n]; }
    
    /** Are we missing? */
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    bool missing() const { return isnanf(value()); }
    
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    unsigned bucket() const { return index->buckets_[n]; }

    /** The label of this training example. */
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Label label() const { return index->labels_[n]; }
    
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    int label_as_int() const { return index->labels_[n].label_; }
    
    /** The example number. */
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    unsigned example() const
    {
        if (index->examples_) return index->examples_[n];
        else return n;
    }
    
    /** How many times this feature is duplicated for this particular
        example number. */
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    int example_counts() const
    {
        if (index->counts_) return index->counts_[n];
        else return 1;
    }
    
    /** Returns true if this example only appeared one time. */
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    bool one_example() const
    {
        return !index->counts_ || index->counts_[n] == 1;
    }
    
    /** Reciprocal of example_counts() */
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    double divisor() const
    {
        if (!index->divisors_) return 1.0;
        else return index->divisors_[n];
    }

#ifndef JML_COMPILER_NVCC
    std::string print() const;

    static std::string titles();
#endif // JML_COMPILER_NVCC

private:
    /** The Joint_Index we are looking at */
    const Joint_Index * index;
    
    /** The number of th entry that we are up to. */
    unsigned n;
};


} // namespace ML

#include "training_index_iterators_impl.h"

#endif /* __boosting__training_index_iterator_h__ */

