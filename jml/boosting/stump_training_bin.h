/* stump_training_bin.h                                           -*- C++ -*-
   Jeremy Barnes, 20 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$
   
   We can use this one when:
   1.  There are 2 labels;
   2.  The weights for each label for each example are equal;
   3.  There is exactly one correct answer for each label;
   4.  Each feature occurs exactly 0 or 1 times for each example.
   
   It is quite a lot faster, as these assumptions knock off a lot of work.
*/

#ifndef __boosting__stump_training_bin_h__
#define __boosting__stump_training_bin_h__

#include "config.h"
#include "jml/compiler/compiler.h"
#include "jml/arch/exception.h"

#include "jml/arch/tick_counter.h"
#include "jml/arch/format.h"
#include <cstddef>

#include <string>

#include "split_fwd.h"
#include "fixed_point_accum.h"

namespace ML {

extern double non_missing_ticks;
extern size_t non_missing_calls;


/*****************************************************************************/
/* W_BINSYM                                                                  */
/*****************************************************************************/

/** We specialise this one for the two-label case:
    - We don't need to keep values for both labels, since the weights are
      equal
    - We don't need to keep variably sized structures, as we know that
      there is only one label, which makes addressing calculations simpler
*/

template<class Float>
struct W_binsymT {

    W_binsymT(size_t nl)
    {
        if (nl != 2)
            throw Exception("Attempt to use W_binsym for a non-binary "
                            "problem");
        for (unsigned cat = 0;  cat <= MISSING;  ++cat)
            for (unsigned corr = 0;  corr <= true;  ++corr)
                data[cat][corr] = 0.0;
    }
    
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    Float operator () (int l, int cat, bool corr) const
    {
        return data[cat][corr ^ (l != 0)];
    }
    
    JML_ALWAYS_INLINE  JML_COMPUTE_METHOD
    Float & operator () (int l, int cat, bool corr)
    {
        //if (l != 0) throw Exception("expected l == 0");
        return data[cat][corr];
    }
    
#ifndef JML_COMPILER_NVCC
    std::string print() const
    {
        std::string result;
        for (size_t l = 0;  l < nl();  ++l) {
            result += format("  l = %zd:\n", l);
            result
                +=format("    t,c:%6.4f t,i:%6.4f",
                         (double)(*this)(l, true, true),
                         (double)(*this)(l, true, false))
                + format("    f,c:%6.4f f,i:%6.4f",
                         (double)(*this)(l, false, true),
                         (double)(*this)(l, false, false))
                + format("    m,c:%6.4f m,i:%6.4f",
                         (double)(*this)(l, MISSING, true),
                         (double)(*this)(l, MISSING, false))
                + "\n";
        }
        return result;
    }
#endif
    
    JML_ALWAYS_INLINE  JML_COMPUTE_METHOD
    size_t nl() const { return 2; }
    
    /** Add weight to a bucket over all labels.
        \param correct_label The correct label for this training sample.
        \param bucket       The bucket add transfer weight to.
        \param it           An iterator pointing to the start of the range of
                            sample weights.
    */
    template<class Iterator>
    JML_ALWAYS_INLINE  JML_COMPUTE_METHOD
    void add(int correct_label, int bucket, Iterator it, int advance)
    {
        data[bucket][!correct_label] += *it;
    }

    template<class Iterator>
    JML_ALWAYS_INLINE  JML_COMPUTE_METHOD
    void atomic_add(int correct_label, int bucket, Iterator it, int advance);
    
    /** Add weight to a bucket over all labels, weighted.
        \param correct_label The correct label for this training sample.
        \param bucket       The bucket add transfer weight to.
        \param weight       The weight to add it with.
        \param it           An iterator pointing to the start of the range of
                            sample weights.
    */
    template<class Iterator>
    JML_ALWAYS_INLINE  JML_COMPUTE_METHOD
    void add(int correct_label, int bucket, float weight, Iterator it,
             int advance)
    {
        data[bucket][!correct_label] += *it * weight;
    }

    template<class Iterator>
    JML_ALWAYS_INLINE  JML_COMPUTE_METHOD
    void atomic_add(int correct_label, int bucket, float weight, Iterator it,
                    int advance);

    /** Transfer weight from one bucket to another over all labels.
        \param label        The correct label for this training sample.
        \param from         The bucket to transfer weight from.
        \param to           The bucket to transfer weight to.
        \param weight       The amount by which to scale the weights from
                            the weight array.
        \param it           An iterator pointing to the start of the range of
                            sample weights.
    */
    template<class Iterator>
    JML_ALWAYS_INLINE  JML_COMPUTE_METHOD
    void transfer(int label, int from, int to, float weight, Iterator it,
                  int advance)
    {
        double amount = (*it) * weight;
        data[from][!label] -= amount;
        data[to  ][!label] += amount;
    }

    JML_COMPUTE_METHOD
    void transfer(int from, int to, const W_binsymT & weights)
    {
        double amount_true = weights.data[true][true];
        data[true][true] -= amount_true;
        data[false][true] += amount_true;
        double amount_false = weights.data[true][false];
        data[true][false] -= amount_false;
        data[false][false] += amount_false;
    }

    /** This function ensures that the values in the MISSING bucket are all
        greater than zero.  They can get less than zero due to rounding errors
        when accumulating. */
    JML_ALWAYS_INLINE JML_COMPUTE_METHOD
    void clip(int bucket)
    {
#ifndef JML_COMPILER_NVCC
        using std::max;
#endif
        data[bucket][true]  = max<Float>(0.0, data[bucket][true]);
        data[bucket][false] = max<Float>(0.0, data[bucket][false]);
    }
    
    JML_ALWAYS_INLINE  JML_COMPUTE_METHOD
    void swap_buckets(int b1, int b2)
    {
#ifndef JML_COMPILER_NVCC
        using std::swap;
#endif
        swap(data[b1][true],  data[b2][true]);
        swap(data[b1][false], data[b2][false]);
    }

private:
    Float data[3][2];  // stored directly in the class, should be speedy
};

//typedef W_binsymT<double> W_binsym;
typedef W_binsymT<FixedPointAccum64> W_binsym;

struct Z_binsym {

    static constexpr double worst = 1.0;
    static constexpr double none = 2.0;
    static constexpr double perfect = 0.0;  // best possible Z value

    template<class W>
    JML_ALWAYS_INLINE
    double operator () (const W & w, bool optional = false) const
    {
        return non_missing(w, missing(w, optional));
    }

    /* Return the constant missing part. */
    template<class W>
    JML_ALWAYS_INLINE
    double missing(const W & w, bool optional = false) const
    {
        if (optional) return 2.0 * (w(0, MISSING, false) + w(0, MISSING, true));
        return 4.0 * sqrt(w(0, MISSING, false) * w(0, MISSING, true));
    }
    
    /* Return the non-missing part. */
    template<class W>
    JML_ALWAYS_INLINE
    double non_missing(const W & w, double missing) const
    {
#ifndef JML_COMPILER_NVCC
        double before = ticks();
#endif // JML_COMPILER_NVCC

        double result = 0.0;
        result += sqrt(w(0, false, false) * w(0, false, true));
        result += sqrt(w(0, true,  false) * w(0, true,  true));

        result = result * 4.0 + missing;

#ifndef JML_COMPILER_NVCC
        non_missing_ticks += ticks() - before - ticks_overhead;
        non_missing_calls += 1;
#endif // JML_COMPILER_NVCC

        return result;
    }
    
    /* Return the non-missing part. */
    template<class W>
    JML_ALWAYS_INLINE
    double non_missing_presence(const W & w, double missing) const
    {
        double result = 0.0;
        result += sqrt(w(0, true,  false) * w(0, true,  true));
        return result * 4.0 + missing;
    }
    
    /** Return true if it is possible for us to beat the Z score already
        given. */
    template<class W>
    JML_ALWAYS_INLINE
    bool can_beat(const W & w, double missing, double z_best) const
    {
        return (missing <= (z_best * 1.00001));
    }
};


} // namespace ML


#endif /* __boosting__stump_training_bin_h__ */

