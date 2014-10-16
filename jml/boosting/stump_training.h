/* stump_training.h                                                -*- C++ -*-
   Jeremy Barnes, 20 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.

   Implementation of the code to train a decision stump classifier.
*/

#ifndef __boosting__stump_training_h__
#define __boosting__stump_training_h__

#include "config.h"
#include <string>
#include <vector>
#include "jml/stats/distribution.h"
#include <boost/multi_array.hpp>
#include "jml/arch/exception.h"
#include <numeric>
#include "stump.h"
#include "jml/algebra/multi_array_utils.h"
#include "jml/math/xdiv.h"
#include "split.h"
#include "fixed_point_accum.h"

namespace ML {


/*****************************************************************************/
/* W ARRAY                                                                   */
/*****************************************************************************/

/* W array. 
   W[l][j][b]

   j = 0 or 1
   l = one of the target labels
   b = +1 or -1 (encoded 1 or 0)

   For label l, W[l][j][b] is the total weight where:

   W[l][1][+1] = predicate holds and l is the correct label
   W[l][0][+1] = predicate doesn't hold and l is the correct label
   W[l][1][-1] = predicate holds and l is not the correct label
   W[l][0][-1] = predicate doesn't hold and l is not the correct label
   W[l][2][+1] = feature is missing and l is the correct label
   W[l][2][-1] = feature is missing and l is not the correct label
*/

template<class Float>
struct W_normalT {
    W_normalT(size_t nl)
        : data(boost::extents[3][2][nl])
    {
    }
    
    W_normalT(const W_normalT & other)
        : data(other.data)
    {
    }

    void swap(W_normalT & other)
    {
        swap_multi_arrays(data, other.data);
    }

    W_normalT & operator = (const W_normalT & other)
    {
        W_normalT new_me(other);
        swap(new_me);
        return *this;
    }

    Float operator () (int l, int cat, bool corr) const
    {
        return data[cat][corr][l];
    }
    
    Float & operator () (int l, int cat, bool corr)
    {
        return data[cat][corr][l];
    }
    
    std::string print() const
    {
        std::string result;
        for (size_t l = 0;  l < nl();  ++l) {
            result += format("  l = %zd:\n", l);
            result
                += format("  t,c:%11.9f t,i:%11.9f",
                          (double)(*this)(l, true, true),
                          (double)(*this)(l, true, false))
                +  format(" f,c:%11.9f f,i:%11.9f",
                          (double)(*this)(l, false, true),
                          (double)(*this)(l, false, false))
                +  format(" m,c:%11.9f m,i:%11.9f",
                          (double)(*this)(l, MISSING, true),
                          (double)(*this)(l, MISSING, false))
                + "\n";
        }
        return result;
    }
    
    size_t nl() const { return data.shape()[2]; }

    /** Add weight to a bucket over all labels.
        \param correct_label The correct label for this training sample.
        \param bucket       The bucket add transfer weight to.
        \param it           An iterator pointing to the start of the range of
                            sample weights.
    */
    template<class Iterator>
    void add(int correct_label, int bucket, Iterator it, int advance)
    {
        for (unsigned l = 0;  l < nl();  ++l) {
            bool corr = correct_label == l;
            (*this)(l, bucket, corr) += *it;
            it += advance;
        }
    }

    /** Add weight to a bucket over all labels, weighted.
        \param correct_label The correct label for this training sample.
        \param bucket       The bucket add transfer weight to.
        \param weight       The weight to add it with.
        \param it           An iterator pointing to the start of the range of
                            sample weights.
    */
    template<class Iterator>
    void add(int correct_label, int bucket, float weight, Iterator it,
             int advance)
    {
        for (unsigned l = 0;  l < nl();  ++l) {
            bool corr = correct_label == l;
            (*this)(l, bucket, corr) += *it * weight;
            it += advance;
        }
    }

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
    void transfer(int correct_label, int from, int to, float weight,
                  Iterator it, int advance)
    {
#if 0
        for (unsigned l = 0;  l < nl();  ++l) {
            Float amount = *(it + advance * l) * weight;
            (*this)(l, from, false) -= amount;
        }

        for (unsigned l = 0;  l < nl();  ++l) {
            Float amount = *(it + advance * l) * weight;
            (*this)(l, to, false) += amount;
        }

        Float amount = *(it + advance * correct_label) * weight;
        (*this)(correct_label, from, false) += amount;
        (*this)(correct_label, from, true)  -= amount;
        (*this)(correct_label, to, false)   -= amount;
        (*this)(correct_label, to, true)    += amount;

#else
        for (unsigned l = 0;  l < nl();  ++l) {
            bool corr = correct_label == l;
            Float amount = (*it) * weight;
            
            (*this)(l, from, corr) -= amount;
            (*this)(l, to,   corr) += amount;

            it += advance;
        }
#endif
    }

    void transfer(int from, int to, const W_normalT & weights)
    {
        for (unsigned l = 0;  l < nl();  ++l) {
            Float amount_true = weights(l, true, true);
            (*this)(l, true,  true) -= amount_true;
            (*this)(l, false, true) += amount_true;
            Float amount_false = weights(l, true, false);
            (*this)(l, true,  false) -= amount_false;
            (*this)(l, false, false) += amount_false;
        }
    }

    /** This function ensures that the values in the MISSING bucket are all
        greater than zero.  They can get less than zero due to rounding errors
        when accumulating. */
    void clip(int bucket)
    {
        using std::max;
        for (unsigned l = 0;  l < nl();  ++l) {
            (*this)(l, bucket, true)
                = max<Float>(0.0, (*this)(l, bucket, true));
            (*this)(l, bucket, false)
                = max<Float>(0.0, (*this)(l, bucket, false));
        }
    }

    /** Swap the weight between the given two buckets. */
    void swap_buckets(int b1, int b2)
    {
        for (unsigned l = 0;  l < nl();  ++l) {
            std::swap((*this)(l, b1, true),  (*this)(l, b2, true));
            std::swap((*this)(l, b1, false), (*this)(l, b2, false));
        }
    }

private:
    boost::multi_array<Float, 3> data;
};

//typedef W_normalT<double> W_normal;
typedef W_normalT<FixedPointAccum64> W_normal;


/*****************************************************************************/
/* Z FORMULA                                                                 */
/*****************************************************************************/

/** Return the Z score.  This is defined as
    \f[
        Z = 2 \sum_{j \in \{0,1\}} \sum_{l \in \mathcal{y}}
        \sqrt{W^{jl}_{+1} W^{jl}_{-1} }
    \f]

    The lower the Z score, the better the fit.
*/

/* Function object to calculate Z for the non-trueonly case. */
struct Z_normal {
    static constexpr double worst   = 1.01;  // worst possible Z value
    static constexpr double none    = 2.0;  // flag to indicate couldn't calculate
    static constexpr double perfect = 0.0;  // best possible Z value

    /* Return the constant missing part. */
    template<class W>
    double missing(const W & w, bool optional) const
    {
        double result = 0.0;
        if (optional) {
            for (unsigned l = 0;  l < w.nl();  ++l)
                result += w(l, MISSING, false) + w(l, MISSING, true);
            return result;
        }
        else {
            for (unsigned l = 0;  l < w.nl();  ++l)
                result += sqrt(w(l, MISSING, false) * w(l, MISSING, true));
            return result * 2.0;
        }
    }

    /* Return the non-missing part. */
    template<class W>
    double non_missing(const W & w, double missing) const
    {
        double result = 0.0;

        for (unsigned l = 0;  l < w.nl();  ++l) {
            result += sqrt(w(l, false, false) * w(l, false, true));
            result += sqrt(w(l, true,  false) * w(l, true,  true));
        }
        return result + result + missing;  // result * 2 + missing
    }

    template<class W>
    double non_missing_presence(const W & w, double missing) const
    {
        double result = 0.0;

        for (unsigned l = 0;  l < w.nl();  ++l)
            result += sqrt(w(l, true,  false) * w(l, true,  true));
        return result + result + missing;  // result * 2 + missing
    }
    
    template<class W>
    double operator () (const W & w) const
    {
        return non_missing(w, missing(w));
    }
    
    /** Return true if it is possible for us to beat the Z score already
        given. */
    template<class W>
    bool can_beat(const W & w, double missing, double z_best) const
    {
        return missing < (1.00001 * z_best);
    }
};


/*****************************************************************************/
/* Z FORMULA FOR CONFIDENCE FUNCTIONS                                        */
/*****************************************************************************/

/** Return the Z score.

    Z = sum(j) 2 w

    The lower the Z score, the better the fit.
*/

/* Function object to calculate Z for the non-trueonly case. */
struct Z_conf {
    static constexpr double worst   = 1.01;  // worst possible Z value
    static constexpr double none    = 2.0;  // flag to indicate couldn't calculate
    static constexpr double perfect = 0.0;  // best possible Z value

    /* Return the constant missing part. */
    template<class W>
    double missing(const W & w, bool optional) const
    {
        double result = 0.0;
        if (optional) {
            for (unsigned l = 0;  l < w.nl();  ++l)
                result += w(l, MISSING, false) + w(l, MISSING, true);
            return result;
        }
        else {
            for (unsigned l = 0;  l < w.nl();  ++l) {
                float den = w(l, MISSING, false) + w(l, MISSING, true);
                result += xdiv<float>(w(l, MISSING, true), den)
                    * sqrt(w(l, MISSING, false) * w(l, MISSING, true));
            }
            return result * 2.0;
        }
    }

    /* Return the non-missing part. */
    template<class W>
    double non_missing(const W & w, double missing) const
    {
        double result = 0.0;

        for (unsigned l = 0;  l < w.nl();  ++l) {
            float den = w(l, false, false) + w(l, false, true);
            result += xdiv<float>(w(l, false, true), den)
                * sqrt(w(l, false, false) * w(l, false, true));
            
            den = w(l, true, false) + w(l, true, true);
            result += xdiv<float>(w(l, true, true), den)
                * sqrt(w(l, true,  false) * w(l, true,  true));
        }
        return result + result + missing;  // result * 2 + missing
    }

    /* Return the non-missing part. */
    template<class W>
    double non_missing_presence(const W & w, double missing) const
    {
        return non_missing(w, missing);
    }

    template<class W>
    double operator () (const W & w) const
    {
        return non_missing(w, missing(w));
    }
    
    /** Return true if it is possible for us to beat the Z score already
        given. */
    template<class W>
    bool can_beat(const W & w, double missing, double z_best) const
    {
        return missing < (1.00001 * z_best);
    }
};


/*****************************************************************************/
/* C FORMULA                                                                 */
/*****************************************************************************/

/** Return the C scores.  The vector returned will have 3 rows, which are
    each a set of values over all categories.  The rows correspond to
    "predicate is false", "predicate is true" and "feature is missing".

    \f[
        c_{jl} =
        \frac{1}{2} \ln \left(
            \frac{W^{jl}_{+1} + \epsilon}{W^{jl}_{-1} + \epsilon}
        \right)
    \f]

    The \f$ \epsilon \f$ parameter is a smoothing parameter, which helps with
    both overfitting an numerical difficulaties.  It has been suggested that
    its value be set to \f$ 1 / mk \f$ where m is the number of training
    examples, and k the number of labels (ie, label_count()).
*/

struct C_normal {

    template<class W>
    std::vector<distribution<float> >
    operator() (const W & w, float epsilon, bool optional) const
    {
        std::vector<distribution<float> >
            result(3, distribution<float>(w.nl(), 0.0));
        
        for (unsigned j = 0;  j < 3 - optional;  ++j) {
            for (unsigned l = 0;  l < w.nl();  ++l) {
                result[j][l] = 0.5
                    * log ((w(l, j, true) + epsilon)
                           / (w(l, j, false) + epsilon));
            }
        }
        return result;
    }

    template<class W>
    void
    operator() (float * dist, int forwhich,
                const W & w, float epsilon, bool optional) const
    {
        for (unsigned l = 0;  l < w.nl();  ++l) {
            dist[l] = 0.5
                * log ((w(l, forwhich, true) + epsilon)
                       / (w(l, forwhich, false) + epsilon));
        }
    }

    Stump::Update update_alg() const { return Stump::NORMAL; }
};

/** Return C which are simply the probability of the given class being
    correct given the split  The vector returned will have 3 rows, which are
    each a set of values over all categories.  The rows correspond to
    "predicate is false", "predicate is true" and "feature is missing".
*/

struct C_prob {

    template<class W>
    std::vector<distribution<float> >
    operator() (const W & w, float epsilon, bool optional) const
    {
        std::vector<distribution<float> >
            result(3, distribution<float>(w.nl(), 0.0));
        
        for (unsigned j = 0;  j < 3 - optional;  ++j)
            for (unsigned l = 0;  l < w.nl();  ++l)
                result[j][l] 
                    = xdiv(w(l, j, true),
                           w(l, j, true) + w(l, j, false));
        return result;
    }

    template<class W>
    void
    operator() (float * dist, int forwhich,
                const W & w, float epsilon, bool optional) const
    {
        for (unsigned l = 0;  l < w.nl();  ++l)
            dist[l] 
                = xdiv(w(l, forwhich, true),
                       w(l, forwhich, true) + w(l, forwhich, false));
    }

    Stump::Update update_alg() const { return Stump::PROB; }
};

/** A gentler version, that results in less violent updates.  See Friedman,
    Hastie and Tibshirani (1999): "Additive Logistic Regression: a Statistical
    view of Boosting.  The update here ranges between -1 and +1, rather than
    the nearly unbounded (except for epsilon) values of the other one.
*/

struct C_gentle {

    template<class W>
    std::vector<distribution<float> >
    operator() (const W & w, float epsilon, bool optional) const
    {
        std::vector<distribution<float> >
            result(3, distribution<float>(w.nl(), 0.0));
        
        for (unsigned j = 0;  j < 3 - optional;  ++j)
            for (unsigned l = 0;  l < w.nl();  ++l)
                result[j][l] 
                    = xdiv(w(l, j, true) - w(l, j, false),
                           w(l, j, true) + w(l, j, false));
        return result;
    }

    template<class W>
    void
    operator() (float * dist, int forwhich,
                const W & w, float epsilon, bool optional) const
    {
        for (unsigned l = 0;  l < w.nl();  ++l)
            dist[l] 
                = xdiv(w(l, forwhich, true) - w(l, forwhich, false),
                       w(l, forwhich, true) + w(l, forwhich, false));
    }

    Stump::Update update_alg() const { return Stump::GENTLE; }
};

/** A switched version of the C function. */
struct C_any {
    C_any(Stump::Update alg) : alg(alg) {}

    template<class W>
    std::vector<distribution<float> >
    operator() (const W & w, float epsilon, bool optional) const
    {
        if (alg == Stump::NORMAL) return C_normal()(w, epsilon, optional);
        else if (alg == Stump::GENTLE) return C_gentle()(w, epsilon, optional);
        else if (alg == Stump::PROB) return C_prob()(w, epsilon, optional);
        else throw Exception("C_any: unknown update alg");
    }    

    // Simpler version that only calculates one part and doesn't allocate lots
    // of memory.  The forwhich parameter should be false, true or MISSING.
    // Note that dist must already be the right size (nl entries).
    template<class W>
    void
    operator() (float * dist, int forwhich,
                const W & w, float epsilon, bool optional) const
    {
        if (alg == Stump::NORMAL)
            return C_normal()(dist, forwhich, w, epsilon, optional);
        else if (alg == Stump::GENTLE)
            return C_gentle()(dist, forwhich, w, epsilon, optional);
        else if (alg == Stump::PROB)
            return C_prob()(dist, forwhich, w, epsilon, optional);
        else throw Exception("C_any: unknown update alg");
    }    

    Stump::Update update_alg() const { return alg; }

    Stump::Update alg;
};

} // namespace ML



#endif /* __boosting__stump_training_h__ */

