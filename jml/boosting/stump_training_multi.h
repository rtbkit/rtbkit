/* stump_training_multi.h                                          -*- C++ -*-
   Jeremy Barnes, 29 August 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Optimised version of training for when we have multiple labels (more
   than 7).
*/

#ifndef __boosting__stump_training_multi_h__
#define __boosting__stump_training_multi_h__

#include "stump_training.h"
#include "jml/arch/simd_vector.h"
#include "jml/arch/sse.h"
#include "jml/arch/sse2.h"
#include "jml/algebra/multi_array_utils.h"

#undef USE_SIMD_SSE2
#define USE_SIMD_SSE2 0

namespace ML {


inline void clip_bucket(float * p, int n)
{
    if (false) ;

#if (USE_SIMD_SSE1)

    else if (Arch::has_sse1()) {

        using namespace Arch::SSE;
        
        v4sf zzero = load_const(0.0);
        
        while (n >= 4) {
            v4sf xx = loadaps(p);
            xx      = andps(xx, cmpnleps(xx, zzero));
            storeaps(p, xx);
            p += 4;
            n -= 4;
        }
    }

#endif /* USE_SIMD_SSE1 */
    
    for (unsigned i = 0;  i < n;  ++i)
        p[i] *= (p[i] <= 0.0F);
}

inline void clip_bucket(double * p, int n)
{
    if (false) ;

#if (USE_SIMD_SSE2)

    else if (Arch::has_sse2()) {

        using namespace Arch::SSE2;
        
        v2df zzero = load_const(0.0);
        
        while (n >= 2) {
            v2df xx = loadapd(p);
            xx      = andpd(xx, cmpnlepd(xx, zzero));
            storeapd(p, xx);
            p += 2;
            n -= 2;
        }
    }
#endif /* USE_SIMD_SSE2 */


    for (unsigned i = 0;  i < n;  ++i)
        p[i] *= (p[i] <= 0.0);
}

template<class Float>
inline void clip_val(Float & f)
{
    f = std::max<Float>(0.0, f);
}

inline void
transfer_core(float * from, float * to, const float * weights, float k,
              int label, int nl)
{

    if (false) ;

#if (USE_SIMD_SSE1)

    else if (Arch::has_sse1()) {
        using namespace Arch::SSE;

        static const float idx_f[4] = { 0.0, 1.0, 2.0, 3.0 };

        v4sf idx  = loadups(idx_f);
        v4sf zero = load_const(0.0);
        v4sf kk   = load_const(k);
    
        while (nl >= 4) {
            v4sf ww;
            v4sf ff, tt;
            v4sf iinc, corr;

            corr   = load_const((float)label);
            //cerr   << "corr = " << corr << endl;

            ww     = loadups(weights);
            ww     = mulps(ww, kk);
            iinc   = cmpneqps(idx, corr);
        
            //cerr << "iinc = " << iinc << endl;

            // where it is correct, we don't add anything
            ww     = andps(ww, iinc);

            //cerr << "ww = " << ww << endl;
        
            ff     = loadaps(from);
            //cerr << "ff before " << ff << endl;
            ff     = subps(ff, ww);
            ff     = maxps(zero, ff);
            //cerr << "ff after " << ff << endl;
            storeaps(from, ff);
        
            tt     = loadaps(to);
            tt     = addps(tt, ww);
            storeaps(to, tt);
        
            nl -= 4;
            from += 4;
            to += 4;
            weights += 4;
            label -= 4;
        }
    }

#endif /* USE_SIMD_SSE1 */
    
    for (unsigned l = 0;  l < nl;  ++l) {
        bool corr = label == l;
        double amount = weights[l] * k * (!corr);
        from[l] -= amount;
        clip_val(from[l]);
        to[l] += amount;
    }
}

JML_ALWAYS_INLINE void
transfer_core(double * from, double * to, const float * weights, float k,
              int label, int nl)
{
    if (false) ;

#if (USE_SIMD_SSE2)

    else if (Arch::has_sse2()) {

        using namespace Arch::SSE;
        using namespace Arch::SSE2;

        static const float idx_f[4] = { 0.0, 1.0, 2.0, 3.0 };

        const v4sf idx  = loadups(idx_f);
        const v2df zero = load_const(0.0);
        const v4sf kk   = load_const(k);
    
        while (nl >= 4) {
            v2df ww0, ww1;
            v2df ff0, ff1, tt0, tt1;
            v4sf wws, iinc, corr;

            wws    = loadups(weights);

            corr   = load_const((float)label);
        
            ff0    = loadapd(from);

            wws    = mulps(wws, kk);
            iinc   = cmpneqps(idx, corr);

            ff1    = loadapd(from + 2);
        
            // where it is correct, we don't add anything
            wws    = andps(wws, iinc);
        
            tt0    = loadapd(to);
            ww0    = cvtps2pd(wws);
            wws    = shufps<3, 2, 3, 2>(wws, wws);
            ff0    = subpd(ff0, ww0);
            ww1    = cvtps2pd(wws);
            tt1    = loadapd(to + 2);
            ff0    = maxpd(zero, ff0);
            tt0    = addpd(ww0, tt0);
        
        
            ff1    = subpd(ff1, ww1);
            storeapd(from, ff0);
        
            ff1    = maxpd(zero, ff1);
        
            storeapd(from + 2, ff1);
        
            tt1    = addpd(ww1, tt1);
        
            storeapd(to, tt0);
            storeapd(to + 2, tt1);
        
            nl -= 4;
            from += 4;
            to += 4;
            weights += 4;
            label -= 4;
        }
    }

#endif /* USE_SIMD_SSE2 */
    
    for (unsigned l = 0;  l < nl;  ++l) {
        bool corr = label == l;
        double amount = weights[l] * k * (!corr);
        //cerr << "l = " << l << " from[l] = " << from[l]
        //     << " to[l] = " << to[l] << " corr = " << corr
        //     << " amount = " << amount << endl;
        from[l] -= amount;
        clip_val(from[l]);
        to[l] += amount;
        //cerr << " from[l] = " << from[l] << " to[l] = " << to[l] << endl;
    }
}


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
struct W_multi {
    W_multi(size_t nl)
        : data(boost::extents[3][2][nl]), nl_(nl)
    {
    }
    
    W_multi(const W_multi & other)
        : data(other.data), nl_(other.nl_)
    {
#if 0
        for (unsigned i = 0;  i < 3;  ++i)
            for (unsigned j = 0;  j < 2;  ++j)
                for (unsigned l = nl_;  l < nidx();  ++l)
                    data[i][j][l] = 0.0;
#endif
    }

    void swap(W_multi & other)
    {
        swap_multi_arrays(data, other.data);
        std::swap(nl_, other.nl_);
    }

    W_multi & operator = (const W_multi & other)
    {
        W_multi new_me(other);
        swap(new_me);
        return *this;
    }

    const Float & operator () (int l, int cat, bool corr) const
    {
        return data[cat][corr][l];
    }
    
    Float & operator () (int l, int cat, bool corr)
    {
        return data[cat][corr][l];
    }

    const Float * operator () (int cat, bool corr) const
    {
        return &data[cat][corr][0];
    }
    
    std::string print() const
    {
        std::string result;
        for (size_t l = 0;  l < nl();  ++l) {
            result += format("  l = %zd:\n", l);
            result
                += format("  t,c:%7.5f t,i:%7.5f",
                          (*this)(l, true, true), (*this)(l, true, false))
                +  format(" f,c:%7.5f f,i:%7.5f",
                          (*this)(l, false, true), (*this)(l, false, false))
                +  format(" m,c:%7.5f m,i:%7.5f",
                          (*this)(l, MISSING, true),(*this)(l, MISSING, false))
                + "\n";
        }
        return result;
    }
    
    size_t nl() const { return nl_; }

    size_t nidx() const { return data.dim(2); }

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
#if 1
        Float * p = &(*this)(0, bucket, false);
        if (advance == 1)
            SIMD::vec_add(p, weight, it, p, nl());
        else
            for (unsigned l = 0;  l < nl();  ++l)
                (*this)(l, bucket, false) += it[l * advance] * weight;

        (*this)(correct_label, bucket, false)
            -= it[correct_label * advance] * weight;
        (*this)(correct_label, bucket, true)
            += it[correct_label * advance] * weight;

        clip_val((*this)(correct_label, bucket, false));
#else
        for (unsigned l = 0;  l < nl();  ++l) {
            bool corr = correct_label == l;
            (*this)(l, bucket, corr) += *it * weight;
            it += advance;
        }
#endif
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
    JML_ALWAYS_INLINE
    void transfer(int correct_label, int from, int to, float weight,
                  Iterator it, int advance)
    {
#if 1
        if (advance == 1) {
            
#ifdef DEBUG_STUMP_MULTI
            W_multi before = *this;

            for (unsigned l = 0;  l < nl();  ++l) {
                bool corr = correct_label == l;
                Float amount = it[l] * weight;
                
                before(l, from, corr) -= amount;
                clip_val(before(l, from, corr));
                before(l, to,   corr) += amount;
            }
#endif

            Float * pfrom = &((*this)(0, from, false));
            Float * pto   = &((*this)(0, to,   false));
            transfer_core(pfrom, pto, it, weight, correct_label, nl());

            Float amount = it[correct_label] * weight;
            
            (*this)(correct_label, from, true)  -= amount;
            (*this)(correct_label, to, true)    += amount;
            
            clip_val((*this)(correct_label, from, true ));

#ifdef DEBUG_STUMP_MULTI
            bool error = false;

            for (unsigned l = 0;  l < nl();  ++l) {
                for (unsigned b = 0;  b < 3;  ++b) {
                    for (unsigned t = 0;  t < 2;  ++t) {
                        double me = (*this)(l, b, t);
                        double them = before(l, b, t);

                        if (abs(me - them) / std::max(me, them) > 0.0001) {
                            cerr << "l = " << l << " b = " << b << " t = "
                                 << t
                                 << " me = " << me << " them = " << them
                                 << endl;
                            error = true;
                        }
                    }
                }
            }
           
            if (error) {
                cerr << "this = " << endl << print() << endl;
                cerr << "other = " << endl << before.print()
                     << endl;
                throw Exception("transfer didn't verify");
            }
#endif
        }
        else {
            for (unsigned l = 0;  l < nl();  ++l) {
                bool corr = correct_label == l;
                Float amount = (*it) * weight;
                
                (*this)(l, from, corr) -= amount;
                (*this)(l, to,   corr) += amount;
                
                it += advance;
            }
        }
#elif 0
        if (advance == 1 && false) {
            //W_multi copy(*this);

            Float * p = &((*this)(0, from, false));
            //for (unsigned l = 0;  l < nl();  ++l)
            //    p[l] -= it[l] * weight;
            SIMD::vec_add(p, -weight, it, p, nl());
            p = &((*this)(0, to, false));
            //for (unsigned l = 0;  l < nl();  ++l)
            //    p[l] += it[l] * weight;
            SIMD::vec_add(p,  weight, it, p, nl());
        }
        else {
            for (unsigned l = 0;  l < nl();  ++l) {
                Float amount = *(it + advance * l) * weight;
                (*this)(l, from, false) -= amount;
            }
            
            for (unsigned l = 0;  l < nl();  ++l) {
                Float amount = *(it + advance * l) * weight;
                (*this)(l, to, false) += amount;
            }
        }
        
        Float amount = *(it + advance * correct_label) * weight;

        (*this)(correct_label, from, false) += amount;
        (*this)(correct_label, from, true)  -= amount;
        (*this)(correct_label, to, false)   -= amount;
        (*this)(correct_label, to, true)    += amount;

        clip_val((*this)(correct_label, from, true ));
        clip_val((*this)(correct_label, to,   false));
#else
        for (unsigned l = 0;  l < nl();  ++l) {
            bool corr = correct_label == l;
            Float amount = (*it) * weight;
            
            (*this)(l, from, corr) -= amount;
            (*this)(l, to,   corr) += amount;

            it += advance;
        }
#endif

#ifdef DEBUG_STUMP_MULTI
        for (unsigned l = 0;  l < nl();  ++l)
            for (unsigned b = 0;  b < 3;  ++b)
                for (unsigned t = 0;  t < 2;  ++t)
                    if ((*this)(l, b, t) < 0.0) {
                        cerr << "transfer(" << correct_label << ", "
                             << from << ", " << to << ", " << weight
                             << ", [";
                        for (unsigned l = 0;  l < nl();  ++l)
                            cerr << " " << it[l];
                        cerr << "], 1)" << endl;
                        cerr << print() << endl;
                        throw Exception("bad transfer");
                    }
#endif
    }

    void transfer(int from, int to, const W_multi & weights)
    {
#if 1        
        SIMD::vec_minus(&((*this)(0, true,  true )), &(weights(0, true, true)),
                       &((*this)(0, true,  true )), nl());
        SIMD::vec_add  (&(*this)(0, false, true ), &weights(0, true, true),
                       &(*this)(0, false, true ), nl());
        SIMD::vec_minus(&(*this)(0, true,  false), &weights(0, true, false),
                       &(*this)(0, true,  false), nl());
        SIMD::vec_add  (&(*this)(0, false, false), &weights(0, true, false),
                       &(*this)(0, false, false), nl());
#else        
        for (unsigned l = 0;  l < nl();  ++l) {
            Float amount_true = weights(l, true, true);
            (*this)(l, true,  true) -= amount_true;
            (*this)(l, false, true) += amount_true;
            Float amount_false = weights(l, true, false);
            (*this)(l, true,  false) -= amount_false;
            (*this)(l, false, false) += amount_false;
        }
#endif

#ifdef DEBUG_STUMP_MULTI
        for (unsigned l = 0;  l < nl();  ++l)
            for (unsigned b = 0;  b < 3;  ++b)
                for (unsigned t = 0;  t < 2;  ++t)
                    if ((*this)(l, b, t) < 0.0) {
                        cerr << "transfer(" << from << ", " << to << "):"
                             << endl;
                        cerr << "other: " << endl << weights.print() << endl;
                        cerr << "this: " << print() << endl;
                        throw Exception("bad transfer");
                    }
#endif
    }

    /** This function ensures that the values in the MISSING bucket are all
        greater than zero.  They can get less than zero due to rounding errors
        when accumulating. */

    void clip(int bucket)
    {
#if 0
#if 0
        using namespace Math;
        Float * p = &(*this)(0, bucket, false);
        clip_bucket(p, nl());
        //SIMD::vec_max(p, 0.0, p, nl());
        p = &(*this)(0, bucket, true);
        //SIMD::vec_max(p, 0.0, p, nl());
        clip_bucket(p, nl());
#else
        for (unsigned l = 0;  l < nl();  ++l) {
            (*this)(l, bucket, true)
                = std::max<Float>(0.0, (*this)(l, bucket, true));
            (*this)(l, bucket, false)
                = std::max<Float>(0.0, (*this)(l, bucket, false));
        }
#endif
#endif
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
    boost::multi_array<Float, 3> data;  // alignment on cache boundary
    size_t nl_;
};


/*****************************************************************************/
/* VECTOR IMPLEMENTATIONS                                                    */
/*****************************************************************************/

inline float accum_sum_sqrt_prod(const float * p1, const float * p2, int n)
{
#if 0
    double result1 = 0.0;
    for (unsigned i = 0;  i < n;  ++i)
        result1 += sqrt(p1[i] * p2[i]);
    //return result;
#endif

    //#else

    double result = 0.0;

    if (false) ;

#if (USE_SIMD_SSE1)

    else if (Arch::has_sse1()) {
        using namespace Arch::SSE;

        v4sf tt    = load_const(0.0);

        v4sf zzero = load_const(0.0);
        v4sf hhalf = load_const(0.5);
        //v4sf ttwo  = load_const(2.0);
        v4sf nn, aa, rr;

        while (n >= 4) {
            nn      = loadaps(p1);
            aa      = loadaps(p2);
            nn      = mulps(nn, aa);
            //cerr << "nn = " << nn << endl;
            rr      = VEC_INSN(rsqrtps, (nn));
            //cerr << "rr = " << rr << endl;
            aa      = accurate_recip(rr);
            
            //cerr << "aa = " << aa << endl;

            rr      = mulps(rr, nn);
            //nn      = cmpnleps(nn, zzero);
            //cerr << "zero = " << cmpnleps(nn, zzero) << endl;
            rr      = addps(aa, rr);
            aa      = mulps(hhalf, rr);  // hoisted out of loop
            aa      = andps(aa, cmpnleps(nn, zzero));
            //cerr << "approx = " << aa << endl;
            //cerr << "squared = " << mulps(aa, aa) << endl;
            //cerr << "error = " << subps(nn, mulps(aa, aa)) << endl;
            tt      = addps(tt, aa);
            //cerr << "tt now " << tt << endl;
            p1 += 4;
            p2 += 4;
            n -= 4;
        }

        result = accum_finish(tt);
    }

#endif /* USE_SIMD_SSE1 */    

    for (unsigned i = 0;  i < n;  ++i)
        result += sqrt(p1[i] * p2[i]);
    
    //cerr << "result1 = " << result1 << " result = " << result
    //     << " difference = " << result - result1 << endl;

    return result;
    //#endif
}

inline double accum_sum_sqrt_prod(const double * p1, const double * p2, int n)
{
    double result = 0.0;

    if (false) ;

#if (USE_SIMD_SSE2)

    else if (Arch::has_sse2()) {

        using namespace Arch::SSE;
        using namespace Arch::SSE2;

        v2df tt    = load_const(0.0);

#if 1
        v2df zzero = load_const(0.0);
        v2df hhalf = load_const(0.5);
        //v2df ttwo  = load_const(2.0);
        v2df rr;
        v4sf ss;
#endif
        v2df nn, aa;

        while (n >= 2) {
            nn      = loadupd(p1);
            aa      = loadupd(p2);
            nn      = mulpd(nn, aa);

#if 1
            ss      = VEC_INSN(cvtpd2ps, (nn));
            ss      = VEC_INSN(rsqrtps, (ss));
            rr      = VEC_INSN(cvtps2pd, (ss));
            //cerr << "nn = " << nn << endl;
            //rr      = VEC_INSN(
            //rr      = VEC_INSN(rsqrtpd, (nn));
            //cerr << "rr = " << rr << endl;
            aa      = accurate_recip(rr);
            
            //cerr << "aa = " << aa << endl;

            rr      = mulpd(rr, nn);
            //nn      = cmpnlepd(nn, zzero);
            //cerr << "zero = " << cmpnlepd(nn, zzero) << endl;
            rr      = addpd(aa, rr);
            aa      = mulpd(hhalf, rr);  // hoisted out of loop
            aa      = andpd(aa, cmpnlepd(nn, zzero));
            //cerr << "approx = " << aa << endl;
            //cerr << "squared = " << mulpd(aa, aa) << endl;
            //cerr << "error = " << subpd(nn, mulpd(aa, aa)) << endl;
#else
            aa      = sqrtpd(nn);
#endif
            tt      = addpd(tt, aa);
            //cerr << "tt now " << tt << endl;
            p1 += 2;
            p2 += 2;
            n -= 2;
        }

        result = accum_finish(tt);
    }

#endif /* USE_SIMD_SSE2 */
    
    for (unsigned i = 0;  i < n;  ++i)
        result += sqrt(p1[i] * p2[i]);
    
    //cerr << "result1 = " << result1 << " result = " << result
    //     << " difference = " << result - result1 << endl;
    
    return result;
}



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
template<class Float>
struct Z_multi {
    static constexpr Float worst   = 1.0;  // worst possible Z value
    static constexpr Float none    = 2.0;  // flag to indicate couldn't calculate
    static constexpr Float perfect = 0.0;  // best possible Z value

    /* Return the constant missing part. */
    Float missing(const W_multi<Float> & w, bool optional) const
    {
        Float result = 0.0;
        if (optional) {
            for (unsigned l = 0;  l < w.nl();  ++l)
                result += w(l, MISSING, false) + w(l, MISSING, true);
            return result;
        }
        else {
            result = accum_sum_sqrt_prod(w(MISSING, false), w(MISSING, true),
                                         w.nl());
            //for (unsigned l = 0;  l < w.nl();  ++l)
            //    result += sqrt(w(l, MISSING, false) * w(l, MISSING, true));
            return result * 2.0;
        }
    }

    /* Return the non-missing part. */
    Float non_missing(const W_multi<Float> & w, Float missing) const
    {
        Float result = 0.0;

        result
            = accum_sum_sqrt_prod(w(false, false), w(false, true),
                                  w.nl())
            + accum_sum_sqrt_prod(w(true,  false), w(true,  true),
                                  w.nl());
        return result + result + missing;  // result * 2 + missing
    }

    Float non_missing_presence(const W_multi<Float> & w, Float missing) const
    {
        Float result
            =  accum_sum_sqrt_prod(w(true,  false), w(true,  true), w.nl());
        return result + result + missing;  // result * 2 + missing
    }

    Float operator () (const W_multi<Float> & w) const
    {
        return non_missing(w, missing(w));
    }
    
    /** Return true if it is possible for us to beat the Z score already
        given. */
    bool can_beat(const W_multi<Float> & w, Float missing, Float z_best) const
    {
        return missing < (1.00001 * z_best);
    }
};

} // namespace ML



#endif /* __boosting__stump_training_multi_h__ */


