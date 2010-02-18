/* glz.h                                                           -*- C++ -*-
   Jeremy Barnes, 26 February 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Generalised linear modeling.  Contains functions to calculate the link
   functions and their inverses, to solve for the model given a set of
   variables, and to save and load models to and from disk.  (Well, it
   might eventually...)
*/

#ifndef __stats__glz_h__
#define __stats__glz_h__

#include <cmath>
#include <utility>
#include "jml/stats/distribution.h"
#include "jml/stats/distribution_ops.h"
#include <boost/multi_array.hpp>
#include "irls.h"
#include "least_squares.h"
#include <iomanip>

namespace ML {

/*****************************************************************************/
/* LINK FUNCTIONS                                                            */
/*****************************************************************************/

/* These classes give all of the information necessary to calculate something
   based upon the link function.
*/

/** Implements the logistics link function.
    This class implements the logistics link function:
    \f[
        \eta = \log \left( \frac{\mu}{1 - \mu} \right) 
    \f]

    \f[
        \mu = \exp \left( \frac{\eta}{1 + \eta} \right)
    \f]

    \f[
        \frac{d \eta}{d \mu} = \frac{1}{\mu (1-\mu)}
    \f]
 */

template<class Float>
struct Logit_Link {

    typedef distribution<Float> Vector;

    static std::pair<Float, Float> domain()
    {
        return std::make_pair(0.0, 1.0);
    }

    static std::pair<Float, Float> range()
    {
        return std::make_pair(-INFINITY, INFINITY);
    }

    static Float forward(Float mu)
    {
        return std::log(mu / (1.0 - mu));
    }

    static Vector forward(Vector mu)
    {
        mu = bound(mu, (Float)0.0001, (Float)0.9999);
        return log(mu / ((Float)1.0 - mu));
    }

    static Float inverse(Float eta)
    {
        //if (!isfinite(e)) throw Exception("Logit_Link::inverse: not finite");
        Float e = std::exp(eta);
        if (!std::isfinite(e)) return 0.99999;
        return e / (1.0 + e);
    }

    static Vector inverse(Vector eta)
    {
        eta = bound(eta, (Float)-10.0, (Float)10.0);
        Vector e = exp(eta);
        return e / ((Float)1.0 + e);
    }

    static Float diff(Float mu)
    {
        return 1.0 / (mu * (1.0 - mu));
    }

    static Vector diff(Vector mu)
    {
        mu = bound(mu, (Float)0.0001, (Float)0.9999);
        Vector result = (Float)1.0 / (mu * ((Float)1.0 - mu));
        //cerr << "result[0] = " << result << "  mu[0] = " << mu[0] << endl;
        return result;
    }
};



/** Implements the probit link function.  Used for binomial variables.
    Not finished, as it requires the error function to be evaluated and we
    don't yet have a function to do so.
*/

template<class Float>
struct Probit_Link {

    typedef distribution<Float> Vector;

    static std::pair<Float, Float> domain()
    {
        return std::make_pair(0.0, 1.0);
    }

    static std::pair<Float, Float> range()
    {
        return std::make_pair(-INFINITY, INFINITY);
    }

    static Float forward(Float mu, Float m)
    {
        static const double SQRT2 = std::sqrt(2.0);
        static const double RSQRT2 = 1.0 / SQRT2;

        return SQRT2 * erfinv(2.0 * (mu / m) - 1.0);
    }

    static Float forward(Float mu)
    {
        static const double SQRT2 = std::sqrt(2.0);

        return SQRT2 * erfinv(2.0 * mu - 1.0);
    }

    static Vector forward(const Vector & mu, const Vector & m)
    {
        Vector result = m;
        for (unsigned i = 0;  i < result.size();  ++i)
            result[i] = forward(mu[i], m[i]);
        return result;
    }

    static Vector forward(const Vector & mu)
    {
        Vector result = mu;
        for (unsigned i = 0;  i < result.size();  ++i)
            result[i] = forward(mu[i]);
        return result;
    }

    static Float inverse(Float eta, Float m)
    {
        static const double SQRT2 = std::sqrt(2.0);
        static const double RSQRT2 = 1.0 / SQRT2;

        return m * (1.0 + erf(eta * RSQRT2)) * 0.5;
    }

    static Float inverse(Float eta)
    {
        static const double SQRT2 = std::sqrt(2.0);
        static const double RSQRT2 = 1.0 / SQRT2;

        return (1.0 + erf(eta * RSQRT2)) * 0.5;
    }

    static Vector inverse(const Vector & eta, const Vector & m)
    {
        Vector result = m;
        for (unsigned i = 0;  i < result.size();  ++i)
            result[i] = inverse(eta[i], m[i]);
        return result;
    }

    static Vector inverse(const Vector & eta)
    {
        Vector result = eta;
        for (unsigned i = 0;  i < result.size();  ++i)
            result[i] = inverse(eta[i]);
        return result;
    }

    static Float diff(Float mu, Float m)
    {
        static const double SQRT2 = std::sqrt(2.0);
        static const double RSQRT2 = 1.0 / SQRT2;
        static const double SQRT2PI = std::sqrt(2.0 * M_PI);
        
        return (SQRT2PI / m) * exp(sqr(erfinv((2.0 * mu / m) - 1.0)));
    }

    JML_ALWAYS_INLINE static Float sqr(Float x) { return x * x; }
    
    static Float diff(Float mu)
    {
        static const double SQRT2PI = std::sqrt(2.0 * M_PI);
        if (mu < 0.0001) mu = 0.0001;

        Float result
            = SQRT2PI * std::exp(sqr(erfinv((2.0 * mu) - 1.0)));
        if (!std::isfinite(result))
            throw Exception(format("Probit_Link::diff(): diff(%f) = %f",
                                   mu, result));
        return result;
    }

    static Vector diff(const Vector & mu, const Vector & m)
    {
        Vector result = m;
        for (unsigned i = 0;  i < result.size();  ++i)
            result[i] = diff(mu[i], m[i]);
        return result;
    }

    static Vector diff(const Vector & mu)
    {
        Vector result = mu;
        for (unsigned i = 0;  i < result.size();  ++i)
            result[i] = diff(mu[i]);
        return result;
    }
};

/** Implements the complementary log-log link function.
*/

template<class Float>
struct Comp_Log_Log_Link {

    typedef distribution<Float> Vector;

    static std::pair<Float, Float> domain()
    {
        return std::make_pair(0.0, 1.0);
    }

    static std::pair<Float, Float> range()
    {
        return std::make_pair(-INFINITY, INFINITY);
    }

    static Float forward(Float mu)
    {
        return std::log(-std::log(1.0 - mu));
    }

    static Vector forward(const Vector & mu)
    {
        Vector result = mu;
        for (unsigned i = 0;  i < result.size();  ++i)
            result[i] = forward(mu[i]);
        return result;
    }

    static Float inverse(Float eta)
    {
        return 1.0 - std::exp(-std::exp(eta));
    }

    static Vector inverse(const Vector & eta)
    {
        Vector result = eta;
        for (unsigned i = 0;  i < result.size();  ++i)
            result[i] = inverse(eta[i]);
        return result;
    }

    static Float diff(Float mu, Float m = 1.0)
    {
        if (std::abs(mu - m) < 0.0001) mu -= 0.01;
        mu = std::max(mu, (Float)0.01);
        return 1.0 / ((mu - m) * std::log(1.0 - (mu / m)));
    }

    static Vector diff(const Vector & mu)
    {
        Vector result = mu;
        for (unsigned i = 0;  i < result.size();  ++i)
            result[i] = diff(mu[i]);
        return result;
    }
};


template<class Float>
struct Logarithm_Link {
    
    typedef distribution<Float> Vector;

    static std::pair<Float, Float> domain()
    {
        return std::make_pair(0.0, INFINITY);
    }

    static std::pair<Float, Float> range()
    {
        return std::make_pair(-INFINITY, INFINITY);
    }

    static Float forward(Float mu)
    {
        if (mu <= 0) mu == 1;
        return std::log(mu);
    }

    static Vector forward(Vector mu)
    {
        mu = bound(mu, (Float)0.001, (Float)INFINITY);
        return log(mu);
    }

    static Float inverse(Float eta)
    {
        return std::exp(eta);
    }

    static Vector inverse(Vector eta)
    {
        eta = bound(eta, (Float)-10.0, (Float)10.0);
        return exp(eta);
    }

    static Float diff(Float mu)
    {
        if (mu <= 0) mu = 0.00001;
        return 1.0 / mu;
    }
    
    static Vector diff(Vector mu)
    {
        mu = bound(mu, (Float)0.0001, (Float)INFINITY);
        Vector result = (Float)1.0 / mu;
        return result;
    }
};

template<class Float>
struct Linear_Link {
    
    typedef distribution<Float> Vector;

    static std::pair<Float, Float> domain()
    {
        return std::make_pair(-INFINITY, INFINITY);
    }

    static std::pair<Float, Float> range()
    {
        return std::make_pair(-INFINITY, INFINITY);
    }

    static Float forward(Float mu)
    {
        return mu;
    }

    static const Vector & forward(const Vector & mu)
    {
        return mu;
    }

    static Float inverse(Float eta)
    {
        return eta;
    }

    static const Vector & inverse(const Vector & eta)
    {
        return eta;
    }

    static Float diff(Float mu)
    {
        return 1.0;
    }
    
    static Vector diff(const Vector & mu)
    {
        return Vector(mu.size(), 1.0);
    }
};


/*****************************************************************************/
/* DISTRIBUTIONS                                                             */
/*****************************************************************************/

/** The binomial distribution.  This is the error distribution of choice for
    probabilistic phenomena.  It is usually paired with the logit, probit or
    complementary log-log distributions (see above).
*/

template<class Float>
struct Binomial_Dist {

    typedef distribution<Float> Vector;

    /** Calculates the variance:
        \f[
            \sigma^2 = \mu - \frac{\mu^2}{m};
        \f]
     */
    static Float variance(Float mu, Float m = 1.0)
    {
        return mu - (mu * mu / m);
    }

    /** Calculates the variance:
        \f[
            \sigma^2 = \mu - \frac{\mu^2}{m};
        \f]
     */
    static Vector variance(Vector mu, Vector m)
    {
        return mu - (mu * mu / m);
    }

    /** Calculates the variance:
        \f[
            \sigma^2 = \mu - \mu^2;
        \f]
     */
    static Vector variance(Vector mu)
    {
        mu = min(max(mu, (Float)0.0001), (Float)0.9999);
        return mu - (mu * mu);
    }

    /** Calculates the deviance:
        \f[
            r = 2 \sum w_i \left[ y \log \left( \frac{y}{\mu} \right) \right]
                           \left[ (1-y) \log \left( \frac{1-y}{1-\mu} \right)
                           \right]
        \f]
        of a sample y with respect to the binomial distribution.
     */
    static Float deviance(const Vector & y, const Vector & mu,
                          const Vector & weights)
    {
        Vector part1 = y * log(y / mu);

        /* Any where y == 0 will give NaN even though lim(y->0) y log y = 0
           so we fix it here. */
        for (unsigned i = 0;  i < y.size();  ++i)
            if (y[i] == 0) part1[i] = 0;
        Vector onemy = (Float)1.0 - y;
        Vector part2
            = (onemy) * log(onemy / ((Float)1.0 - mu));

        /* Ditto above. */
        for (unsigned i = 0;  i < y.size();  ++i)
            if (y[i] == 1.0) part2[i] = 0;
        return 2.0 * (weights * (part1 + part2)).total();
    }
};

/** The binomial distribution.  This is the error distribution of choice for
    probabilistic phenomena.  It is usually paired with the logit, probit or
    complementary log-log distributions (see above).
*/

template<class Float>
struct Normal_Dist {

    typedef distribution<Float> Vector;

    /** Calculates the variance:
        \f[
            \sigma^2 = \mu - \frac{\mu^2}{m};
        \f]
     */
    static Float variance(Float mu, Float m = 1.0)
    {
        return 1.0;
    }

    /** Calculates the variance:
        \f[
            \sigma^2 = \mu - \frac{\mu^2}{m};
        \f]
     */
    static Vector variance(Vector mu, Vector m)
    {
        return Vector(mu.size(), 1.0);
    }

    /** Calculates the variance:
        \f[
            \sigma^2 = \mu - \mu^2;
        \f]
     */
    static Vector variance(Vector mu)
    {
        return Vector(mu.size(), 1.0);
    }

    /** Calculates the deviance:
        \f[
            r = 2 \sum w_i \left[ y \log \left( \frac{y}{\mu} \right) \right]
                           \left[ (1-y) \log \left( \frac{1-y}{1-\mu} \right)
                           \right]
        \f]
        of a sample y with respect to the binomial distribution.
     */
    static Float deviance(const Vector & y, const Vector & mu,
                          const Vector & weights)
    {
        return weights.dotprod(sqr(y - mu));
    }
};


/*****************************************************************************/
/* GLZ                                                                       */
/*****************************************************************************/

/** This is the class that holds a generalized linear model.  It contains the
    methods necessary to train and apply it, as well as save to and load from
    a disk.
*/

template<class Link, class Dist, typename Float>
class GLZ {
public:
    GLZ(Link link = Link(), Dist dist = Dist())
        : link(link), dist(dist) {}
    
    /** Apply the GLZ to a given parameter vector. */
    Float apply(const distribution<Float> & params) const
    {
        if (params.size() != b.size())
            throw Exception("glz parameter size mismatch");

        return link.inverse((params * b).total());
    }

    /** Train (weighted) the GLZ from a parameter matrix x and a set of
        responses y.
    */
    void train(const distribution<Float> & y,
               const boost::multi_array<Float, 2> & x,
               const distribution<Float> & w)
    {
        b = ML::irls(y, x, w, link, dist);
    }

    /** Train (unweighted) the GLZ from a parameter matrix x and a set of
        responses y.
    */
    void train(const distribution<Float> & y,
               const boost::multi_array<Float, 2> & x)
    {
        distribution<Float> w(y.size(), 1.0);
        b = ML::irls(y, x, w, link, dist);
    }
    
    distribution<Float> b;  ///< fit parameters
    Link link;              ///< the link function object
    Dist dist;              ///< the distribution object
};


} // namespace ML



#endif /* __stats__glz_h__ */
