/* irls.cc
   Jeremy Barnes, 19 March 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Iteratively Reweighted Least Squares functionality.
*/

#include "irls.h"
#include "glz.h"
#include <boost/timer.hpp>
#include "stats/distribution_simd.h"
#include "utils/vector_utils.h"
#include "boosting/config_impl.h"
#include "arch/simd_vector.h"
#include "multi_array_utils.h"
#include "boosting/config_impl.h"
#include "utils/string_functions.h"
#include <boost/version.hpp>

using namespace std;
using namespace Stats;

// TODO:in which boost version was this added?
#if (BOOST_VERSION > 103500)
#include <boost/math/special_functions/erf.hpp>


namespace ML {

__thread std::ostream * debug_irls = 0;

double erf(double x)
{
    return boost::math::erf(x);
}

double erfinv(double y)
{
    return boost::math::erf_inv(y);
}

} // namespace ML

#else

namespace ML {

double erf(double x)
{
    return ::erf(x);
}

double erfinv(double y)
{
    return ::erfinv(y);
}

} // namespace ML

#endif

namespace ML {

template<class FloatIn, class FloatCalc>
vector<int> remove_dependent_impl(boost::multi_array<FloatIn, 2> & x,
                                  std::vector<distribution<FloatCalc> > & y,
                                  double tolerance)
{
    boost::timer t;
    size_t ni = x.shape()[0], nj = x.shape()[1];
    /* Perform an orthogonalization to determine which variables are
       linearly dependent.  This procedure uses the Modified Gram-Schmidt
       process.
    */

    vector<distribution<FloatCalc> > v(ni);  // orthonormal basis vectors
    //distribution<Float> r(ni); // to detect dependent vectors
    vector<distribution<FloatCalc> > r(ni, distribution<FloatCalc>(ni, 0.0));

    // matrix such that v[i] = sum j ( x[j] z[i][j] )
    vector<distribution<FloatCalc> > z(ni, distribution<FloatCalc>(ni, 0.0));

    for (unsigned i = 0;  i < ni;  ++i) {
        v[i] = distribution<FloatCalc>(&x[i][0], &x[i][0] + nj);
        z[i][i] = 1.0;   // because v[i] = x[i]
    }

    for (int i = ni - 1;  i >= 0;  --i) {
        r[i][i] = v[i].two_norm();

        if (r[i][i] < tolerance) {
            z[i][i] = 0.0;
            r[i][i] = 0.0;
            continue;  // linearly dependent
        }

        v[i] /= r[i][i];
        z[i] /= r[i][i];
        

        // Check that our v is orthogonal to all other vs
        for (int i2 = ni - 1;  i2 > i;  --i2) {
            double error = v[i2].dotprod(v[i]);
            if (error > tolerance) {
                cerr << "error: between " << i << " and " << i2 << ": "
                     << error << endl;
            }
        }

        // TODO: re-vectorize

        for (int j = i - 1;  j >= 0;  --j) {
            r[j][i] = v[i].dotprod(v[j]);
            v[j] -= r[j][i] * v[i];

            // Check that v[i] and v[j] are now orthogonal
            double error = v[i].dotprod(v[j]);
            if (error > tolerance)
                cerr << "tolerance: v[i] and v[j] aren't orthogonal"
                     << endl;

            //SIMD::vec_add(&v[j][0], -r[j][i], &v[i][0], &v[j][0], nj);
            z[j] -= r[j][i] * z[i];
        }

        // Check that we can 
    }

    //cerr << "done gram schmidt" << endl;

    //for (unsigned i = 0;  i < ni;  ++i) {
    //    cerr << "r[" << i << "] = " << r[i][i] << "  ";
    //}
    //cerr << endl;

    //for (unsigned i = 0;  i < ni;  ++i) {
    //    cerr << "z[" << i << "] = " << z[i] << endl;
    //}
    
    /* Keep track of what is nonsingular, and where they come from. */
    vector<int> source;
    vector<int> dest(ni, -1);  // -1 indicates it was left out
    for (unsigned i = 0;  i < ni;  ++i) {
        if (r[i][i] >= tolerance) {
            /* Nonsingular. */
            dest[i] = source.size();
            source.push_back(i);
        }
        else {
            //cerr << "column " << i << " is dependent; r vector is "
            //     << r[i] << endl;
            //cerr << "column " << i << " is dependent; basis vector is "
            //     << v[i] << endl;
        }
    }

    y.clear();
    for (unsigned i = 0;  i < ni;  ++i) {
        distribution<FloatCalc> this_y(ni, 0.0);
        for (unsigned j = 0;  j < ni;  ++j)
            this_y += r[i][j] * z[j];
        //cerr << "y[" << i << "] = " << this_y << endl;
        y.push_back(this_y);
    }

    //cerr << "done y" << endl;

    unsigned ni2 = source.size();

    //cerr << "ni = " << ni << " ni2 = " << ni2 << endl; 
        
    if (ni2 < ni) {
        /* We have removed some columns, so we need to rearrange our
           matrix. */
        boost::multi_array<FloatIn, 2> x2(boost::extents[ni2][nj]);
        for (unsigned i = 0;  i < ni2;  ++i)
            for (unsigned j = 0;  j < nj;  ++j)
                x2[i][j] = x[source[i]][j];
        
        swap_multi_arrays(x, x2);

        /* Same for the y matrix.  We remove columns which are dependent. */
        vector<distribution<FloatCalc> > y2(ni, distribution<FloatCalc>(ni2, 0.0));
        for (unsigned i = 0;  i < ni;  ++i)
            for (unsigned j = 0;  j < ni2;  ++j)
                y2[i][j] = y[i][source[j]];
        y.swap(y2);
    }

    //cerr << "remove_dependent: " << ni << "x" << nj << ": " 
    //     << t.elapsed() << "s" << endl;
    //cerr << "dest = " << dest << endl;

    //for (unsigned i = 0;  i < ni;  ++i) {
    //    cerr << "y[" << i << "] = " << y[i] << endl;
    //}

    return dest;
}

template vector<int>
remove_dependent_impl(boost::multi_array<float, 2> & x,
                      std::vector<distribution<float> > & y,
                      double tolerance);
template vector<int>
remove_dependent_impl(boost::multi_array<float, 2> & x,
                      std::vector<distribution<double> > & y,
                      double tolerance);
template vector<int>
remove_dependent_impl(boost::multi_array<double, 2> & x,
                      std::vector<distribution<double> > & y,
                      double tolerance);

vector<int> remove_dependent(boost::multi_array<double, 2> & x)
{
    vector<distribution<double> > y;
    return remove_dependent_impl(x, y);
}

vector<int> remove_dependent(boost::multi_array<float, 2> & x)
{
    vector<distribution<float> > y;
    return remove_dependent_impl(x, y);
}

distribution<double>
irls_logit(const distribution<double> & correct,
           const boost::multi_array<double, 2> & outputs,
           const distribution<double> & w)
{
    return irls(correct, outputs, w, Logit_Link<double>(),
                Binomial_Dist<double>());
}

distribution<double>
irls_log(const distribution<double> & correct,
         const boost::multi_array<double, 2> & outputs,
         const distribution<double> & w)
{
    return irls(correct, outputs, w, Logarithm_Link<double>(),
                Binomial_Dist<double>());
}

distribution<double>
irls_linear(const distribution<double> & correct,
            const boost::multi_array<double, 2> & outputs,
            const distribution<double> & w)
{
    return irls(correct, outputs, w, Linear_Link<double>(),
                Binomial_Dist<double>());
}

distribution<double>
irls_probit(const distribution<double> & correct,
            const boost::multi_array<double, 2> & outputs,
            const distribution<double> & w)
{
    return irls(correct, outputs, w, Probit_Link<double>(),
                Binomial_Dist<double>());
}

distribution<double>
irls_complog(const distribution<double> & correct,
             const boost::multi_array<double, 2> & outputs,
             const distribution<double> & w)
{
    return irls(correct, outputs, w, Comp_Log_Log_Link<double>(),
                Binomial_Dist<double>());
}

distribution<double>
run_irls(const distribution<double> & correct,
         const boost::multi_array<double, 2> & outputs,
         const distribution<double> & w, Link_Function func)
{
    switch (func) {

    case LOGIT:
        return irls_logit(correct, outputs, w);

    case LOG:
        return irls_log(correct, outputs, w);

    case LINEAR:
        return irls_linear(correct, outputs, w);

    case PROBIT:
        return irls_probit(correct, outputs, w);

    case COMP_LOG_LOG:
        return irls_complog(correct, outputs, w);

    default:
        throw Exception(format("run_irls(): function %d "
                               "not implemented", func));
    }
}

distribution<float>
irls_logit(const distribution<float> & correct,
           const boost::multi_array<float, 2> & outputs,
           const distribution<float> & w)
{
    return irls(correct, outputs, w, Logit_Link<float>(),
                Binomial_Dist<float>());
}

distribution<float>
irls_log(const distribution<float> & correct,
         const boost::multi_array<float, 2> & outputs,
         const distribution<float> & w)
{
    return irls(correct, outputs, w, Logarithm_Link<float>(),
                Binomial_Dist<float>());
}

distribution<float>
irls_linear(const distribution<float> & correct,
            const boost::multi_array<float, 2> & outputs,
            const distribution<float> & w)
{
    return irls(correct, outputs, w, Linear_Link<float>(),
                Binomial_Dist<float>());
}

distribution<float>
irls_probit(const distribution<float> & correct,
            const boost::multi_array<float, 2> & outputs,
            const distribution<float> & w)
{
    return irls(correct, outputs, w, Probit_Link<float>(),
                Binomial_Dist<float>());
}

distribution<float>
irls_complog(const distribution<float> & correct,
             const boost::multi_array<float, 2> & outputs,
             const distribution<float> & w)
{
    return irls(correct, outputs, w, Comp_Log_Log_Link<float>(),
                Binomial_Dist<float>());
}

distribution<float>
run_irls(const distribution<float> & correct,
         const boost::multi_array<float, 2> & outputs,
         const distribution<float> & w, Link_Function func)
{
    switch (func) {

    case LOGIT:
        return irls_logit(correct, outputs, w);

    case LOG:
        return irls_log(correct, outputs, w);

    case LINEAR:
        return irls_linear(correct, outputs, w);

    case PROBIT:
        return irls_probit(correct, outputs, w);

    case COMP_LOG_LOG:
        return irls_complog(correct, outputs, w);

    default:
        throw Exception(format("run_irls(): function %d "
                               "not implemented", func));
    }
}

double apply_link_inverse(double val, Link_Function func)
{
    switch (func) {
    case LOGIT:
        return Logit_Link<double>::inverse(val);

    case LOG:
        return Logarithm_Link<double>::inverse(val);
        
    case LINEAR:
        return Linear_Link<double>::inverse(val);
        
    case PROBIT:
        return Probit_Link<double>::inverse(val);
        
    case COMP_LOG_LOG:
        return Comp_Log_Log_Link<double>::inverse(val);
        
    default:
        throw Exception("apply_irls: unknown link function");
    }
}

std::string print(Link_Function link)
{
    switch (link) {
    case LOGIT:        return "LOGIT";
    case LOG:          return "LOG";
    case LINEAR:       return "LINEAR";
    case PROBIT:       return "PROBIT";
    case COMP_LOG_LOG: return "COMP_LOG_LOG";
    default:           return format("Link_Function(%d)", link);
    }
}

Link_Function parse_link_function(const std::string & link_name)
{
    Link_Function link;
    
    if (lowercase(link_name) == "logit")
        link = LOGIT;
    else if (lowercase(link_name) == "log")
        link = LOG;
    else if (lowercase(link_name) == "linear")
        link = LINEAR;
    else if (lowercase(link_name) == "probit")
        link = PROBIT;
    else if (lowercase(link_name) == "comploglog")
        link = COMP_LOG_LOG;
    else throw Exception("parse_link_function(): link function '"
                         + link_name + "' is not known ('logit', 'log', "
                         "'linear', 'probit', 'comploglog' accepted)");
    return link;
}

/** Put a link function in a stream. */
std::ostream & operator << (std::ostream & stream, Link_Function link)
{
    return stream << print(link);
}

COMPACT_PERSISTENT_ENUM_IMPL(Link_Function);
} // namespace ML

ENUM_INFO_NAMESPACE
  
const Enum_Opt<ML::Link_Function>
Enum_Info<ML::Link_Function>::OPT[5] = {
    { "logit",        ML::LOGIT          },
    { "probit",       ML::PROBIT         },
    { "comp_log_log", ML::COMP_LOG_LOG   },
    { "linear",       ML::LINEAR         },
    { "log",          ML::LOG            } };

const char * Enum_Info<ML::Link_Function>::NAME
   = "Link_Function";

END_ENUM_INFO_NAMESPACE

