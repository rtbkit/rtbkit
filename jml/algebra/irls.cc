/* irls.cc
   Jeremy Barnes, 19 March 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Iteratively Reweighted Least Squares functionality.
*/

#include "irls.h"
#include "glz.h"
#include <boost/timer.hpp>
#include "jml/stats/distribution_simd.h"
#include "jml/utils/vector_utils.h"
#include "jml/boosting/config_impl.h"
#include "jml/arch/simd_vector.h"
#include "jml/arch/timers.h"
#include "multi_array_utils.h"
#include "jml/boosting/config_impl.h"
#include "jml/utils/string_functions.h"
#include "jml/algebra/lapack.h"
#include <boost/version.hpp>

using namespace std;

#ifndef UNDER_VALGRIND
#define UNDER_VALGRIND 0
#endif

// TODO:in which boost version was this added?
#if (BOOST_VERSION > 103500 && !UNDER_VALGRIND)
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

__thread std::ostream * debug_remove_dependent = 0;
__thread bool check_remove_dependent = 0;

template<class FloatIn, class FloatCalc>
vector<int> remove_dependent_impl(boost::multi_array<FloatIn, 2> & x,
                                  std::vector<distribution<FloatCalc> > & y,
                                  double tolerance)
{
    boost::timer t;
    size_t nrows = x.shape()[0], ncols = x.shape()[1];
    /* Perform an orthogonalization to determine which variables are
       linearly dependent.  This procedure uses the Modified Gram-Schmidt
       process.
    */

    vector<distribution<FloatCalc> > v(nrows);  // orthonormal basis vectors
    //distribution<Float> r(nrows); // to detect dependent vectors
    vector<distribution<FloatCalc> > r(nrows, distribution<FloatCalc>(nrows, 0.0));

    // matrix such that v[i] = sum j ( x[j] z[i][j] )
    vector<distribution<FloatCalc> > z(nrows, distribution<FloatCalc>(nrows, 0.0));

    for (unsigned i = 0;  i < nrows;  ++i) {
        v[i] = distribution<FloatCalc>(&x[i][0], &x[i][0] + ncols);
        z[i][i] = 1.0;   // because v[i] = x[i]
    }

    for (int i = nrows - 1;  i >= 0;  --i) {
        r[i][i] = v[i].two_norm();

        if (debug_remove_dependent) {
            cerr << "i = " << i << " nrows = " << nrows << " r[i][i] = "
                 << r[i][i] << " v[i] = " << v[i] << " z[i] = "
                 << z[i] << endl;
        }

        if (r[i][i] < tolerance) {
            z[i][i] = 0.0;
            r[i][i] = 0.0;
            continue;  // linearly dependent
        }

        v[i] /= r[i][i];
        z[i] /= r[i][i];
        

        // Check that our v is orthogonal to all other vs
        if (check_remove_dependent || debug_remove_dependent) {
            bool any_errors = false;
            for (int i2 = nrows - 1;  i2 > i;  --i2) {
                double error = v[i2].dotprod(v[i]);
                if (error > tolerance) {
                    cerr << "error: between " << i << " and " << i2 << ": "
                         << error << endl;
                    if (error > tolerance * 10)
                        any_errors = true;
                }
            }
            if (any_errors) {
                cerr << __PRETTY_FUNCTION__ << endl;
                throw Exception("remove_dependent: not independent");
            }
        }

        bool any_errors = false;

        for (int j = i - 1;  j >= 0;  --j) {
            r[j][i] = v[i].dotprod(v[j]);
            SIMD::vec_add(&v[j][0], -r[j][i], &v[i][0], &v[j][0], ncols);
            SIMD::vec_add(&z[j][0], -r[j][i], &z[i][0], &z[j][0], nrows);


            // Check that v[i] and v[j] are now orthogonal
            if (check_remove_dependent || debug_remove_dependent) {
                double error = v[i].dotprod(v[j]);
                if (error > tolerance) {
                    cerr << "v[i] and v[j] aren't orthogonal to tolerance "
                         << tolerance << ": dot_product = "
                         << error << endl;
                    if (error > tolerance * 10)
                        any_errors = true;
                }
            }
        }

        if (any_errors) {
            cerr << __PRETTY_FUNCTION__ << endl;
            throw Exception("remove_dependent: not independent 2");
        }
    }

    if (debug_remove_dependent) {
        cerr << "finished remove_dependent" << endl;
        for (unsigned i = 0;  i < nrows;  ++i) {
            cerr << "i = " << i << " nrows = " << nrows << " r[i][i] = "
                 << r[i][i] << " v[i] = " << v[i] << " z[i] = "
                 << z[i] << endl;
        }
    }
    
    //cerr << "done gram schmidt" << endl;

    //for (unsigned i = 0;  i < nrows;  ++i) {
    //    cerr << "r[" << i << "] = " << r[i][i] << "  ";
    //}
    //cerr << endl;

    //for (unsigned i = 0;  i < nrows;  ++i) {
    //    cerr << "z[" << i << "] = " << z[i] << endl;
    //}
    
    /* Keep track of what is nonsingular, and where they come from. */
    vector<int> source;
    vector<int> dest(nrows, -1);  // -1 indicates it was left out
    for (unsigned i = 0;  i < nrows;  ++i) {
        if (r[i][i] >= tolerance) {
            /* Nonsingular. */
            dest[i] = source.size();
            source.push_back(i);
        }
        else {
            if (debug_remove_dependent) {
                cerr << "column " << i << " is dependent; r " << r[i]
                     << " v " << v[i] << endl;
            }
        }
    }

    if (debug_remove_dependent) {
        cerr << "source = " << source << endl;
        cerr << "dest   = " << dest << endl;
    }

    y.clear();
    for (unsigned i = 0;  i < nrows;  ++i) {
        distribution<FloatCalc> this_y(nrows, 0.0);
        for (unsigned j = 0;  j < nrows;  ++j)
            this_y += r[i][j] * z[j];
        //cerr << "y[" << i << "] = " << this_y << endl;
        y.push_back(this_y);
    }

    //cerr << "done y" << endl;

    unsigned nrows2 = source.size();

    //cerr << "nrows = " << nrows << " nrows2 = " << nrows2 << endl; 
        
    if (nrows2 < nrows) {
        /* We have removed some columns, so we need to rearrange our
           matrix. */
        boost::multi_array<FloatIn, 2> x2(boost::extents[nrows2][ncols]);
        for (unsigned i = 0;  i < nrows2;  ++i)
            for (unsigned j = 0;  j < ncols;  ++j)
                x2[i][j] = x[source[i]][j];
        
        swap_multi_arrays(x, x2);

        /* Same for the y matrix.  We remove columns which are dependent. */
        vector<distribution<FloatCalc> > y2(nrows, distribution<FloatCalc>(nrows2, 0.0));
        for (unsigned i = 0;  i < nrows;  ++i)
            for (unsigned j = 0;  j < nrows2;  ++j)
                y2[i][j] = y[i][source[j]];
        y.swap(y2);
    }

    //cerr << "remove_dependent: " << nrows << "x" << ncols << ": " 
    //     << t.elapsed() << "s" << endl;
    //cerr << "dest = " << dest << endl;

    //for (unsigned i = 0;  i < nrows;  ++i) {
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


/*****************************************************************************/
/* REGRESSOR                                                                 */
/*****************************************************************************/

Regressor::
~Regressor()
{
}


Least_Squares_Regressor::
~Least_Squares_Regressor()
{
}

distribution<float>
Least_Squares_Regressor::
calc(const boost::multi_array<float, 2> & A,
     const distribution<float> & b) const
{
    return least_squares(A, b);
}

distribution<double>
Least_Squares_Regressor::
calc(const boost::multi_array<double, 2> & A,
     const distribution<double> & b) const
{
    return least_squares(A, b);
}

Ridge_Regressor::
Ridge_Regressor(double lambda)
    : lambda(lambda)
{
}

Ridge_Regressor::
~Ridge_Regressor()
{
}

distribution<float>
Ridge_Regressor::
calc(const boost::multi_array<float, 2> & A,
     const distribution<float> & b) const
{
    return ridge_regression(A, b, lambda);
}

distribution<double>
Ridge_Regressor::
calc(const boost::multi_array<double, 2> & A,
     const distribution<double> & b) const
{
    return ridge_regression(A, b, lambda);
}

const Regressor & default_regressor()
{
    static const Least_Squares_Regressor result;
    return result;
}


/*****************************************************************************/
/* IRLS                                                                      */
/*****************************************************************************/

template<class Float>
distribution<Float>
perform_irls_impl(const distribution<Float> & correct,
                  const boost::multi_array<Float, 2> & outputs,
                  const distribution<Float> & w,
                  Link_Function link_function,
                  bool ridge_regression)
{
    int nx = correct.size();
    int nv = outputs.shape()[0];

    if (outputs.shape()[1] != nx)
        throw Exception("wrong shape for outputs");

    bool verify = false;
    //verify = true;

    Timer timer;

    auto doneTimer = [&] (const std::string & what)
        {
            cerr << "  finished " << what << ": " << timer.elapsed() << endl;
            timer.restart();
        };

    distribution<Float> svalues1
        (std::min(outputs.shape()[0], outputs.shape()[1]) + 1);

    if (verify) {
        boost::multi_array<Float, 2> outputs2 = outputs;

        svalues1.push_back(0.0);

        int result = LAPack::gesdd("N",
                                   outputs2.shape()[1],
                                   outputs2.shape()[0],
                                   outputs2.data(), outputs2.shape()[1], 
                                   &svalues1[0], 0, 1, 0, 1);
    
        if (result != 0)
            throw Exception("error in SVD");

        cerr << "svalues1 = " << svalues1 << endl;
        doneTimer("verify");
    }
    
    boost::multi_array<Float, 2> outputs3 = outputs;

    /* Factorize the matrix with partial pivoting.  This allows us to find the
       largest number of linearly independent columns possible. */
    
    distribution<Float> tau(nv, 0.0);
    distribution<int> permutations(nv, 0);
    
    int res = LAPack::geqp3(outputs3.shape()[1],
                            outputs3.shape()[0],
                            outputs3.data(), outputs3.shape()[1],
                            &permutations[0],
                            &tau[0]);
    
    doneTimer("geqp3");

    if (res != 0)
        throw Exception(format("geqp3: error in parameter %d", -res));

    // Convert back to c indexes
    permutations -= 1;

    //cerr << "permutations = " << permutations << endl;
    //cerr << "tau = " << tau << endl;

    distribution<Float> diag(nv);
    for (unsigned i = 0;  i < nv;  ++i)
        diag[i] = outputs3[i][i];

    //cerr << "diag = " << diag << endl;
    
    int nkeep = nv;
    while (nkeep > 0 && fabs(diag[nkeep - 1]) < 0.01) --nkeep;
    
    //cerr << "keeping " << nkeep << " of " << nv << endl;
    
    vector<int> new_loc(nv, -1);
    for (unsigned i = 0;  i < nkeep;  ++i)
        new_loc[permutations[i]] = i;

    boost::multi_array<Float, 2> outputs_reduced(boost::extents[nkeep][nx]);
    for (unsigned i = 0;  i < nx;  ++i)
        for (unsigned j = 0;  j < nv;  ++j)
            if (new_loc[j] != -1)
                outputs_reduced[new_loc[j]][i] = outputs[j][i];
    
    double svreduced = svalues1[nkeep - 1];

    //cerr << "svreduced = " << svreduced << endl;

    doneTimer("reduced");

    if (verify && svreduced < 0.001)
        throw Exception("not all linearly dependent columns were removed");

#if 0
    cerr << "w.size() = " << w.size() << endl;
    cerr << "correct.size() = " << correct.size() << endl;
    cerr << "w.total() = " << w.total() << endl;

    cerr << "outputs_reduced: " << outputs_reduced.shape()[0] << "x"
         << outputs_reduced.shape()[1] << endl;
#endif









    distribution<Float> trained;

    if (link_function == LINEAR && (w.min() == w.max())) {
        if (ridge_regression) {
            Ridge_Regressor regressor(1e-5);
            trained = regressor.calc(transpose(outputs_reduced), correct);
        }
        else {
            Least_Squares_Regressor regressor;
            trained = regressor.calc(transpose(outputs_reduced), correct);
        }
    }
    else {
        if (ridge_regression) {
            Ridge_Regressor regressor(1e-5);
            trained
                = run_irls(correct, outputs_reduced, w, link_function, regressor);
        }
        else {
            Least_Squares_Regressor regressor;
            trained
                = run_irls(correct, outputs_reduced, w, link_function, regressor);
        }
    }

    doneTimer("training");

    distribution<Float> parameters(nv);
    for (unsigned v = 0;  v < nv;  ++v)
        if (new_loc[v] != -1)
            parameters[v] = trained[new_loc[v]];
    

    if (abs(parameters).max() > 1000.0) {

        distribution<Float> svalues_reduced
            (std::min(outputs_reduced.shape()[0],
                      outputs_reduced.shape()[1]));
        
#if 0
        filter_ostream out(format("good.model.%d.txt.gz", model));

        out << "model " << model << endl;
        out << "trained " << trained << endl;
        out << "svalues_reduced " << svalues_reduced << endl;
        out << "parameters " << parameters << endl;
        out << "permuations " << permutations << endl;
        out << "svalues1 " << svalues1 << endl;
        out << "correct " << correct << endl;
        out << "outputs_reduced: " << outputs_reduced.shape()[0] << "x"
            << outputs_reduced.shape()[1] << endl;
        
        for (unsigned i = 0;  i < nx;  ++i) {
            out << "example " << i << ": ";
            for (unsigned j = 0;  j < nkeep;  ++j)
                out << " " << outputs_reduced[j][i];
            out << endl;
        }
#endif

        int result = LAPack::gesdd("N", outputs_reduced.shape()[1],
                                   outputs_reduced.shape()[0],
                                   outputs_reduced.data(),
                                   outputs_reduced.shape()[1], 
                                   &svalues_reduced[0], 0, 1, 0, 1);

        doneTimer("gesdd");

        //cerr << "model = " << model << endl;
        cerr << "trained = " << trained << endl;
        cerr << "svalues_reduced = " << svalues_reduced << endl;

        cerr << "parameters.two_norm() = " << parameters.two_norm()
             << endl;

        if (result != 0)
            throw Exception("gesdd returned error");
        
        if (svalues_reduced.back() <= 0.001) {
            throw Exception("didn't remove all linearly dependent");
        }

        // We reject this later
        //if (abs(parameters).max() > 1000.0) {
        //    throw Exception("IRLS returned inplausibly high weights");
        //}
        
    }

    //cerr << "irls returned parameters " << parameters << endl;

    return parameters;
}

distribution<float>
perform_irls(const distribution<float> & correct,
             const boost::multi_array<float, 2> & outputs,
             const distribution<float> & w,
             Link_Function link_function,
             bool ridge_regression)
{
    return perform_irls_impl(correct, outputs, w, link_function,
                             ridge_regression);
}

distribution<double>
perform_irls(const distribution<double> & correct,
             const boost::multi_array<double, 2> & outputs,
             const distribution<double> & w,
             Link_Function link_function,
             bool ridge_regression)
{
    return perform_irls_impl(correct, outputs, w, link_function,
                             ridge_regression);
}

distribution<double>
irls_logit(const distribution<double> & correct,
           const boost::multi_array<double, 2> & outputs,
           const distribution<double> & w,
           const Regressor & regressor)
{
    return irls(correct, outputs, w, Logit_Link<double>(),
                Binomial_Dist<double>(), regressor);
}

distribution<double>
irls_log(const distribution<double> & correct,
         const boost::multi_array<double, 2> & outputs,
         const distribution<double> & w,
         const Regressor & regressor)
{
    return irls(correct, outputs, w, Logarithm_Link<double>(),
                Binomial_Dist<double>(), regressor);
}

distribution<double>
irls_linear(const distribution<double> & correct,
            const boost::multi_array<double, 2> & outputs,
            const distribution<double> & w,
            const Regressor & regressor)
{
    return irls(correct, outputs, w, Linear_Link<double>(),
                Normal_Dist<double>(), regressor);
}

distribution<double>
irls_probit(const distribution<double> & correct,
            const boost::multi_array<double, 2> & outputs,
            const distribution<double> & w,
            const Regressor & regressor)
{
    return irls(correct, outputs, w, Probit_Link<double>(),
                Binomial_Dist<double>(), regressor);
}

distribution<double>
irls_complog(const distribution<double> & correct,
             const boost::multi_array<double, 2> & outputs,
             const distribution<double> & w,
             const Regressor & regressor)
{
    return irls(correct, outputs, w, Comp_Log_Log_Link<double>(),
                Binomial_Dist<double>(), regressor);
}

distribution<double>
run_irls(const distribution<double> & correct,
         const boost::multi_array<double, 2> & outputs,
         const distribution<double> & w, Link_Function func,
         const Regressor & regressor)
{
    switch (func) {

    case LOGIT:
        return irls_logit(correct, outputs, w, regressor);

    case LOG:
        return irls_log(correct, outputs, w, regressor);

    case LINEAR:
        return irls_linear(correct, outputs, w, regressor);

    case PROBIT:
        return irls_probit(correct, outputs, w, regressor);

    case COMP_LOG_LOG:
        return irls_complog(correct, outputs, w, regressor);

    default:
        throw Exception(format("run_irls(): function %d "
                               "not implemented", func));
    }
}

distribution<float>
irls_logit(const distribution<float> & correct,
           const boost::multi_array<float, 2> & outputs,
           const distribution<float> & w,
           const Regressor & regressor)
{
    return irls(correct, outputs, w, Logit_Link<float>(),
                Binomial_Dist<float>(), regressor);
}

distribution<float>
irls_log(const distribution<float> & correct,
         const boost::multi_array<float, 2> & outputs,
         const distribution<float> & w,
         const Regressor & regressor)
{
    return irls(correct, outputs, w, Logarithm_Link<float>(),
                Binomial_Dist<float>(), regressor);
}

distribution<float>
irls_linear(const distribution<float> & correct,
            const boost::multi_array<float, 2> & outputs,
            const distribution<float> & w,
            const Regressor & regressor)
{
    return irls(correct, outputs, w, Linear_Link<float>(),
                Normal_Dist<float>(), regressor);
}

distribution<float>
irls_probit(const distribution<float> & correct,
            const boost::multi_array<float, 2> & outputs,
            const distribution<float> & w,
            const Regressor & regressor)
{
    return irls(correct, outputs, w, Probit_Link<float>(),
                Binomial_Dist<float>(), regressor);
}

distribution<float>
irls_complog(const distribution<float> & correct,
             const boost::multi_array<float, 2> & outputs,
             const distribution<float> & w,
             const Regressor & regressor)
{
    return irls(correct, outputs, w, Comp_Log_Log_Link<float>(),
                Binomial_Dist<float>(), regressor);
}

distribution<float>
run_irls(const distribution<float> & correct,
         const boost::multi_array<float, 2> & outputs,
         const distribution<float> & w, Link_Function func,
         const Regressor & regressor)
{
    switch (func) {

    case LOGIT:
        return irls_logit(correct, outputs, w, regressor);

    case LOG:
        return irls_log(correct, outputs, w, regressor);

    case LINEAR:
        return irls_linear(correct, outputs, w, regressor);

    case PROBIT:
        return irls_probit(correct, outputs, w, regressor);

    case COMP_LOG_LOG:
        return irls_complog(correct, outputs, w, regressor);

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

