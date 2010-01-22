/* tsne.cc
   Jeremy Barnes, 15 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Implementation of the t-SNE algorithm.
*/

#include "tsne.h"
#include "stats/distribution.h"
#include "stats/distribution_ops.h"
#include "stats/distribution_simd.h"
#include "algebra/matrix_ops.h"
#include "arch/simd_vector.h"
#include <boost/tuple/tuple.hpp>
#include "algebra/lapack.h"
#include <cmath>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include "boosting/worker_task.h"
#include <boost/timer.hpp>
#include "arch/timers.h"
#include "arch/sse2.h"
#include "arch/cache.h"
#include "utils/guard.h"
#include <boost/bind.hpp>


using namespace std;

namespace ML {

/** Compute the perplexity and the P for a given value of beta.
    TODO: pre-compute the exp(D) values to remove exp from this loop*/
template<typename Float>
std::pair<double, distribution<Float> >
perplexity_and_prob(const distribution<Float> & D, double beta = 1.0,
                    int i = -1)
{
    distribution<double> P(D.size());
    SIMD::vec_exp(&D[0], -beta, &P[0], D.size());
    if (i != -1) P[i] = 0;
    double tot = P.total();

    if (!isfinite(tot) || tot == 0) {
        cerr << "beta = " << beta << endl;
        cerr << "D = " << D << endl;
        throw Exception("non-finite total for perplexity");
    }

    double H = log(tot) + beta * D.dotprod(P) / tot;
    P *= 1.0 / tot;

    if (!isfinite(P.total())) {
        cerr << "beta = " << beta << endl;
        cerr << "D = " << D << endl;
        cerr << "tot = " << tot << endl;
        throw Exception("non-finite total for perplexity");
    }


    return make_pair(H, P);
}

std::pair<double, distribution<float> >
perplexity_and_prob(const distribution<float> & D, double beta,
                    int i)
{
    return perplexity_and_prob<float>(D, beta, i);
}

std::pair<double, distribution<double> >
perplexity_and_prob(const distribution<double> & D, double beta,
                    int i)
{
    return perplexity_and_prob<double>(D, beta, i);
}

template<typename Float>
struct V2D_Job_2 {
    const boost::multi_array<Float, 2> & X;
    boost::multi_array<Float, 2> & D;
    const Float * sum_X;
    int i0, i1;
    
    V2D_Job_2(const boost::multi_array<Float, 2> & X,
              boost::multi_array<Float, 2> & D,
              const Float * sum_X,
              int i0, int i1)
        : X(X), D(D), sum_X(sum_X), i0(i0), i1(i1)
    {
    }

    void operator () ()
    {
        for (unsigned i = i0;  i < i1;  ++i) {
            D[i][i] = 0.0f;

            for (unsigned j = 0;  j < i;  ++j) {
                float XXT = (X[i][0] * X[j][0]) + (X[i][1]) * (X[j][1]);
                Float val = sum_X[i] + sum_X[j] - 2.0f * XXT;
                D[i][j] = val;
            }
        }
    }
};

template<typename Float>
double
vectors_to_distances(const boost::multi_array<Float, 2> & X,
                     boost::multi_array<Float, 2> & D,
                     bool fill_upper = true)
{
    // again, ||y_i - y_j||^2 
    //     = sum_d ( y_id - y_jd )^2
    //     = sum_d ( y_id^2 + y_jd^2 - 2 y_id y_jd)
    //     = sum_d ( y_id^2) + sum_d(y_jd^2) - 2 sum_d(y_id y_jd)
    //     = ||y_i||^2 + ||y_j||^2 - 2 sum_d(y_id y_jd)
    
    int n = X.shape()[0];

    if (D.shape()[0] != n || D.shape()[1] != n)
        throw Exception("D matrix should be square with (n x n) shape");
    
    int d = X.shape()[1];

    distribution<Float> sum_X(n);

    if (d < 16) {
        for (unsigned i = 0;  i < n;  ++i) {
            double total = 0.0;  // accum in double precision for accuracy
            for (unsigned j = 0;  j < d;  ++j)
                total += X[i][j] * X[i][j];
            sum_X[i] = total;
        }
    }
    else {
        for (unsigned i = 0;  i < n;  ++i)
            sum_X[i] = SIMD::vec_dotprod_dp(&X[i][0], &X[i][0], d);
    }
    
    double total = 0.0;

    Worker_Task & worker = Worker_Task::instance(num_threads() - 1);

    if (d == 2) {

#if 1
        int group;
        {
            int parent = -1;  // no parent group
            group = worker.get_group(NO_JOB, "", parent);
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         boost::ref(worker),
                                         group));
        
            int chunk_size = 50;

            for (unsigned i = 0;  i < n;  i += chunk_size) {
                int i0 = i;
                int i1 = min(i0 + chunk_size, n);

                worker.add(V2D_Job_2<Float>(X, D, &sum_X[0], i0, i1),
                           "", group);
            }
        }

        worker.run_until_finished(group);

#else
        for (unsigned i = 0;  i < n;  ++i) {
            D[i][i] = 0.0f;
            for (unsigned j = 0;  j < i;  ++j) {
                float XXT = (X[i][0] * X[j][0]) + (X[i][1]) * (X[j][1]);
                Float val = sum_X[i] + sum_X[j] - 2.0f * XXT;
                D[i][j] = val;
            }
        }
#endif
    }
    else if (d < 8) {
        for (unsigned i = 0;  i < n;  ++i) {
            D[i][i] = 0.0f;
            for (unsigned j = 0;  j < i;  ++j) {
                double XXT = 0.0;  // accum in double precision for accuracy
                for (unsigned k = 0;  k < d;  ++k)
                    XXT += X[i][k] * X[j][k];

                Float val = sum_X[i] + sum_X[j] - 2.0f * XXT;
                D[i][j] = val;
            }
        }
    }
    else {
        for (unsigned i = 0;  i < n;  ++i) {
            D[i][i] = 0.0f;
            for (unsigned j = 0;  j < i;  ++j) {
                float XXT = SIMD::vec_dotprod_dp(&X[i][0], &X[j][0], d);
                Float val = sum_X[i] + sum_X[j] - 2.0f * XXT;
                D[i][j] = val;
            }
        }
    }

    if (fill_upper)
        copy_lower_to_upper(D);
    
    return total;
}

double
vectors_to_distances(const boost::multi_array<float, 2> & X,
                     boost::multi_array<float, 2> & D,
                     bool fill_upper)
{
    return vectors_to_distances<float>(X, D, fill_upper);
}

double
vectors_to_distances(const boost::multi_array<double, 2> & X,
                     boost::multi_array<double, 2> & D,
                     bool fill_upper)
{
    return vectors_to_distances<double>(X, D, fill_upper);
}

/** Given a matrix of distances, normalize them */

/** Calculate the beta for a single point.
    
    \param Di     The i-th row of the D matrix, for which we want to calculate
                  the probabilities.
    \param i      Which row number it is.

    \returns      The i-th row of the P matrix, which has the distances in D
                  converted to probabilities with the given perplexity.
 */
std::pair<distribution<float>, double>
binary_search_perplexity(const distribution<float> & Di,
                         double required_perplexity,
                         int i,
                         double tolerance = 1e-5)
{
    double betamin = -INFINITY, betamax = INFINITY;
    double beta = 1.0;

    distribution<float> P;
    double log_perplexity;
    double log_required_perplexity = log(required_perplexity);

    boost::tie(log_perplexity, P) = perplexity_and_prob(Di, beta, i);

    bool verbose = false;

    if (verbose)
        cerr << "iter currperp targperp     diff toleranc   betamin     beta  betamax" << endl;
    
    for (unsigned iter = 0;  iter != 50;  ++iter) {
        if (verbose) 
            cerr << format("%4d %8.4f %8.4f %8.4f %8.4f  %8.4f %8.4f %8.4f\n",
                           iter,
                           log_perplexity, log_required_perplexity,
                           fabs(log_perplexity - log_required_perplexity),
                           tolerance,
                           betamin, beta, betamax);
        
        if (fabs(log_perplexity - log_required_perplexity) < tolerance)
            break;

        if (log_perplexity > log_required_perplexity) {
            betamin = beta;
            if (!isfinite(betamax))
                beta *= 2;
            else beta = (beta + betamax) * 0.5;
        }
        else {
            betamax = beta;
            if (!isfinite(betamin))
                beta /= 2;
            else beta = (beta + betamin) * 0.5;
        }
        
        boost::tie(log_perplexity, P) = perplexity_and_prob(Di, beta, i);
    }

    return make_pair(P, beta);
}

/* Given a matrix of distances, convert to probabilities */
boost::multi_array<float, 2>
distances_to_probabilities(boost::multi_array<float, 2> & D,
                           double tolerance,
                           double perplexity)
{
    int n = D.shape()[0];
    if (D.shape()[1] != n)
        throw Exception("D is not square");

    boost::multi_array<float, 2> P(boost::extents[n][n]);

    distribution<float> beta(n, 1.0);

    for (unsigned i = 0;  i < n;  ++i) {
        //cerr << "i = " << i << endl;
        if (i % 250 == 0)
            cerr << "P-values for point " << i << " of " << n << endl;
        
        distribution<float> D_row(&D[i][0], &D[i][0] + n);
        distribution<float> P_row;
        boost::tie(P_row, beta[i])
            = binary_search_perplexity(D_row, perplexity, i, tolerance);

        if (P_row.size() != n)
            throw Exception("P_row has the wrong size");
        if (P_row[i] != 0.0) {
            cerr << "i = " << i << endl;
            //cerr << "D_row = " << D_row << endl;
            //cerr << "P_row = " << P_row << endl;
            cerr << "P_row.total() = " << P_row.total() << endl;
            cerr << "P_row[i] = " << P_row[i] << endl;
            throw Exception("P_row diagonal entry was not zero");
        }
        
        std::copy(P_row.begin(), P_row.end(), &P[i][0]);
    }

    cerr << "mean sigma is " << sqrt(1.0 / beta).mean() << endl;

    return P;
}

boost::multi_array<float, 2>
pca(boost::multi_array<float, 2> & coords, int num_dims)
{
    // TODO: normalize the input coordinates (especially if it seems to be
    // ill conditioned)

    int nx = coords.shape()[0];
    int nd = coords.shape()[1];

    int nvalues = std::min(nd, nx);

    int ndr = std::min(nvalues, num_dims);

    if (ndr < num_dims)
        throw Exception("svd_reduction: num_dims not low enough");
        
    distribution<float> svalues(nvalues);
    boost::multi_array<float, 2> lvectorsT(boost::extents[nvalues][nd]);
    boost::multi_array<float, 2> rvectors(boost::extents[nx][nvalues]);

    int res = LAPack::gesdd("S", nd, nx,
                            coords.data(), nd,
                            &svalues[0],
                            &lvectorsT[0][0], nd,
                            &rvectors[0][0], nvalues);
    
    // If some vectors are singular, ignore them
    // TODO: do...
        
    if (res != 0)
        throw Exception("gesdd returned non-zero");
        
    boost::multi_array<float, 2> result(boost::extents[nx][ndr]);
    for (unsigned i = 0;  i < nx;  ++i)
        std::copy(&rvectors[i][0], &rvectors[i][0] + ndr, &result[i][0]);

    return result;
}

double
kl(const float * p, const float * q, size_t n)
{
    double result = 0.0;
    for (unsigned i = 0;  i < n;  ++i)
        result += p[i] * logf(p[i] / q[i]);
    return result;
}

double calc_D_row(float * Di, int n)
{
    unsigned i = 0;

    double total = 0.0;

    if (false) ;
    else if (n >= 8) {
        using namespace SIMD;

        v2df rr = vec_splat(0.0);
        
        v4sf one = vec_splat(1.0f);

        for (; i + 16 <= n;  i += 16) {
            v4sf xxxx0 = __builtin_ia32_loadups(Di + i + 0);
            v4sf xxxx1 = __builtin_ia32_loadups(Di + i + 4);
            xxxx0      = xxxx0 + one;
            xxxx1      = xxxx1 + one;
            xxxx0      = one / xxxx0;
            v4sf xxxx2 = __builtin_ia32_loadups(Di + i + 8);
            xxxx1      = one / xxxx1;
            __builtin_ia32_storeups(Di + i + 0, xxxx0);
            xxxx2      = xxxx2 + one;
            v2df xx0a, xx0b;  vec_f2d(xxxx0, xx0a, xx0b);
            __builtin_ia32_storeups(Di + i + 4, xxxx1);
            xx0a       = xx0a + xx0b;
            rr         = rr + xx0a;
            v4sf xxxx3 = __builtin_ia32_loadups(Di + i + 12);
            v2df xx1a, xx1b;  vec_f2d(xxxx1, xx1a, xx1b);
            xxxx2      = one / xxxx2;
            xx1a       = xx1a + xx1b;
            __builtin_ia32_storeups(Di + i + 8, xxxx2);
            rr         = rr + xx1a;
            v2df xx2a, xx2b;  vec_f2d(xxxx2, xx2a, xx2b);
            xxxx3      = xxxx3 + one;
            xx2a       = xx2a + xx2b;
            xxxx3      = one / xxxx3;
            rr         = rr + xx2a;
            v2df xx3a, xx3b;  vec_f2d(xxxx3, xx3a, xx3b);
            __builtin_ia32_storeups(Di + i + 12, xxxx3);
            xx3a       = xx3a + xx3b;
            rr         = rr + xx3a;
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf xxxx0 = __builtin_ia32_loadups(Di + i + 0);
            xxxx0      = xxxx0 + one;
            xxxx0      = one / xxxx0;
            __builtin_ia32_storeups(Di + i + 0, xxxx0);

            v2df xx0a, xx0b;
            vec_f2d(xxxx0, xx0a, xx0b);

            rr      = rr + xx0a;
            rr      = rr + xx0b;
        }

        double results[2];
        *(v2df *)results = rr;

        total = (results[0] + results[1]);
    }
    
    for (;  i < n;  ++i) {
        Di[i] = 1.0f / (1.0f + Di[i]);
        total += Di[i];
    }

    return total;
}

void calc_Q_row(float * Qi, float qfactor, const float * Di, float minval,
                int n)
{
    unsigned i = 0;

    if (false) ;
    else if (n >= 8) {
        using namespace SIMD;

        v4sf mmmm = vec_splat(minval);
        v4sf ffff = vec_splat(qfactor);

        for (; i + 16 <= n;  i += 16) {
            v4sf qqqq0 = __builtin_ia32_loadups(Di + i + 0);
            qqqq0      = qqqq0 * ffff;
            qqqq0      = __builtin_ia32_maxps(qqqq0, mmmm);
            __builtin_ia32_storeups(Qi + i + 0, qqqq0);

            v4sf qqqq1 = __builtin_ia32_loadups(Di + i + 4);
            qqqq1      = qqqq1 * ffff;
            qqqq1      = __builtin_ia32_maxps(qqqq1, mmmm);
            __builtin_ia32_storeups(Qi + i + 4, qqqq1);

            v4sf qqqq2 = __builtin_ia32_loadups(Di + i + 8);
            qqqq2      = qqqq2 * ffff;
            qqqq2      = __builtin_ia32_maxps(qqqq2, mmmm);
            __builtin_ia32_storeups(Qi + i + 8, qqqq2);

            v4sf qqqq3 = __builtin_ia32_loadups(Di + i + 12);
            qqqq3      = qqqq3 * ffff;
            qqqq3      = __builtin_ia32_maxps(qqqq3, mmmm);
            __builtin_ia32_storeups(Qi + i + 12, qqqq3);
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf qqqq0 = __builtin_ia32_loadups(Di + i + 0);
            qqqq0      = qqqq0 * ffff;
            qqqq0      = __builtin_ia32_maxps(qqqq0, mmmm);
            __builtin_ia32_storeups(Qi + i + 0, qqqq0);
        }
    }

    for (; i < n;  ++i)
        Qi[i] = std::max(1e-12f, Di[i] * qfactor);
}

JML_ALWAYS_INLINE
void calc_PmQxD_row(float * PmQxDi, const float * Pi, const float * Qi,
                    const float * Di, int n)
{
    unsigned i = 0;

    if (false) ;
    else if (n >= 8) {
        using namespace SIMD;

        for (; i + 4 <= n;  i += 4) {
            v4sf pppp0 = __builtin_ia32_loadups(Pi + i + 0);
            v4sf qqqq0 = __builtin_ia32_loadups(Qi + i + 0);
            v4sf dddd0 = __builtin_ia32_loadups(Di + i + 0);
            v4sf rrrr0 = (pppp0 - qqqq0) * dddd0;
            __builtin_ia32_storeups(PmQxDi + i + 0, rrrr0);
        }
    }

    for (; i < n;  ++i)
        PmQxDi[i] = (Pi[i] - Qi[i]) * Di[i];
}

namespace {

double t_v2d = 0.0, t_D = 0.0, t_dY = 0.0, t_update = 0.0;
double t_recenter = 0.0, t_cost = 0.0, t_PmQxD = 0.0, t_clu = 0.0;
double t_Q_stiffness = 0.0;
struct AtEnd {
    ~AtEnd()
    {
        cerr << "tsne core profile:" << endl;
        cerr << "  v2d:        " << t_v2d << endl;
        cerr << "  Q_stiffness:" << t_Q_stiffness << endl;
        cerr << "    D         " << t_D << endl;
        cerr << "    (P-Q)D    " << t_PmQxD << endl;
        cerr << "    clu       " << t_clu << endl;
        cerr << "  dY:         " << t_dY << endl;
        cerr << "  update:     " << t_update << endl;
        cerr << "  recenter:   " << t_recenter << endl;
        cerr << "  cost:       " << t_cost << endl;
    }
} atend;

} // file scope


void tsne_calc_Q_stiffness(boost::multi_array<float, 2> & Q,
                           boost::multi_array<float, 2> & PmQxD,
                           boost::multi_array<float, 2> & D,
                           const boost::multi_array<float, 2> & P,
                           float min_prob)
{
    boost::timer t;

    int n = Q.shape()[0];
    if (Q.shape()[1] != n)
        throw Exception("Q has wrong shape");

    if (PmQxD.shape()[0] != n || PmQxD.shape()[1] != n)
        throw Exception("PmQxD has wrong shape");

    if (D.shape()[0] != n || D.shape()[1] != n)
        throw Exception("D has wrong shape");

    if (P.shape()[0] != n || P.shape()[1] != n)
        throw Exception("P has wrong shape");

    double d_total_offdiag = 0.0;

    for (unsigned i = 0;  i < n;  ++i) {
        d_total_offdiag += 2.0 * calc_D_row(&D[i][0], i);
        D[i][i] = 0.0f;
    }
    
    t_D += t.elapsed();  t.restart();
    
    // Q matrix: q_{i,j} = d_{ij} / sum_{k != l} d_{kl}
    float qfactor = 1.0 / d_total_offdiag;
    for (unsigned i = 0;  i < n;  ++i) {
        Q[i][i] = min_prob;
        calc_Q_row(&Q[i][0], qfactor, &D[i][0], min_prob, i);
        calc_PmQxD_row(&PmQxD[i][0], &P[i][0], &Q[i][0], &D[i][0], i);
    }
    
    t_PmQxD += t.elapsed();  t.restart();
    
    copy_lower_to_upper(PmQxD);
    
    t_clu += t.elapsed();  t.restart();
}

inline void
calc_dY_rows_2d(boost::multi_array<float, 2> & dY,
                const boost::multi_array<float, 2> & PmQxD,
                const boost::multi_array<float, 2> & Y,
                int i, int n)
{
#if 0
    v4sf totals1 = vec_splat(0.0f), totals2 = totals1;
    v4sf four = vec_splat(4.0f);

    for (unsigned j = 0;  j + 4 < n;  j += 4) {
        //v4sf ffff = { PmQxD[i + 0][j], PmQxD[i + 1][j],
        //              PmQxD[i + 2][j], PmQxD[i + 3][j] };
        // TODO: expand inplace

        v4sf ffff1 = { PmQxD[i + 0][j], PmQxD[i + 0][j],
                       PmQxD[i + 1][j], PmQxD[i + 1][j] };
        v4sf ffff2 = { PmQxD[i + 2][j], PmQxD[i + 2][j],
                       PmQxD[i + 3][j], PmQxD[i + 3][j] };

        ffff1 = ffff1 * four;
        ffff2 = ffff2 * four;
        
        // Load the elements
    }   

        
#else
    enum { b = 4 };

    float totals[b][2];
    for (unsigned ii = 0;  ii < b;  ++ii)
        totals[ii][0] = totals[ii][1] = 0.0f;
            
    for (unsigned j = 0;  j < n;  ++j) {
        float Yj0 = Y[j][0];
        float Yj1 = Y[j][1];
        
        for (unsigned ii = 0;  ii < b;  ++ii) {
            float factor = 4.0f * PmQxD[i + ii][j];
            totals[ii][0] += factor * (Y[i + ii][0] - Yj0);
            totals[ii][1] += factor * (Y[i + ii][1] - Yj1);
        }
    }
    
    for (unsigned ii = 0;  ii < b;  ++ii) {
        dY[i + ii][0] = totals[ii][0];
        dY[i + ii][1] = totals[ii][1];
    }
#endif
}

inline void
calc_dY_row_2d(float * dYi, const float * PmQxDi,
               const boost::multi_array<float, 2> & Y,
               int i,
               int n)
{
    float total0 = 0.0f, total1 = 0.0f;
    for (unsigned j = 0;  j < n;  ++j) {
        float factor = 4.0f * PmQxDi[j];
        total0 += factor * (Y[i][0] - Y[j][0]);
        total1 += factor * (Y[i][1] - Y[j][1]);
    }
    
    dYi[0] = total0;
    dYi[1] = total1;
}

void tsne_calc_gradient(boost::multi_array<float, 2> & dY,
                        const boost::multi_array<float, 2> & Y,
                        const boost::multi_array<float, 2> & PmQxD)
{
    // Gradient
    // Implements formula 5 in (Van der Maaten and Hinton, 2008)
    // dC/dy_i = 4 * sum_j ( (p_ij - q_ij)(y_i - y_j)d_ij )

    
    int n = Y.shape()[0];
    int d = Y.shape()[1];
    
    if (dY.shape()[0] != n || dY.shape()[1] != d)
        throw Exception("dY matrix has wrong shape");

    if (PmQxD.shape()[0] != n || PmQxD.shape()[1] != n)
        throw Exception("PmQxD matrix has wrong shape");

    if (d == 2) {
        unsigned i = 0;
        
        for (;  i + 4 <= n;  i += 4)
            calc_dY_rows_2d(dY, PmQxD, Y, i, n);
        
        for (; i < n;  ++i)
            calc_dY_row_2d(&dY[i][0], &PmQxD[i][0], Y, i, n);
    }
    else {
        // TODO: optimize better than this...
        std::fill(dY.data(), dY.data() + dY.num_elements(), 0.0f);
        
        for (unsigned j = 0;  j < n;  ++j) {
            for (unsigned i = 0;  i < n;  ++i) {
                if (i == j) continue;
                float factor = 4.0f * PmQxD[j][i];
                for (unsigned k = 0;  k < d;  ++k)
                    dY[i][k] += factor * (Y[i][k] - Y[j][k]);
            }
        }
    }
}

void tsne_update(boost::multi_array<float, 2> & Y,
                 boost::multi_array<float, 2> & dY,
                 boost::multi_array<float, 2> & iY,
                 boost::multi_array<float, 2> & gains,
                 bool first_iter,
                 float momentum,
                 float eta,
                 float min_gain)
{
    int n = Y.shape()[0];
    int d = Y.shape()[1];

    // Implement scheme in Jacobs, 1988.  If we go in the same direction as
    // last time, we increase the learning speed of the parameter a bit.
    // If on the other hand the direction changes, we reduce exponentially
    // the rate.
    
    for (unsigned i = 0;  !first_iter && i < n;  ++i) {
        // We use != here as we gradients in dY are the negatives of what
        // we want.
        for (unsigned j = 0;  j < d;  ++j) {
            if (dY[i][j] * iY[i][j] < 0.0f)
                gains[i][j] = gains[i][j] + 0.2f;
            else gains[i][j] = gains[i][j] * 0.8f;
            gains[i][j] = std::max(min_gain, gains[i][j]);
        }
    }

    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < d;  ++j)
            iY[i][j] = momentum * iY[i][j] - (eta * gains[i][j] * dY[i][j]);
    Y = Y + iY;
}
    
template<typename Float>
void recenter_about_origin(boost::multi_array<Float, 2> & Y)
{
    int n = Y.shape()[0];
    int d = Y.shape()[1];

    // Recenter Y values about the origin
    double Y_means[d];
    std::fill(Y_means, Y_means + d, 0.0);
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < d;  ++j)
            Y_means[j] += Y[i][j];
    
    Float n_recip = 1.0f / n;
    
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < d;  ++j)
            Y[i][j] -= Y_means[j] * n_recip;
}

boost::multi_array<float, 2>
tsne(const boost::multi_array<float, 2> & probs,
     int num_dims,
     const TSNE_Params & params)
{
    int n = probs.shape()[0];
    if (n != probs.shape()[1])
        throw Exception("probabilities were the wrong shape");

    int d = num_dims;

    boost::mt19937 rng;
    boost::normal_distribution<float> norm;

    boost::variate_generator<boost::mt19937,
                             boost::normal_distribution<float> >
        randn(rng, norm);
    
    boost::multi_array<float, 2> Y(boost::extents[n][d]);
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < d;  ++j)
            Y[i][j] = 0.0001 * randn();

#if 0 // pseudo-random, for testing (matches the Python version)
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < d;  ++j)
            Y[i][j] = (((((i * 18446744073709551557ULL) + j) * 18446744073709551557ULL) % 4099) / 1050.0) - 2.0;
#endif // pseudo-random

    //for (unsigned i = 0;  i < 10;  ++i)
    //    cerr << "Y[" << i << "] = "
    //         << distribution<float>(&Y[i][0], &Y[i][0] + d)
    //         << endl;

    boost::multi_array<float, 2> iY(boost::extents[n][d]);
    boost::multi_array<float, 2> gains(boost::extents[n][d]);
    std::fill(gains.data(), gains.data() + gains.num_elements(), 1.0f);

    // Symmetrize and probabilize P
    boost::multi_array<float, 2> P = probs + transpose(probs);

    // TODO: symmetric so only need to total the diagonal
    double sumP = 0.0;
    for (unsigned i = 0;  i < n;  ++i)
        sumP += SIMD::vec_sum_dp(&P[i][0], n);
    
    //cerr << "sumP = " << sumP << endl;

    // Factor that P should be multiplied by in all calculations
    // We boost it by 4 in early iterations to force the clusters to be
    // spread apart
    float pfactor = 4.0 / sumP;

    // TODO: do we need this?   P = Math.maximum(P, 1e-12);
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < n;  ++j)
            P[i][j] = std::max((i != j) * pfactor * P[i][j], 1e-12f);

    //for (unsigned i = 0;  i < 10;  ++i)
    //    cerr << "P[" << i << "] = "
    //         << distribution<float>(&P[i][0], &P[i][0] + n)
    //         << endl;


    //cerr << "n = " << n << endl;

    Timer timer;

    boost::multi_array<float, 2> D(boost::extents[n][n]);
    boost::multi_array<float, 2> Q(boost::extents[n][n]);
    boost::multi_array<float, 2> PmQxD(boost::extents[n][n]);
    boost::multi_array<float, 2> dY(boost::extents[n][d]);


    for (int iter = 0;  iter < params.max_iter;  ++iter) {

        boost::timer t;

        /*********************************************************************/
        // Pairwise affinities Qij
        // Implements formula 4 in (Van der Maaten and Hinton, 2008)
        // q_{ij} = d_{ij} / sum_{k,l, k != l} d_{kl}
        // where d_{ij} = 1 / (1 + ||y_i - y_j||^2)

        // TODO: these will all be symmetric; we could save lots of work by
        // using upper/lower diagonal matrices.

        vectors_to_distances(Y, D, false /* fill_upper */);

        t_v2d += t.elapsed();  t.restart();

        tsne_calc_Q_stiffness(Q, PmQxD, D, P, params.min_prob);

        t_Q_stiffness += t.elapsed();  t.restart();

        
        /*********************************************************************/
        // Gradient
        // Implements formula 5 in (Van der Maaten and Hinton, 2008)
        // dC/dy_i = 4 * sum_j ( (p_ij - q_ij)(y_i - y_j)d_ij )

        tsne_calc_gradient(dY, Y, PmQxD);

        t_dY += t.elapsed();  t.restart();


        /*********************************************************************/
        // Update

        float momentum = (iter < 20
                          ? params.initial_momentum
                          : params.final_momentum);

        tsne_update(Y, dY, iY, gains, iter == 0, momentum, params.eta,
                    params.min_gain);

        t_update += t.elapsed();  t.restart();


        /*********************************************************************/
        // Recenter about the origin

        recenter_about_origin(Y);

        t_recenter += t.elapsed();  t.restart();


        /*********************************************************************/
        // Calculate cost

        if ((iter + 1) % 20 == 0 || iter == params.max_iter - 1) {
            double cost = 0.0;
            for (unsigned i = 0;  i < n;  ++i) {
                cost += 2.0 * kl(&P[i][0], &Q[i][0], i);
            }
            cerr << "iteration " << (iter + 1)
                 << " cost = " << cost << " elapsed "
                 << timer.elapsed() << endl;
            timer.restart();
        }
        
        t_cost += t.elapsed();  t.restart();

        // Stop lying about P values if we're finished
        if (iter == 100) {
            for (unsigned i = 0;  i < n;  ++i)
                for (unsigned j = 0;  j < n;  ++j)
                    P[i][j] *= 0.25f;
        }
    }

    return Y;
}

} // namespace ML
