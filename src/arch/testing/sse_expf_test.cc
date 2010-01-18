/* sse_expf_test.cc
   Jeremy Barnes, 18 January 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Test for the SSE2 expf function.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "arch/sse2_expf.h"
#include "arch/demangle.h"
#include "utils/vector_utils.h"

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <vector>
#include <set>
#include <iostream>
#include <cmath>
#include "arch/tick_counter.h"
#include "utils/string_functions.h"
#include <sys/time.h>
#include "utils/floating_point.h"

using namespace ML;
using namespace std;
using namespace ML::SIMD;

using boost::unit_test::test_suite;

float extract_scalar(v4sf i)
{
    float vals[4];
    *((v4sf *)vals) = i;
    if (isnan(vals[0])) {
        BOOST_CHECK(isnan(vals[1]));
        BOOST_CHECK(isnan(vals[2]));
        BOOST_CHECK(isnan(vals[3]));
    }
    else {
        BOOST_CHECK_EQUAL(vals[0], vals[1]);
        BOOST_CHECK_EQUAL(vals[0], vals[2]);
        BOOST_CHECK_EQUAL(vals[0], vals[3]);
    }
    return vals[0];
}

double extract_scalar(v2df i)
{
    double vals[2];
    *((v2df *)vals) = i;
    if (isnan(vals[0])) {
        BOOST_CHECK(isnan(vals[1]));
    }
    else {
        BOOST_CHECK_EQUAL(vals[0], vals[1]);
    }
    return vals[0];
}

#define test_floor_value(input) \
{ \
    float in2 = float(input); \
    float output1 = floorf(in2); \
    float output2 = extract_scalar(sse2_floor(vec_splat(in2))); \
    if (isnan(output1)) \
        BOOST_CHECK(isnan(output2));            \
    else BOOST_CHECK_EQUAL(floorf(float(input)), output2);      \
}

static const float NaN = std::numeric_limits<float>::quiet_NaN();

BOOST_AUTO_TEST_CASE( floor_test )
{
    test_floor_value(0.0);
    test_floor_value(-0.0);
    test_floor_value(1.0);
    test_floor_value(2.0);
    test_floor_value(2.5);
    test_floor_value(-0.49);
    test_floor_value(-0.5);
    test_floor_value(-0.51);
    test_floor_value(-0.99);
    test_floor_value(-1.0);
    test_floor_value(-1.01);
    test_floor_value(2.5);

#if 0
    test_floor_value(NaN);
    test_floor_value(-NaN);
    test_floor_value(INFINITY);
    test_floor_value(-INFINITY);
#endif
}

#define test_trunc_value(input) \
{ \
    float in2 = float(input); \
    float output1 = truncf(in2); \
    float output2 = extract_scalar(sse2_trunc(vec_splat(in2))); \
    if (isnan(output1)) \
        BOOST_CHECK(isnan(output2));            \
    else BOOST_CHECK_EQUAL(truncf(float(input)), output2);      \
}

BOOST_AUTO_TEST_CASE( trunc_test )
{
    test_trunc_value(0.0);
    test_trunc_value(-0.0);
    test_trunc_value(1.0);
    test_trunc_value(2.0);
    test_trunc_value(2.5);
    test_trunc_value(-0.49);
    test_trunc_value(-0.5);
    test_trunc_value(-0.51);
    test_trunc_value(-0.99);
    test_trunc_value(-1.0);
    test_trunc_value(-1.01);
    test_trunc_value(2.5);

#if 0
    test_trunc_value(NaN);
    test_trunc_value(-NaN);
    test_trunc_value(INFINITY);
    test_trunc_value(-INFINITY);
#endif
}

#define test_expf_value(input)                                          \
    {                                                                   \
        float in2 = float(input);                                       \
        float output1 = expf(in2);                                      \
        float output2 = extract_scalar(sse2_expf(vec_splat(in2)));      \
        if (isnan(output1)) {                                           \
            if (!isnan(output2)) {                                      \
                cerr << "input = " << in2 << " output1 = " << output1 << " output2 = " << output2 \
                 << endl;                                               \
            }                                                           \
            BOOST_CHECK(isnan(output2));                                \
        }                                                               \
        else if (output1 != output2) {                                  \
            int i1 = reinterpret_as_int(output1);                       \
            int i2 = reinterpret_as_int(output2);                       \
            if (abs(i1 - i2) > 1) {                                     \
                cerr << format("%12.8f: %14.9f != %14.9f: %08x !- %08x (%4d ulps)\n", \
                               in2, output1, output2, i1, i2, (i1 - i2)); \
                BOOST_CHECK_EQUAL(expf(float(input)), output2);         \
            }                                                           \
        }                                                               \
    }

BOOST_AUTO_TEST_CASE( test_expf )
{
    test_expf_value(0.0);
    test_expf_value(1.0);
    test_expf_value(2.0);
    test_expf_value(3.0);
    test_expf_value(-0.0);
    test_expf_value(-1.0);
    test_expf_value(-2.0);
    test_expf_value(-3.0);

    test_expf_value(-10.0);
    test_expf_value(-20.0);
    test_expf_value(-30.0);
    test_expf_value(-50.0);
    test_expf_value(-100.0);
    test_expf_value(-1000.0);

    test_expf_value(10.0);
    test_expf_value(20.0);
    test_expf_value(30.0);
    test_expf_value(50.0);
    test_expf_value(100.0);
    test_expf_value(1000.0);

    test_expf_value(NaN);
    test_expf_value(-NaN);
    test_expf_value(INFINITY);
    test_expf_value(-INFINITY);

    int nvals = 65536;
    for (int i = 0;  i < nvals;  ++i) {
        float f = 105.0 * (2 * i - nvals) / (1.0 * nvals);
        test_expf_value(f);
    }
}

#define test_exp_value(input)                                          \
    {                                                                   \
        double in2 = double(input);                                       \
        double output1 = exp(in2);                                      \
        double output2 = extract_scalar(sse2_exp(vec_splat(in2)));      \
        if (isnan(output1)) {                                           \
            if (!isnan(output2)) {                                      \
                cerr << "input = " << in2 << " output1 = " << output1 << " output2 = " << output2 \
                 << endl;                                               \
            }                                                           \
            BOOST_CHECK(isnan(output2));                                \
        }                                                               \
        else if (output1 != output2) {                                  \
            int i1 = reinterpret_as_int(output1);                       \
            int i2 = reinterpret_as_int(output2);                       \
            if (abs(i1 - i2) > 1) {                                     \
                cerr << format("%12.8f: %14.9f != %14.9f: %08x !- %08x (%4d ulps)\n", \
                               in2, output1, output2, i1, i2, (i1 - i2)); \
                BOOST_CHECK_EQUAL(exp(double(input)), output2);         \
            }                                                           \
        }                                                               \
    }

BOOST_AUTO_TEST_CASE( test_exp )
{
    test_exp_value(0.0);
    test_exp_value(1.0);
    test_exp_value(2.0);
    test_exp_value(3.0);
    test_exp_value(-0.0);
    test_exp_value(-1.0);
    test_exp_value(-2.0);
    test_exp_value(-3.0);

    test_exp_value(-10.0);
    test_exp_value(-20.0);
    test_exp_value(-30.0);
    test_exp_value(-50.0);
    test_exp_value(-100.0);
    test_exp_value(-1000.0);

    test_exp_value(10.0);
    test_exp_value(20.0);
    test_exp_value(30.0);
    test_exp_value(50.0);
    test_exp_value(100.0);
    test_exp_value(1000.0);

    test_exp_value(NaN);
    test_exp_value(-NaN);
    test_exp_value(INFINITY);
    test_exp_value(-INFINITY);

#if 0
    int nvals = 65536;
    for (int i = 0;  i < nvals;  ++i) {
        double f = 105.0 * (2 * i - nvals) / (1.0 * nvals);
        test_exp_value(f);
    }
#endif
}

namespace {

// TODO: use clock_gettime
double elapsed_since(const timeval & tv_start)
{
    struct timeval tv_end;
    gettimeofday(&tv_end, 0);

    double start_sec = tv_start.tv_sec + (tv_start.tv_usec / 1000000.0);
    double end_sec = tv_end.tv_sec + (tv_end.tv_usec / 1000000.0);

    return (end_sec - start_sec);
}

} // file scope

void warm_up_cpu(double seconds = 1.0)
{
    // One second of activity to push up the CPU speed if boost is enabled
    struct timeval tv;
    gettimeofday(&tv, 0);
    
    while (elapsed_since(tv) < seconds);
    
}

double builtin_expf_array(float * vals, int nvals)
{
    sched_yield();
    size_t before = ticks();

    for (unsigned i = 0;  i < nvals;  ++i)
        vals[i] = expf(vals[i]);

    size_t after = ticks();

    return (after - before);
}

double builtin_exp_array(float * vals, int nvals)
{
    sched_yield();
    size_t before = ticks();

    for (unsigned i = 0;  i < nvals;  ++i)
        vals[i] = exp(double(vals[i]));

    size_t after = ticks();

    return (after - before);
}

double sse2_expf_array(float * vals, int nvals)
{
    sched_yield();
    size_t before = ticks();

    int nvecs = nvals / 4;
    v4sf * vvals = (v4sf *)vals;
    
    for (unsigned i = 0;  i < nvecs;  ++i)
        vvals[i] = sse2_expf(vvals[i]);

    size_t after = ticks();
    return (after - before);
}

double sse2_exp_array(float * vals_, int nvals)
{
    double vals[nvals];
    std::copy(vals_, vals_ + nvals, vals);

    sched_yield();
    size_t before = ticks();

    int nvecs = nvals / 2;
    v2df * vvals = (v2df *)vals;
    
    for (unsigned i = 0;  i < nvecs;  ++i)
        vvals[i] = sse2_exp(vvals[i]);

    size_t after = ticks();
    return (after - before);
}

typedef double (*profile_function) (float *, int);

void profile_expf(int nvals, const std::string & desc,
                  profile_function function)
{
    double overhead = calc_ticks_overhead();
    double tps = calc_ticks_per_second();

    // First, warm it up
    float vals[nvals];
    for (int i = 0;  i < nvals;  ++i)
        vals[i] = 10.0 * (2 * i - nvals) / (1.0 * nvals);

    function(vals, nvals);
    
    vector<double> timings(20);
    for (unsigned trial = 0;  trial < 20;  ++trial) {
        for (int i = 0;  i < nvals;  ++i)
            vals[i] = 10.0 * (2 * i - nvals) / (1.0 * nvals);
        timings[trial] = function(vals, nvals);
    }

    std::sort(timings.begin(), timings.end());

    //cerr << "timings = " << timings << endl;

    cerr << format("%-30s %8.2f %10.1f %8.5f\n", desc.c_str(),
                   (timings[10] - overhead) / (1.0 * nvals),
                   tps, overhead);
}

BOOST_AUTO_TEST_CASE( profile_expf_test )
{
    static const int NVALS = 512;

    warm_up_cpu();

    profile_expf(NVALS, "builtin", &builtin_expf_array);
    profile_expf(NVALS, "builtin double", &builtin_exp_array);
    profile_expf(NVALS, "sse2 discrete", &sse2_expf_array);
    profile_expf(NVALS, "sse2 discrete double", &sse2_exp_array);
}
