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

int extract_scalar(v4si i)
{
    int vals[4];
    *((v4si *)vals) = i;
    BOOST_CHECK_EQUAL(vals[0], vals[1]);
    BOOST_CHECK_EQUAL(vals[0], vals[2]);
    BOOST_CHECK_EQUAL(vals[0], vals[3]);
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

BOOST_AUTO_TEST_CASE( ldexp_test )
{
    BOOST_CHECK_EQUAL(ldexp(1.0, 0), extract_scalar(ldexp(vec_splat(1.0), vec_splat(0))));
    BOOST_CHECK_EQUAL(ldexp(1.0, 1), extract_scalar(ldexp(vec_splat(1.0), vec_splat(1))));
}

void test_functions(double val)
{
    cerr << format("%5.2f  %5.2f  %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n",
                   val,
                   floor(val + 0.5),
                   round(val),
                   trunc(val),
                   floor(val),
                   ceil(val),
                   rint(val),
                   double(int(val)));
}

BOOST_AUTO_TEST_CASE( value_test )
{
    cerr << "  val   func  round trunc floor  ceil  rint  conv" << endl;
    test_functions(0.0);
    cerr << endl;
    test_functions(0.1);
    test_functions(0.49);
    test_functions(0.50);
    test_functions(0.51);
    test_functions(0.99);
    test_functions(1.00);
    test_functions(1.01);
    test_functions(1.50);
    test_functions(1.51);
    cerr << endl;
    test_functions(-0.1);
    test_functions(-0.49);
    test_functions(-0.50);
    test_functions(-0.51);
    test_functions(-0.99);
    test_functions(-1.00);
    test_functions(-1.01);
    test_functions(-1.50);
    test_functions(-1.51);
    cerr << endl;
}

#define test_floor_value(input) \
{ \
    float in2 = float(input); \
    float output1 = floorf(in2); \
    float output2 = extract_scalar(sse2_floor(vec_splat(in2))); \
    if (isnan(output1)) \
        BOOST_CHECK(isnan(output2));            \
    else BOOST_CHECK_EQUAL(floorf(float(input)), output2);      \
} \
{ \
    double in2 = double(input); \
    double output1 = floorf(in2); \
    double output2 = extract_scalar(sse2_floor(vec_splat(in2))); \
    if (isnan(output1)) \
        BOOST_CHECK(isnan(output2));            \
    else BOOST_CHECK_EQUAL(floorf(double(input)), output2);      \
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
    test_floor_value(NaN);
    test_floor_value(-NaN);
    test_floor_value(INFINITY);
    test_floor_value(-INFINITY);
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
    test_trunc_value(NaN);
    test_trunc_value(-NaN);
    test_trunc_value(INFINITY);
    test_trunc_value(-INFINITY);
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
                cerr << format("%12.8f: %14.9f != %14.9f: %08x != %08x (%4d ulps)\n", \
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
            /* Allow 2 ulps as there are a very few values that */      \
            /* have this error */                                       \
            if (abs(i1 - i2) > 2) {                                     \
                cerr << format("%12.8f: %14.9f != %14.9f: %08x != %08x (%4d ulps)\n", \
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

    // Test over the whole range
    int nvals = 65536;
    for (int i = 0;  i < nvals;  ++i) {
        double f = 1000.0 * (2 * i - nvals) / (1.0 * nvals);
        test_exp_value(f);
    }

    // Test more closely over (-10, 10)
    for (int i = 0;  i < nvals;  ++i) {
        double f = 10.0 * (2 * i - nvals) / (1.0 * nvals);
        test_exp_value(f);
    }
}

inline double pow2(int input)
{
    return pow(2.0, double(input));
}

#define test_pow2_value(input)                                          \
    {                                                                   \
        double output1 = pow2(input);                                      \
        double output2 = extract_scalar(sse2_pow2(vec_splat(input)));      \
        if (isnan(output1)) {                                           \
            if (!isnan(output2)) {                                      \
                cerr << "input = " << input << " output1 = " << output1 << " output2 = " << output2 \
                 << endl;                                               \
            }                                                           \
            BOOST_CHECK(isnan(output2));                                \
        }                                                               \
        else if (output1 != output2) {                                  \
            int i1 = reinterpret_as_int(output1);                       \
            int i2 = reinterpret_as_int(output2);                       \
            if (abs(i1 - i2) > 1) {                                     \
                cerr << format("%12.8f: %14.9f != %14.9f: %08x != %08x (%4d ulps)\n", \
                               input, output1, output2, i1, i2, (i1 - i2)); \
                BOOST_CHECK_EQUAL(pow2(double(input)), output2);         \
            }                                                           \
        }                                                               \
    }

BOOST_AUTO_TEST_CASE( test_pow2 )
{
    test_pow2_value(0);
    test_pow2_value(1);
    test_pow2_value(2);
    test_pow2_value(3);
    test_pow2_value(-0);
    test_pow2_value(-1);
    test_pow2_value(-2);
    test_pow2_value(-3);

    test_pow2_value(-10);
    test_pow2_value(-20);
    test_pow2_value(-30);
    test_pow2_value(-50);
    test_pow2_value(-100);
    test_pow2_value(-1000);

    test_pow2_value(10);
    test_pow2_value(20);
    test_pow2_value(30);
    test_pow2_value(50);
    test_pow2_value(100);
    test_pow2_value(1000);

#if 0
    test_pow2_value(NaN);
    test_pow2_value(-NaN);
    test_pow2_value(INFINITY);
    test_pow2_value(-INFINITY);

    // Test over the whole range
    int nvals = 65536;
    for (int i = 0;  i < nvals;  ++i) {
        double f = 1000.0 * (2 * i - nvals) / (1.0 * nvals);
        test_pow2_value(f);
    }

    // Test more closely over (-10, 10)
    for (int i = 0;  i < nvals;  ++i) {
        double f = 10.0 * (2 * i - nvals) / (1.0 * nvals);
        test_pow2_value(f);
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
    cerr << endl;
    cerr << "profiling expf" << endl;

    static const int NVALS = 512;

    warm_up_cpu();

    profile_expf(NVALS, "builtin", &builtin_expf_array);
    profile_expf(NVALS, "builtin double", &builtin_exp_array);
    profile_expf(NVALS, "sse2 discrete", &sse2_expf_array);
    profile_expf(NVALS, "sse2 discrete double", &sse2_exp_array);
}

#if 0
v4sf sse2_logf(v4sf val)
{
    float f[4];
    *(v4sf *)f = val;

    f[0] = logf(f[0]);
    f[1] = logf(f[1]);
    f[2] = logf(f[2]);
    f[3] = logf(f[3]);

    return *(v4sf *)f;
}
#endif

// Splits a number into a coefficient and an integral power of 2 such that
// x = y * 2^exp.  Doesn't handle denormals, zeros, NaN or infinities.
v4sf sse2_frexpf_unsafe_nz_nodn( v4sf x, v4si & exp)
{
    //cerr << "x = " << x << endl;

    v4si exps = (v4si)x;
    // Shift out the mantissa
    exps = __builtin_ia32_psrldi128(exps, 23);
    
    //cerr << "  exps = " << exps << endl;

    // Clear out the sign
    exp = (v4si)__builtin_ia32_andps((v4sf)exps, (v4sf)vec_splat(0x000000ff));

    //cerr << "  exp = " << exp << endl;

    // Subtract the offset
    exp -= vec_splat(0x7e);

    //cerr << "  exp = " << exp << endl;

    // Mantissa: keep it and the sign, then add in a bias to make it one
    x = __builtin_ia32_andps(x, (v4sf)vec_splat((int)0x807fffff));
    x = __builtin_ia32_orps(x, (v4sf)vec_splat(0x3f000000));

    //cerr << "  result = " << x << endl;
    
    return x;
}

v4sf sse2_frexpf_unsafe_nodn( v4sf x, v4si & exp)
{
    v4sf iszero = (v4sf)__builtin_ia32_cmpeqps(x, vec_splat(0.0f));
    v4sf res    = sse2_frexpf_unsafe_nz_nodn(x, exp);
    res         = __builtin_ia32_andnps((v4sf)iszero, res);
    exp         = (v4si)__builtin_ia32_andnps((v4sf)iszero, (v4sf)exp);
    return res;
}

v4sf sse2_frexpf_nodn( v4sf x, v4si & exp)
{
    return pass_nan_inf(x, sse2_frexpf_unsafe_nodn(x, exp));
}

v4sf sse2_frexpf( v4sf x, v4si & exp)
{
    return sse2_frexpf_nodn(x, exp);
}

#if 0
// Splits a number into a coefficient and an integral power of 2 such that
// x = y * 2^exp
float frexpf( float x, v4si & exp)
{
    // Exponent: right 
    union {
        float y;
        unsigned short i[2];
    } u;
    int i, k;
    short *q;
    
    u.y = x;

    q = &u.i[1];

    /* find the exponent (power of 2) */

    i  = ( *q >> 7) & 0xff;

    if( i == 0 ) {
	if( u.y == 0.0 ) {
            *pw2 = 0;
            return(0.0);
        }
        /* Number is denormal or zero */
	do {
            u.y *= 2.0;
            i -= 1;
            k  = ( *q >> 7) & 0xff;
        }
	while( k == 0 );
	i = i + k;
    }
    i -= 0x7e;
    *pw2 = i;
    *q &= 0x807f;	/* strip all exponent bits */
    *q |= 0x3f00;	/* mantissa between 0.5 and 1 */
    return( u.y );
}
#endif

float MINLOGF = -103.278929903431851103f; /* log(2^-149) */
float SQRTHF = 0.707106781186547524f;

// Doesn't handle NaN, exp, inf, zeros or negative numbers

inline v4sf polevl( v4sf x, float * coef, int N )
{
    float * p = coef;
    v4sf ans = vec_splat(*p++);
    int i = N;

    do ans = ans * x + vec_splat(*p++);
    while (--i);

    return ans;
}

inline v4sf polevl_1( v4sf x, float * coef, int N )
{
    float * p = coef;
    v4sf ans = vec_splat(*p++) * x;
    int i = N;

    do ans = (ans + vec_splat(*p++)) * x;
    while (--i);
    
    return ans;
}

inline float polevl( float x, float * coef, int N )
{
    float * p = coef;
    float ans = *p++;
    int i = N;

    do ans = ans * x + *p++;
    while (--i);

    return ans;
}


float logf_coef[9] = { 
      7.0376836292E-2f,
    - 1.1514610310E-1f,
    + 1.1676998740E-1f,
    - 1.2420140846E-1f,
    + 1.4249322787E-1f,
    - 1.6668057665E-1f,
    + 2.0000714765E-1f,
    - 2.4999993993E-1f,
    + 3.3333331174E-1f };


inline v4sf sse2_logf_unsafe(v4sf xx)
{
    v4sf y, x, z;
    v4si e;

    float x_, z_, fe_;
    int e_;

    x_ = extract_scalar(xx);
    fe_ = 0.0;

    x = xx;
    x = sse2_frexpf_unsafe_nz_nodn( x, e );

    x_ = extract_scalar(xx);
    x_ = frexpf( x_, &e_ );

    if( x_ < SQRTHF ) {
        e_ -= 1;
	x_ = x_ + x_ - 1.0; /*  2x - 1  */
    }	
    else x_ = x_ - 1.0;

    z_ = x_ * x_;

    v4sf ltsqrthf = (v4sf)__builtin_ia32_cmpltps(x, vec_splat(SQRTHF));
    v4si minusone = (v4si)__builtin_ia32_andps(ltsqrthf, (v4sf)vec_splat(-1));
    v4sf xmasked  = __builtin_ia32_andps(ltsqrthf, x);

    x = x + xmasked;
    x = x - vec_splat(1.0f);
    e = e + minusone;

    z = x * x;

    cerr << "xx= " << xx << endl;
    cerr << "x = " << x << " x_ = " << x_ << endl;
    cerr << "e = " << e << " e_ = " << e_ << endl;
    cerr << "z = " << z << " z_ = " << z_ << endl;



    y = polevl_1(x, logf_coef, 8) * z;

    float y_ = polevl(x_, logf_coef, 8) * z_;

    float y2_ =
        (((((((( 7.0376836292E-2 * x_
                 - 1.1514610310E-1) * x_
               + 1.1676998740E-1) * x_
              - 1.2420140846E-1) * x_
             + 1.4249322787E-1) * x_
            - 1.6668057665E-1) * x_
           + 2.0000714765E-1) * x_
          - 2.4999993993E-1) * x_
         + 3.3333331174E-1) * x_ * z_;

    float y3_ =
        (((((((( 7.0376836292E-2f * x_
                 - 1.1514610310E-1f) * x_
               + 1.1676998740E-1f) * x_
              - 1.2420140846E-1f) * x_
             + 1.4249322787E-1f) * x_
            - 1.6668057665E-1f) * x_
           + 2.0000714765E-1f) * x_
          - 2.4999993993E-1f) * x_
         + 3.3333331174E-1f) * x_ * z_;

    cerr << "y = " << y << " y_ = " << y_ << " y2_ = " << y2_
         << " y3_ = " << y3_ << endl;

    v4sf fe = __builtin_ia32_cvtdq2ps(e);
    y += vec_splat(-2.12194440e-4f) * fe;
    y += vec_splat(-0.5f) * z;
    z = x + y;
    z += vec_splat(0.693359375f) * fe;

    return z;
}

inline v4sf sse2_logf(v4sf x)
{
    int mask = 0;

    // For out of range results, we have to use the other values
    if (JML_UNLIKELY(mask = out_of_range_mask_oo(x, 0.0f, INFINITY))) {
        //using namespace std;
        //cerr << "mask = " << mask << " x = " << x << endl;

        v4sf unsafe_result = vec_splat(0.0f);
        if (mask != 15)
            unsafe_result = sse2_logf_unsafe(x);
        float xin[4];
        unpack(x, xin);
        
        float xout[4];
        unpack(unsafe_result, xout);
        
        for (unsigned i = 0;  i < 4;  ++i)
            if (mask & (1 << i))
                xout[i] = logf(xin[i]);

        return pass_nan(x, pack(xout));
    }

    return pass_nan(x, sse2_logf_unsafe(x));
}

v2df sse2_log(v2df val)
{
    double f[2];
    *(v2df *)f = val;

    f[0] = log(f[0]);
    f[1] = log(f[1]);

    return *(v2df *)f;
}

#define test_frexpf_value(input)                                          \
    {                                                                   \
        float in2 = float(input);                                       \
        int exp1 = 0, exp2 = 0;                                         \
        float output1 = frexpf(in2, &exp1);                              \
        v4si eexp2;                                                     \
        float output2 = extract_scalar(sse2_frexpf(vec_splat(in2), eexp2)); \
        exp2 = extract_scalar(eexp2);                                   \
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
                cerr << format("%12.8f: %14.9f != %14.9f: %08x != %08x (%4d ulps)\n", \
                               in2, output1, output2, i1, i2, (i1 - i2)); \
                BOOST_CHECK_EQUAL(frexpf(float(input), &exp1), output2); \
            }                                                           \
        }                                                               \
        if (exp1 != exp2 && isfinite(input)) {                          \
            BOOST_CHECK_EQUAL(make_pair(input, exp1).second, exp2);     \
        }                                                               \
    }

BOOST_AUTO_TEST_CASE( test_frexpf )
{
    test_frexpf_value(0.0);
    test_frexpf_value(1.0);
    test_frexpf_value(2.0);
    test_frexpf_value(3.0);
    test_frexpf_value(-0.0);
    test_frexpf_value(-1.0);
    test_frexpf_value(-2.0);
    test_frexpf_value(-3.0);

    test_frexpf_value(-10.0);
    test_frexpf_value(-20.0);
    test_frexpf_value(-30.0);
    test_frexpf_value(-50.0);
    test_frexpf_value(-100.0);
    test_frexpf_value(-1000.0);

    test_frexpf_value(10.0);
    test_frexpf_value(20.0);
    test_frexpf_value(30.0);
    test_frexpf_value(50.0);
    test_frexpf_value(100.0);
    test_frexpf_value(1000.0);

    test_frexpf_value(-0.0);

    test_frexpf_value(NaN);
    test_frexpf_value(-NaN);
    test_frexpf_value(INFINITY);
    test_frexpf_value(-INFINITY);

#if 0
    int nvals = 65536;
    for (int i = 0;  i < nvals;  ++i) {
        float f = 105.0 * (2 * i - nvals) / (1.0 * nvals);
        test_frexp_value(f);
    }
#endif
}


#define test_logf_value(input)                                          \
    {                                                                   \
        float in2 = float(input);                                       \
        float output1 = logf(in2);                                      \
        float output2 = extract_scalar(sse2_logf(vec_splat(in2)));      \
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
                cerr << format("%12.8f: %14.9f != %14.9f: %08x != %08x (%4d ulps)\n", \
                               in2, output1, output2, i1, i2, (i1 - i2)); \
                BOOST_CHECK_EQUAL(logf(float(input)), output2);         \
            }                                                           \
        }                                                               \
    }

BOOST_AUTO_TEST_CASE( test_logf )
{
    test_logf_value(0.0);
    test_logf_value(1.0);
    test_logf_value(2.0);
    test_logf_value(3.0);
    test_logf_value(-0.0);
    test_logf_value(-1.0);
    test_logf_value(-2.0);
    test_logf_value(-3.0);

    test_logf_value(-10.0);
    test_logf_value(-20.0);
    test_logf_value(-30.0);
    test_logf_value(-50.0);
    test_logf_value(-100.0);
    test_logf_value(-1000.0);

    test_logf_value(10.0);
    test_logf_value(20.0);
    test_logf_value(30.0);
    test_logf_value(50.0);
    test_logf_value(100.0);
    test_logf_value(1000.0);

    test_logf_value(-0.0);
    test_logf_value(NaN);
    test_logf_value(-NaN);
    test_logf_value(INFINITY);
    test_logf_value(-INFINITY);

#if 0
    int nvals = 65536;
    for (int i = 0;  i < nvals;  ++i) {
        float f = 105.0 * (2 * i - nvals) / (1.0 * nvals);
        test_logf_value(f);
    }
#endif
}

double builtin_logf_array(float * vals, int nvals)
{
    sched_yield();
    size_t before = ticks();

    for (unsigned i = 0;  i < nvals;  ++i)
        vals[i] = logf(vals[i]);

    size_t after = ticks();

    return (after - before);
}

double builtin_log_array(float * vals, int nvals)
{
    sched_yield();
    size_t before = ticks();

    for (unsigned i = 0;  i < nvals;  ++i)
        vals[i] = log(double(vals[i]));

    size_t after = ticks();

    return (after - before);
}

double sse2_logf_array(float * vals, int nvals)
{
    sched_yield();
    size_t before = ticks();

    int nvecs = nvals / 4;
    v4sf * vvals = (v4sf *)vals;
    
    for (unsigned i = 0;  i < nvecs;  ++i)
        vvals[i] = sse2_logf(vvals[i]);

    size_t after = ticks();
    return (after - before);
}

double sse2_log_array(float * vals_, int nvals)
{
    double vals[nvals];
    std::copy(vals_, vals_ + nvals, vals);

    sched_yield();
    size_t before = ticks();

    int nvecs = nvals / 2;
    v2df * vvals = (v2df *)vals;
    
    for (unsigned i = 0;  i < nvecs;  ++i)
        vvals[i] = sse2_log(vvals[i]);

    size_t after = ticks();
    return (after - before);
}

typedef double (*profile_function) (float *, int);

void profile_logf(int nvals, const std::string & desc,
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

BOOST_AUTO_TEST_CASE( profile_logf_test )
{
    cerr << endl;
    cerr << "profiling logf" << endl;

    static const int NVALS = 512;

    warm_up_cpu();

    profile_logf(NVALS, "builtin", &builtin_logf_array);
    profile_logf(NVALS, "builtin double", &builtin_log_array);
    profile_logf(NVALS, "sse2 discrete", &sse2_logf_array);
    profile_logf(NVALS, "sse2 discrete double", &sse2_log_array);
}

