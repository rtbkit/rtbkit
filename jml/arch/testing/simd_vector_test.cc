/* cpuid_test.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of the CPUID detection code.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/arch/simd_vector.h"
#include "jml/arch/demangle.h"

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <vector>
#include <set>
#include <iostream>
#include <cmath>


using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

BOOST_AUTO_TEST_CASE( test1 )
{
    BOOST_CHECK(cpuid_flags() != 0);
}

void vec_scale_k_test_case(int nvals)
{
    cerr << "testing vec_scale with nvals: " << nvals << endl;

    double x[nvals], r[nvals], r2[nvals];

    double k = 3.0;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        r2[i] = x[i] * k;
    }

    SIMD::vec_scale(x, k, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

BOOST_AUTO_TEST_CASE( vec_scale_k_mixed_test_case )
{
    vec_scale_k_test_case(1);
    vec_scale_k_test_case(2);
    vec_scale_k_test_case(3);
    vec_scale_k_test_case(4);
    vec_scale_k_test_case(5);
    vec_scale_k_test_case(12);
    vec_scale_k_test_case(16);
    vec_scale_k_test_case(123);
}


void vec_add_k_mixed_test_case(int nvals)
{
    cerr << "testing " << nvals << endl;

    double x[nvals], r[nvals], r2[nvals];
    float y[nvals];

    double k = 3.0;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        r2[i] = x[i] + k * y[i];
    }

    SIMD::vec_add(x, k, y, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

BOOST_AUTO_TEST_CASE( vec_add_k_mixed_test )
{
    vec_add_k_mixed_test_case(1);
    vec_add_k_mixed_test_case(2);
    vec_add_k_mixed_test_case(3);
    vec_add_k_mixed_test_case(4);
    vec_add_k_mixed_test_case(5);
    vec_add_k_mixed_test_case(8);
    vec_add_k_mixed_test_case(9);
    vec_add_k_mixed_test_case(12);
    vec_add_k_mixed_test_case(16);
    vec_add_k_mixed_test_case(123);
}

void vec_add_k_double_test_case(int nvals)
{
    cerr << "testing " << nvals << endl;

    double x[nvals], r[nvals], r2[nvals], y[nvals];

    double k = 3.0;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        r2[i] = x[i] + k * y[i];
    }

    SIMD::vec_add(x, k, y, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

BOOST_AUTO_TEST_CASE( vec_add_k_double_test )
{
    vec_add_k_double_test_case(1);
    vec_add_k_double_test_case(2);
    vec_add_k_double_test_case(3);
    vec_add_k_double_test_case(4);
    vec_add_k_double_test_case(5);
    vec_add_k_double_test_case(8);
    vec_add_k_double_test_case(9);
    vec_add_k_double_test_case(12);
    vec_add_k_double_test_case(16);
    vec_add_k_double_test_case(123);
}

void vec_add_k_float_test_case(int nvals)
{
    cerr << "testing " << nvals << endl;

    float x[nvals], r[nvals], r2[nvals], y[nvals];

    float k = 3.0;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        r2[i] = x[i] + k * y[i];
    }

    SIMD::vec_add(x, k, y, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

BOOST_AUTO_TEST_CASE( vec_add_k_float_test )
{
    vec_add_k_float_test_case(1);
    vec_add_k_float_test_case(2);
    vec_add_k_float_test_case(3);
    vec_add_k_float_test_case(4);
    vec_add_k_float_test_case(5);
    vec_add_k_float_test_case(8);
    vec_add_k_float_test_case(9);
    vec_add_k_float_test_case(12);
    vec_add_k_float_test_case(16);
    vec_add_k_float_test_case(123);
}

void vec_prod_double_test_case(int nvals)
{
    cerr << "testing " << nvals << endl;

    double x[nvals], r[nvals], r2[nvals], y[nvals];

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        r2[i] = x[i] * y[i];
    }

    SIMD::vec_prod(x, y, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

BOOST_AUTO_TEST_CASE( vec_prod_double_test )
{
    vec_prod_double_test_case(1);
    vec_prod_double_test_case(2);
    vec_prod_double_test_case(3);
    vec_prod_double_test_case(4);
    vec_prod_double_test_case(5);
    vec_prod_double_test_case(8);
    vec_prod_double_test_case(9);
    vec_prod_double_test_case(12);
    vec_prod_double_test_case(16);
    vec_prod_double_test_case(123);
}

void vec_prod_float_test_case(int nvals)
{
    cerr << "testing " << nvals << endl;

    float x[nvals], r[nvals], r2[nvals], y[nvals];

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        r2[i] = x[i] * y[i];
    }

    SIMD::vec_prod(x, y, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

BOOST_AUTO_TEST_CASE( vec_prod_float_test )
{
    vec_prod_float_test_case(1);
    vec_prod_float_test_case(2);
    vec_prod_float_test_case(3);
    vec_prod_float_test_case(4);
    vec_prod_float_test_case(5);
    vec_prod_float_test_case(8);
    vec_prod_float_test_case(9);
    vec_prod_float_test_case(12);
    vec_prod_float_test_case(16);
    vec_prod_float_test_case(123);
}

float get_eps(float)
{
    return 1e-7;
}

double get_eps(double)
{
    return 1e-10;
}

template<typename T>
void vec_dotprod_test_case(int nvals)
{
    cerr << "nvals = " << nvals << endl;

    T x[nvals], y[nvals], r = 0.0, r2;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        r += x[i] * y[i];
    }
    
    r2 = SIMD::vec_dotprod(x, y, nvals);

    T eps = get_eps(T());
    BOOST_CHECK(fabs(r - r2) / max(fabs(r), fabs(r2)) < eps);
}

template<typename T>
void vec_dotprod_dp_test_case(int nvals)
{
    T x[nvals], y[nvals];
    double r = 0.0, r2;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        r += x[i] * y[i];
    }
    
    r2 = SIMD::vec_dotprod_dp(x, y, nvals);

    T eps = get_eps(T());
    if (fabs(r - r2) / max(fabs(r), fabs(r2)) >= eps) {
        cerr << "difference = " << (abs(r - r2) / max(abs(r), abs(r2)))
             << endl;
        BOOST_CHECK_EQUAL(r, r2);
    }
    BOOST_CHECK(fabs(r - r2) / max(fabs(r), fabs(r2)) < eps);
}

void vec_dotprod_dp_mixed_test_case(int nvals)
{
    double x[nvals];
    float y[nvals];
    double r = 0.0, r2;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        r += x[i] * y[i];
    }
    
    r2 = SIMD::vec_dotprod_dp(x, y, nvals);

    double eps = get_eps(0.0);
    if (fabs(r - r2) / max(fabs(r), fabs(r2)) >= eps) {
        cerr << "difference = " << (abs(r - r2) / max(abs(r), abs(r2)))
             << endl;
        BOOST_CHECK_EQUAL(r, r2);
    }
    BOOST_CHECK(fabs(r - r2) / max(fabs(r), fabs(r2)) < eps);
}

BOOST_AUTO_TEST_CASE(vec_dotprod_test)
{
    cerr << "float" << endl;
    vec_dotprod_test_case<float>(1);
    vec_dotprod_test_case<float>(2);
    vec_dotprod_test_case<float>(3);
    vec_dotprod_test_case<float>(4);
    vec_dotprod_test_case<float>(5);
    vec_dotprod_test_case<float>(8);
    vec_dotprod_test_case<float>(9);
    vec_dotprod_test_case<float>(12);
    vec_dotprod_test_case<float>(16);
    vec_dotprod_test_case<float>(123);

    cerr << "float dp" << endl;
    vec_dotprod_dp_test_case<float>(1);
    vec_dotprod_dp_test_case<float>(2);
    vec_dotprod_dp_test_case<float>(3);
    vec_dotprod_dp_test_case<float>(4);
    vec_dotprod_dp_test_case<float>(5);
    vec_dotprod_dp_test_case<float>(8);
    vec_dotprod_dp_test_case<float>(9);
    vec_dotprod_dp_test_case<float>(12);
    vec_dotprod_dp_test_case<float>(16);
    vec_dotprod_dp_test_case<float>(123);

    cerr << "double" << endl;
    vec_dotprod_test_case<double>(1);
    vec_dotprod_test_case<double>(2);
    vec_dotprod_test_case<double>(3);
    vec_dotprod_test_case<double>(4);
    vec_dotprod_test_case<double>(5);
    vec_dotprod_test_case<double>(8);
    vec_dotprod_test_case<double>(9);
    vec_dotprod_test_case<double>(12);
    vec_dotprod_test_case<double>(16);
    vec_dotprod_test_case<double>(123);

    cerr << "mixed" << endl;
    vec_dotprod_dp_mixed_test_case(1);
    vec_dotprod_dp_mixed_test_case(2);
    vec_dotprod_dp_mixed_test_case(3);
    vec_dotprod_dp_mixed_test_case(4);
    vec_dotprod_dp_mixed_test_case(5);
    vec_dotprod_dp_mixed_test_case(8);
    vec_dotprod_dp_mixed_test_case(9);
    vec_dotprod_dp_mixed_test_case(12);
    vec_dotprod_dp_mixed_test_case(16);
    vec_dotprod_dp_mixed_test_case(123);
}

template<typename T1, typename T2>
void vec_accum_prod3_test_case(int nvals)
{
    cerr << "testing vec_accum_prod3 nvals " << nvals << " T1 "
         << demangle(typeid(T1).name())
         << " T2 " << demangle(typeid(T2).name())
         << endl;

    T1 x[nvals], y[nvals];
    T2 z[nvals];
    double r = 0.0, r2;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        z[i] = rand() / 16384.0;
        r += x[i] * y[i] * z[i];
    }
    
    r2 = SIMD::vec_accum_prod3(x, y, z, nvals);

    T1 eps = get_eps(T1());
    if (fabs(r - r2) / max(fabs(r), fabs(r2)) >= eps) {
        cerr << "difference = " << (abs(r - r2) / max(abs(r), abs(r2)))
             << endl;
        BOOST_CHECK_EQUAL(r, r2);
    }
    BOOST_CHECK(fabs(r - r2) / max(fabs(r), fabs(r2)) < eps);
}

template<typename Float1, typename Float2>
void vec_accum_prod3_test_cases()
{
    vec_accum_prod3_test_case<Float1, Float2>(1);
    vec_accum_prod3_test_case<Float1, Float2>(2);
    vec_accum_prod3_test_case<Float1, Float2>(3);
    vec_accum_prod3_test_case<Float1, Float2>(4);
    vec_accum_prod3_test_case<Float1, Float2>(5);
    vec_accum_prod3_test_case<Float1, Float2>(8);
    vec_accum_prod3_test_case<Float1, Float2>(9);
    vec_accum_prod3_test_case<Float1, Float2>(12);
    vec_accum_prod3_test_case<Float1, Float2>(16);
    vec_accum_prod3_test_case<Float1, Float2>(123);
}

BOOST_AUTO_TEST_CASE(vec_accum_prod3_test)
{
    vec_accum_prod3_test_cases<float, float>();
    vec_accum_prod3_test_cases<float, double>();
    vec_accum_prod3_test_cases<double, float>();
    vec_accum_prod3_test_cases<double, double>();
}

template<typename Float1, typename Float2>
void vec_add_sqr_test_case(int nvals)
{
    cerr << "testing vec_add_sqr nvals " << nvals << " float1 "
         << demangle(typeid(Float1).name())
         << " float2 " << demangle(typeid(Float2).name())
         << endl;

    Float1 x[nvals], r[nvals], r2[nvals];
    Float2 y[nvals];

    Float1 k = 3.0;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        r2[i] = x[i] + k * (y[i] * y[i]);
    }

    SIMD::vec_add_sqr(x, k, y, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        if (r[i] != r2[i]) ;
            
        BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

template<typename Float1, typename Float2>
void vec_add_sqr_test_cases()
{
    vec_add_sqr_test_case<Float1, Float2>(1);
    vec_add_sqr_test_case<Float1, Float2>(2);
    vec_add_sqr_test_case<Float1, Float2>(3);
    vec_add_sqr_test_case<Float1, Float2>(4);
    vec_add_sqr_test_case<Float1, Float2>(5);
    vec_add_sqr_test_case<Float1, Float2>(8);
    vec_add_sqr_test_case<Float1, Float2>(9);
    vec_add_sqr_test_case<Float1, Float2>(12);
    vec_add_sqr_test_case<Float1, Float2>(16);
    vec_add_sqr_test_case<Float1, Float2>(123);
}

BOOST_AUTO_TEST_CASE( vec_add_sqr_test )
{
    vec_add_sqr_test_cases<float, float>();
    vec_add_sqr_test_cases<float, double>();
    vec_add_sqr_test_cases<double, float>();
    vec_add_sqr_test_cases<double, double>();
}

template<typename Float1, typename Float2, typename Float3>
void vec_add_3array_test_case(int nvals)
{
    cerr << "testing vec_add_sqr nvals " << nvals << " float1 "
         << demangle(typeid(Float1).name())
         << " float2 " << demangle(typeid(Float2).name())
         << " float3 " << demangle(typeid(Float3).name())
         << endl;

    Float1 x[nvals], r[nvals], r2[nvals];
    Float2 y[nvals];
    Float3 k[nvals];

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        k[i] = rand() / 16384.0;
        r2[i] = x[i] + k[i] * y[i];
    }

    SIMD::vec_add(x, k, y, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        if (r[i] != r2[i]) cerr << "difference on element " << i << " of "
                                << nvals << endl;;
            
        BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

template<typename Float1, typename Float2, typename Float3>
void vec_add_3array_test_cases()
{
    vec_add_3array_test_case<Float1, Float2, Float3>(1);
    vec_add_3array_test_case<Float1, Float2, Float3>(2);
    vec_add_3array_test_case<Float1, Float2, Float3>(3);
    vec_add_3array_test_case<Float1, Float2, Float3>(4);
    vec_add_3array_test_case<Float1, Float2, Float3>(5);
    vec_add_3array_test_case<Float1, Float2, Float3>(8);
    vec_add_3array_test_case<Float1, Float2, Float3>(9);
    vec_add_3array_test_case<Float1, Float2, Float3>(12);
    vec_add_3array_test_case<Float1, Float2, Float3>(15);
    vec_add_3array_test_case<Float1, Float2, Float3>(16);
    vec_add_3array_test_case<Float1, Float2, Float3>(17);
    vec_add_3array_test_case<Float1, Float2, Float3>(123);
}

BOOST_AUTO_TEST_CASE( vec_add_3array_test )
{
    vec_add_3array_test_cases<float, float, float>();
    vec_add_3array_test_cases<float, double, float>();
    vec_add_3array_test_cases<double, float, float>();
    vec_add_3array_test_cases<double, double, float>();
    vec_add_3array_test_cases<float, float, double>();
    vec_add_3array_test_cases<float, double, double>();
    vec_add_3array_test_cases<double, float, double>();
    vec_add_3array_test_cases<double, double, double>();
}

template<typename Float, typename Precision>
void vec_exp_test_case(int nvals)
{
    cerr << "testing vec_exp " << nvals << " float "
         << demangle(typeid(Float).name())
         << " precision " << demangle(typeid(Precision).name())
         << endl;

    Float x[nvals];
    Precision r[nvals], r2[nvals];

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0 / 1024.0;
        r2[i] = exp(Precision(x[i]));
    }

    SIMD::vec_exp(x, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        if (r[i] != r2[i]) cerr << "difference on element " << i << " of "
                                << nvals << endl;;
          
        // ./arch/testing/simd_vector_test.cc(515): error in "vec_exp_test": check r[i] == r2[i] failed [1.21850755e+20 != 1.21850746e+20]
        //BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

template<typename Float, typename Precision>
void vec_exp_test_cases()
{
    vec_exp_test_case<Float, Precision>(1);
    vec_exp_test_case<Float, Precision>(2);
    vec_exp_test_case<Float, Precision>(3);
    vec_exp_test_case<Float, Precision>(4);
    vec_exp_test_case<Float, Precision>(5);
    vec_exp_test_case<Float, Precision>(8);
    vec_exp_test_case<Float, Precision>(9);
    vec_exp_test_case<Float, Precision>(12);
    vec_exp_test_case<Float, Precision>(15);
    vec_exp_test_case<Float, Precision>(16);
    vec_exp_test_case<Float, Precision>(17);
    vec_exp_test_case<Float, Precision>(123);
}

BOOST_AUTO_TEST_CASE( vec_exp_test )
{
    vec_exp_test_cases<float, float>();
    vec_exp_test_cases<float, double>();
    vec_exp_test_cases<double, double>();
}

template<typename Float, typename Precision>
void vec_exp_k_test_case(int nvals)
{
    cerr << "testing vec_exp with k " << nvals << " float "
         << demangle(typeid(Float).name())
         << " precision " << demangle(typeid(Precision).name())
         << endl;

    Float x[nvals];
    Float k = rand();
    Precision r[nvals], r2[nvals];

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0 / 1024.0;
        r2[i] = exp((double)(k * Precision(x[i])));
    }

    SIMD::vec_exp(x, k, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

template<typename Float, typename Precision>
void vec_exp_k_test_cases()
{
    vec_exp_k_test_case<Float, Precision>(1);
    vec_exp_k_test_case<Float, Precision>(2);
    vec_exp_k_test_case<Float, Precision>(3);
    vec_exp_k_test_case<Float, Precision>(4);
    vec_exp_k_test_case<Float, Precision>(5);
    vec_exp_k_test_case<Float, Precision>(8);
    vec_exp_k_test_case<Float, Precision>(9);
    vec_exp_k_test_case<Float, Precision>(12);
    vec_exp_k_test_case<Float, Precision>(15);
    vec_exp_k_test_case<Float, Precision>(16);
    vec_exp_k_test_case<Float, Precision>(17);
    vec_exp_k_test_case<Float, Precision>(123);
}

BOOST_AUTO_TEST_CASE( vec_exp_k_test )
{
    vec_exp_k_test_cases<float, float>();
    vec_exp_k_test_cases<float, double>();
    vec_exp_k_test_cases<double, double>();
}

template<typename T>
void vec_add_mixed_test_case(int nvals)
{
    cerr << "testing vec_add_mixed " << demangle(typeid(T).name()) << 
        " with " << nvals << endl;

    double x[nvals], r[nvals], r2[nvals];
    T y[nvals];

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        r2[i] = x[i] + y[i];
    }

    SIMD::vec_add(x, y, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

BOOST_AUTO_TEST_CASE( vec_add_mixed_test )
{
    for(auto x : {1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 123}) {
        vec_add_mixed_test_case<float>(x);
        vec_add_mixed_test_case<double>(x);
    }
}


void vec_sum_dp_test_case(int nvals)
{
    cerr << "testing vec_sum_dp " << " with " << nvals << endl;

    float x[nvals];
    double r2 = 0;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        r2 += x[i];
    }

    double r = SIMD::vec_sum_dp(x, nvals);

    BOOST_CHECK_EQUAL(r, r2);
}

BOOST_AUTO_TEST_CASE( vec_sum_dp ) 
{
    for(auto x : {1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 123}) {
        vec_sum_dp_test_case(x);
    }
}


template<typename T>
void vec_k1_x_plus_k2_y_z_test_case(int nvals)
{
    cerr << "testing vec_k1_x_plus_k2_y_z_test_case " << 
        demangle(typeid(T).name()) << " with " << nvals << endl;


    T x[nvals], r[nvals], r2[nvals], y[nvals], z[nvals];

    T k1 = 3.9;
    T k2 = 8.2;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        z[i] = rand() / 16384.0;
        r2[i] = k1* x[i] + k2 * y[i] * z[i];
    }

    SIMD::vec_k1_x_plus_k2_y_z(k1, x, k2, y, z, r, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        BOOST_CHECK_EQUAL(r[i], r2[i]);
    }
}

BOOST_AUTO_TEST_CASE( vec_k1_x_plus_k2_y_z ) 
{
    for(auto x : {1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 123}) {
        vec_k1_x_plus_k2_y_z_test_case<float>(x);
        vec_k1_x_plus_k2_y_z_test_case<double>(x);
    }
}


template<typename T>
void vec_twonorm_sqr_test_case(int nvals)
{
    cerr << "testing vec_twonorm_sqr " << demangle(typeid(T).name()) << 
        " with nvals: " << nvals << endl;

    T x[nvals];
    T r2 = 0;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        r2 += x[i] * x[i];
    }
    
    T r = SIMD::vec_twonorm_sqr(x, nvals);

    BOOST_CHECK_EQUAL(r, r2);
}

BOOST_AUTO_TEST_CASE( vec_twonorm_sqr_test )
{
    for(auto x : {1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 123}) {
        vec_twonorm_sqr_test_case<float>(x);
        vec_twonorm_sqr_test_case<double>(x);
    }
}


void vec_twonorm_sqr_dp_test_case(int nvals)
{
    cerr << "testing vec_twonorm_sqr_dp " << " with nvals: " << nvals << endl;

    float x[nvals];
    double r2 = 0;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        double xx = x[i];
        r2 += xx * xx;
    }
    
    double r = SIMD::vec_twonorm_sqr_dp(x, nvals);

    BOOST_CHECK_EQUAL(r, r2);
}

BOOST_AUTO_TEST_CASE( vec_twonorm_sqr_dp_test )
{
    for(auto x : {1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 123}) {
        vec_twonorm_sqr_dp_test_case(x);
    }
}

void vec_kl_test_case(int nvals)
{
    cerr << "testing vec_kl " << " with " << nvals << endl;

    float p[nvals], q[nvals];
    double r2 = 0;

    for (unsigned i = 0; i < nvals;  ++i) {
        p[i] = rand() / 16384.0;
        q[i] = rand() / 16384.0;
        r2 += p[i] * logf(p[i] / q[i]);
    }

    double r = SIMD::vec_kl(p, q, nvals);

    BOOST_CHECK_CLOSE(r, r2, 0.00001);
}

BOOST_AUTO_TEST_CASE( vec_kl_test )
{
    //for(auto x : {1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 123}) {
    for(int x = 1; x<100; x++) {
        vec_kl_test_case(x);
    }
}


void vec_min_max_el_test_case(int nvals)
{
    cerr << "testing vec_min_max_el with " << nvals << endl;

    float x[nvals], mins[nvals], maxs[nvals];

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        mins[i] = rand() / 16384.0;
        maxs[i] = rand() / 16384.0;
    }

    SIMD::vec_min_max_el(x, mins, maxs, nvals);

    for (unsigned i = 0;  i < nvals;  ++i) {
        BOOST_CHECK( mins[i] <= x[i] );
        BOOST_CHECK( maxs[i] >= x[i] );
    }
}

BOOST_AUTO_TEST_CASE( vec_min_max_el_test )
{
    for(auto x : {1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 123}) {
        vec_min_max_el_test_case(x);
    }
}

