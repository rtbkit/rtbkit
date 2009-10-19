/* cpuid_test.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of the CPUID detection code.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "arch/simd_vector.h"

#include <boost/test/unit_test.hpp>
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

float get_eps(float)
{
    return 1e-7;
}

double get_eps(double)
{
    return 1e-10;
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

template<typename T>
void vec_accum_prod3_test_case(int nvals)
{
    T x[nvals], y[nvals], z[nvals];
    double r = 0.0, r2;

    for (unsigned i = 0; i < nvals;  ++i) {
        x[i] = rand() / 16384.0;
        y[i] = rand() / 16384.0;
        z[i] = rand() / 16384.0;
        r += x[i] * y[i] * z[i];
    }
    
    r2 = SIMD::vec_accum_prod3(x, y, z, nvals);

    T eps = get_eps(T());
    if (fabs(r - r2) / max(fabs(r), fabs(r2)) >= eps) {
        cerr << "difference = " << (abs(r - r2) / max(abs(r), abs(r2)))
             << endl;
        BOOST_CHECK_EQUAL(r, r2);
    }
    BOOST_CHECK(fabs(r - r2) / max(fabs(r), fabs(r2)) < eps);
}

BOOST_AUTO_TEST_CASE(vec_accum_prod3_test)
{
    cerr << "float" << endl;
    vec_accum_prod3_test_case<float>(1);
    vec_accum_prod3_test_case<float>(2);
    vec_accum_prod3_test_case<float>(3);
    vec_accum_prod3_test_case<float>(4);
    vec_accum_prod3_test_case<float>(5);
    vec_accum_prod3_test_case<float>(8);
    vec_accum_prod3_test_case<float>(9);
    vec_accum_prod3_test_case<float>(12);
    vec_accum_prod3_test_case<float>(16);
    vec_accum_prod3_test_case<float>(123);

    cerr << "double" << endl;
    vec_accum_prod3_test_case<double>(1);
    vec_accum_prod3_test_case<double>(2);
    vec_accum_prod3_test_case<double>(3);
    vec_accum_prod3_test_case<double>(4);
    vec_accum_prod3_test_case<double>(5);
    vec_accum_prod3_test_case<double>(8);
    vec_accum_prod3_test_case<double>(9);
    vec_accum_prod3_test_case<double>(12);
    vec_accum_prod3_test_case<double>(16);
    vec_accum_prod3_test_case<double>(123);
}
