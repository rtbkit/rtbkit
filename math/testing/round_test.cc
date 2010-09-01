/* round_test.cc                                            -*- C++ -*-
   Jeremy Barnes, 16 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of rounding function.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/math/round.h"
#include <limits>
#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <stdint.h>

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

template<size_t width>
struct equivalent_int {};

template<>
struct equivalent_int<4> {
    typedef uint32_t type;
};

template<>
struct equivalent_int<8> {
    typedef uint64_t type;
};

template<class X>
typename equivalent_int<sizeof(X)>::type
bitsof(X x)
{
    union {
        X x;
        typename equivalent_int<sizeof(X)>::type eq;
    } res;
    res.x = x;
    return res.eq;
}

template<class X>
void test_type()
{
    BOOST_CHECK_BITWISE_EQUAL(bitsof<X>(round(0.0)), bitsof<X>(0.0));
    BOOST_CHECK_BITWISE_EQUAL(bitsof<X>(round(-0.0)), bitsof<X>(-0.0));
    BOOST_CHECK_EQUAL(round((X) 1.0),  (X) 1.0);
    BOOST_CHECK_EQUAL(round((X)-1.0),  (X)-1.0);
    BOOST_CHECK_EQUAL(round((X) 0.5),  (X) 1.0);
    BOOST_CHECK_EQUAL(round((X)-0.5),  (X)-1.0);
    BOOST_CHECK_EQUAL(round((X) 0.49), (X) 0.0);
    BOOST_CHECK_EQUAL(round((X)-0.49), (X)-0.0);
    BOOST_CHECK_EQUAL(round((X) 0.49), (X) 0.0);
    BOOST_CHECK_EQUAL(round((X) 0.51), (X) 1.0);
    BOOST_CHECK_EQUAL(round((X)-0.51), (X)-1.0);
    BOOST_CHECK_EQUAL(round((X) 11.0),  (X) 11.0);
    BOOST_CHECK_EQUAL(round((X)-11.0),  (X)-11.0);
    BOOST_CHECK_EQUAL(round((X) 10.5),  (X) 11.0);
    BOOST_CHECK_EQUAL(round((X)-10.5),  (X)-11.0);
    BOOST_CHECK_EQUAL(round((X) 10.49), (X) 10.0);
    BOOST_CHECK_EQUAL(round((X)-10.49), (X)-10.0);
    BOOST_CHECK_EQUAL(round((X) 10.49), (X) 10.0);
    BOOST_CHECK_EQUAL(round((X) 10.51), (X) 11.0);
    BOOST_CHECK_EQUAL(round((X)-10.51), (X)-11.0);
    BOOST_CHECK_BITWISE_EQUAL(bitsof<X>(round(INFINITY)), bitsof<X>(INFINITY));
    BOOST_CHECK_BITWISE_EQUAL(bitsof<X>(round(-INFINITY)),
                              bitsof<X>(-INFINITY));
    BOOST_CHECK_BITWISE_EQUAL(bitsof<X>(round(numeric_limits<X>::quiet_NaN())),
                              bitsof<X>(numeric_limits<X>::quiet_NaN()));
}

BOOST_AUTO_TEST_CASE( test1 )
{
    test_type<float>();
    test_type<double>();
}
