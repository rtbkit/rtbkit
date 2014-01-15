/** currency_test.cc                                 -*- C++ -*-
    Eric Robert, 31 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Tests for currency code.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/common/currency.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace std;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( currencyValues )
{
    // 1$
    auto a0 = USD(1);

    // should be a million u$
    auto a1 = MicroUSD(1000000);

    BOOST_CHECK_EQUAL(a0, a1);

    // 1$
    auto b0 = USD(1);

    // should be the same as 1000$ for a thousand
    auto b1 = USD_CPM(1000);

    BOOST_CHECK_EQUAL(b0, b1);

    // 1u$
    auto c0 = MicroUSD(1);

    // should be the same as 1000u$ for a thousand
    auto c1 = MicroUSD_CPM(1000);

    BOOST_CHECK_EQUAL(c0, c1);

    // 1$ CPM
    auto d0 = USD_CPM(1);

    // should be the same as a million u$ CPM
    auto d1 = MicroUSD_CPM(1000000);

    BOOST_CHECK_EQUAL(d0, d1);

    // and check conversion
    BOOST_CHECK_EQUAL(1, (double) USD(1));
    BOOST_CHECK_EQUAL(1, (double) USD_CPM(1));
    BOOST_CHECK_EQUAL(1, (int64_t) MicroUSD(1));

    // precision issue i.e. the value is rounded to the closed 1K
    BOOST_CHECK_EQUAL(1000, (int64_t) MicroUSD_CPM(1000));
}

static inline void test2(CurrencyPool)
{
}

BOOST_AUTO_TEST_CASE( currencyConversion)
{
    {
        USD_CPM price;
        double value = price;
        (void)value;
    }

    {
        MicroUSD_CPM price;
        int64_t value = price;
        (void)value;
    }

    {
        USD price;
        double value = price;
        (void)value;
    }

    {
        MicroUSD price;
        int64_t value = price;
        (void)value;
    }

    {
        USD price;
        test2(price);
    }
}
