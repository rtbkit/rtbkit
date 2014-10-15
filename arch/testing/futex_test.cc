/* futext_test.cc
   Wolfgang Sourdeau, october 10th 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   Test of the futex utility functions
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <time.h>

#include <atomic>
#include <thread>
#include <boost/test/unit_test.hpp>
#include "jml/arch/timers.h"

#include "jml/arch/futex.h"

using namespace std;

/* this helper ensures that the futex_wait does not return before futex_wake
 * is called */
template<typename T>
void
test_futex()
{
    T value(0);

    auto wakerFn = [&] () {
        ML::sleep(1.5);
        value = 5;
        ML::futex_wake(value);
    };
    thread wakerTh(wakerFn);

    time_t start = ::time(nullptr);
    while (!value) {
        ML::futex_wait(value, 0);
    }
    time_t now = ::time(nullptr);
    BOOST_CHECK(now > start);

    wakerTh.join();
}

/* this helper ensures that the futex_wait waits until timeout X when
 * specified and when the value does not change */ 
template<typename T>
void
test_futex_timeout()
{
    T value(0);
    time_t start = ::time(nullptr);
    ML::futex_wait(value, 0, 2.0);
    time_t now = ::time(nullptr);
    BOOST_CHECK(now >= (start + 1));
}

// use the above helpers for "int"
BOOST_AUTO_TEST_CASE( test_futex_int )
{
    test_futex<int>();
    test_futex_timeout<int>();
}

// use the above helpers for "volatile int"
BOOST_AUTO_TEST_CASE( test_futex_volatile_int )
{
    test_futex<volatile int>();
    test_futex_timeout<volatile int>();
}

// use the above helpers for "atomic<int>"
BOOST_AUTO_TEST_CASE( test_futex_atomic_int )
{
    test_futex<atomic<int>>();
    test_futex_timeout<atomic<int>>();
}
