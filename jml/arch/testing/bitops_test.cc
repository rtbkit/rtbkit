/* bitops_test.cc
   Jeremy Barnes, 20 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of the bit operations class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/arch/bitops.h"
#include "jml/arch/tick_counter.h"
#include "jml/arch/demangle.h"
#include "jml/math/xdiv.h"

#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <vector>
#include <stdint.h>
#include <iostream>


using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

template<class X>
void test1_type()
{
    cerr << "testing type " << demangle(typeid(X).name()) << endl;

    BOOST_CHECK_EQUAL(highest_bit((X)0),       -1);
    BOOST_CHECK_EQUAL(highest_bit((X)1),        0);
    BOOST_CHECK_EQUAL(highest_bit((X)2),        1);
    BOOST_CHECK_EQUAL(highest_bit((X)3),        1);
    BOOST_CHECK_EQUAL(highest_bit((X)-1),       sizeof(X) * 8 - 1);
    BOOST_CHECK_EQUAL(highest_bit((X)63),       5);
    BOOST_CHECK_EQUAL(highest_bit((X)64),       6);
    BOOST_CHECK_EQUAL(highest_bit((X)255),      7);

    BOOST_CHECK_EQUAL(lowest_bit((X)0),       -1);
    BOOST_CHECK_EQUAL(lowest_bit((X)1),        0);
    BOOST_CHECK_EQUAL(lowest_bit((X)2),        1);
    BOOST_CHECK_EQUAL(lowest_bit((X)3),        0);
    BOOST_CHECK_EQUAL(lowest_bit((X)-1),       0);
    BOOST_CHECK_EQUAL(lowest_bit((X)63),       0);
    BOOST_CHECK_EQUAL(lowest_bit((X)64),       6);
    BOOST_CHECK_EQUAL(lowest_bit((X)255),      0);
    BOOST_CHECK_EQUAL(lowest_bit((X)128),      7);

    if (sizeof(X) == 1) return;
    BOOST_CHECK_EQUAL(highest_bit((X)256),      8);
    BOOST_CHECK_EQUAL(lowest_bit((X)256),      8);

    if (sizeof(X) == 2) return;
    BOOST_CHECK_EQUAL(highest_bit((X)131071),  16);
    BOOST_CHECK_EQUAL(highest_bit((X)131072),  17);
    BOOST_CHECK_EQUAL(highest_bit((X)131073),  17);

    BOOST_CHECK_EQUAL(lowest_bit((X)131071),  0);
    BOOST_CHECK_EQUAL(lowest_bit((X)131072),  17);
    BOOST_CHECK_EQUAL(lowest_bit((X)131073),  0);
}

BOOST_AUTO_TEST_CASE( test1 )
{
    test1_type<uint8_t>();
    test1_type<int8_t>();
    test1_type<uint16_t>();
    test1_type<int16_t>();
    test1_type<uint32_t>();
    test1_type<int32_t>();
    test1_type<uint64_t>();
    test1_type<int64_t>();
}

uint64_t rand64()
{
    uint64_t high = rand(), low = rand();
    return (high << 32) | low;
}

template<class X>
int fake_highest_bit(X x)
{
    return x;
}

template<class X>
void profile_type(const vector<uint64_t> & vals_)
{
    vector<X> vals(vals_.begin(), vals_.end());

    cerr << "profiling for " << sizeof(X) * 8 << " bits, "
         << ((X)-1 > 0 ? "unsigned" : "signed")
         <<  " with " << vals.size() << " vals" << endl;

    /* Measure the overhead */
    double overhead = 0;

    int trials = 10;

    int total = 0;

    for (unsigned t = 0;  t < trials;  ++t) {
        uint64_t tbefore = ticks();
        for (unsigned i = 0;  i < vals.size();  ++i)
            total += fake_highest_bit(vals[i]);
        uint64_t tafter = ticks();
        overhead += tafter - tbefore - ticks_overhead;

    }

    cerr << "measurement overhead is " << overhead << endl;

    double measured = 0;

    for (unsigned t = 0;  t < trials;  ++t) {

        uint64_t tbefore = ticks();
        for (unsigned i = 0;  i < vals.size();  ++i)
            total += highest_bit(vals[i]);
        
        uint64_t tafter = ticks();
        
        measured += tafter - tbefore - ticks_overhead;
    }

    srand(total);  // use it so it doesn't get optimized out

    cerr << "measured " << measured << " total ticks" << endl;

    double cost = xdiv<double>(measured - overhead, trials * vals.size());

    cerr << "cost: " << cost << " ticks/call" << endl;
}
    
BOOST_AUTO_TEST_CASE( profile )
{
    vector<uint64_t> vals;
    for (unsigned i = 0;  i < 1000;  ++i)
        vals.push_back(rand64());

    profile_type<uint8_t>(vals);
    profile_type<int8_t>(vals);
    profile_type<uint16_t>(vals);
    profile_type<int16_t>(vals);
    profile_type<uint32_t>(vals);
    profile_type<int32_t>(vals);
    profile_type<uint64_t>(vals);
    profile_type<int64_t>(vals);
}

BOOST_AUTO_TEST_CASE(test_rotate)
{
    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(0, 0), 0);
    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(0, 1), 0);
    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(0, 8), 0);

    BOOST_CHECK_EQUAL(rotate_left<uint8_t>(0, 0), 0);
    BOOST_CHECK_EQUAL(rotate_left<uint8_t>(0, 1), 0);
    BOOST_CHECK_EQUAL(rotate_left<uint8_t>(0, 8), 0);

    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(1, 0), 1);
    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(1, 1), 128);
    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(1, 7), 2);
    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(1, 8), 1);
    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(1, 16), 1);
    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(1, 17), 128);

    BOOST_CHECK_EQUAL(rotate_left<uint8_t>(1, 0), 1);
    BOOST_CHECK_EQUAL(rotate_left<uint8_t>(1, 1), 2);
    BOOST_CHECK_EQUAL(rotate_left<uint8_t>(1, 7), 128);
    BOOST_CHECK_EQUAL(rotate_left<uint8_t>(1, 8), 1);
    BOOST_CHECK_EQUAL(rotate_left<uint8_t>(1, 16), 1);
    BOOST_CHECK_EQUAL(rotate_left<uint8_t>(1, 17), 2);

    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(0, 0), 0);
    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(0, 1), 0);
    BOOST_CHECK_EQUAL(rotate_right<uint8_t>(0, 8), 0);

    BOOST_CHECK_EQUAL(rotate_left<int8_t>(0, 0), 0);
    BOOST_CHECK_EQUAL(rotate_left<int8_t>(0, 1), 0);
    BOOST_CHECK_EQUAL(rotate_left<int8_t>(0, 8), 0);

    BOOST_CHECK_EQUAL(rotate_right<int8_t>(1, 0), 1);
    BOOST_CHECK_EQUAL(rotate_right<int8_t>(1, 1), -128);
    BOOST_CHECK_EQUAL(rotate_right<int8_t>(1, 7), 2);
    BOOST_CHECK_EQUAL(rotate_right<int8_t>(1, 8), 1);
    BOOST_CHECK_EQUAL(rotate_right<int8_t>(1, 16), 1);
    BOOST_CHECK_EQUAL(rotate_right<int8_t>(1, 17), -128);

    BOOST_CHECK_EQUAL(rotate_left<int8_t>(1, 0), 1);
    BOOST_CHECK_EQUAL(rotate_left<int8_t>(1, 1), 2);
    BOOST_CHECK_EQUAL(rotate_left<int8_t>(1, 7), -128);
    BOOST_CHECK_EQUAL(rotate_left<int8_t>(1, 8), 1);
    BOOST_CHECK_EQUAL(rotate_left<int8_t>(1, 16), 1);
    BOOST_CHECK_EQUAL(rotate_left<int8_t>(1, 17), 2);
}
