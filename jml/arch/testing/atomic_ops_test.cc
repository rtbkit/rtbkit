/* atomic_ops_test.cc
   Jeremy Barnes, 21 December 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Test of the bit operations class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/arch/atomic_ops.h"
#include "jml/arch/demangle.h"
#include "jml/arch/exception.h"
#include "jml/arch/tick_counter.h"

#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <stdint.h>
#include <iostream>
#include <stdarg.h>
#include <errno.h>

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;


template<class X>
void test1_type()
{
    cerr << "testing type " << demangle(typeid(X).name()) << endl;
    
    X x = 0;
    BOOST_CHECK_EQUAL(x, 0);
    atomic_add(x, 1);
    BOOST_CHECK_EQUAL(x, 1);
    atomic_add(x, -1);
    BOOST_CHECK_EQUAL(x, 0);
}
 
BOOST_AUTO_TEST_CASE( test1 )
{
    test1_type<int>();
    test1_type<uint8_t>();
    test1_type<int8_t>();
    test1_type<uint16_t>();
    test1_type<int16_t>();
    test1_type<uint32_t>();
    test1_type<int32_t>();
    test1_type<uint64_t>();
    test1_type<int64_t>();
}

template<class X>
struct test_atomic_add2_thread {
    test_atomic_add2_thread(boost::barrier & barrier, X & val, int iter, int tnum)
        : barrier(barrier), val(val), iter(iter), tnum(tnum)
    {
    }

    boost::barrier & barrier;
    X & val;
    int iter;
    int tnum;

    void operator () ()
    {
        //cerr << "thread " << tnum << " waiting" << endl;

        barrier.wait();
        
        //cerr << "started thread" << tnum << endl;
        
        for (unsigned i = 0;  i < iter;  ++i)
            atomic_add(val, 1);

        //cerr << "finished thread" << tnum << endl;
    }
};

template<class X>
void test_atomic_add2_type()
{
    cerr << "testing type " << demangle(typeid(X).name()) << endl;
    int nthreads = 8, iter = 1000000;
    boost::barrier barrier(nthreads);
    X val = 0;
    boost::thread_group tg;

    for (unsigned i = 0;  i < nthreads;  ++i)
        tg.create_thread(test_atomic_add2_thread<X>(barrier, val, iter, i));

    tg.join_all();

    BOOST_CHECK_EQUAL(val, (iter * nthreads) & (X)-1);
}

BOOST_AUTO_TEST_CASE( test_atomic_add2 )
{
    cerr << "atomic add" << endl;
    test_atomic_add2_type<uint8_t>();
    test_atomic_add2_type<uint16_t>();
    test_atomic_add2_type<uint32_t>();
    test_atomic_add2_type<uint64_t>();
}

template<class X>
struct test_atomic_max_thread {
    test_atomic_max_thread(boost::barrier & barrier, X & val, int iter,
                           int tnum, size_t & num_errors)
        : barrier(barrier), val(val), iter(iter), tnum(tnum),
          num_errors(num_errors)
    {
    }

    boost::barrier & barrier;
    X & val;
    int iter;
    int tnum;
    size_t num_errors;

    void operator () ()
    {
        for (unsigned i = 0;  i < iter;  ++i) {
            atomic_max(val, i);
            if (val < i)
                atomic_add(num_errors, 1);
        }
    }
};

template<class X>
void test_atomic_max_type()
{
    cerr << "testing type " << demangle(typeid(X).name()) << endl;
    int nthreads = 8, iter = 1000000;
    X iter2 = (X)-1;
    if (iter2 < iter) iter = iter2;
    boost::barrier barrier(nthreads);
    X val = 0;
    boost::thread_group tg;
    size_t num_errors = 0;
    for (unsigned i = 0;  i < nthreads;  ++i)
        tg.create_thread(test_atomic_max_thread<X>(barrier, val, iter, i,
                                                   num_errors));

    tg.join_all();

    BOOST_CHECK_EQUAL(num_errors, 0);
}

BOOST_AUTO_TEST_CASE( test_atomic_max )
{
    cerr << "atomic max" << endl;
    test_atomic_max_type<uint8_t>();
    test_atomic_max_type<uint16_t>();
    test_atomic_max_type<uint32_t>();
    test_atomic_max_type<uint64_t>();
}

BOOST_AUTO_TEST_CASE( test_atomic_set_bits )
{
    cerr << "atomic set bits" << endl;

    int i = 0;
    BOOST_CHECK_EQUAL(i, 0);
    atomic_set_bits(i, 1);
    BOOST_CHECK_EQUAL(i, 1);
    atomic_clear_bits(i, 1);
    BOOST_CHECK_EQUAL(i, 0);
}

BOOST_AUTO_TEST_CASE( test_atomic_test_and_set )
{
    cerr << "atomic test and set" << endl;

    int i = 0;
    BOOST_CHECK_EQUAL(i, 0);
    BOOST_CHECK_EQUAL(atomic_test_and_set(i, 1), 0);
    BOOST_CHECK_EQUAL(i, 2);
    BOOST_CHECK_EQUAL(atomic_test_and_clear(i, 1), 1);
    BOOST_CHECK_EQUAL(i, 0);
}
