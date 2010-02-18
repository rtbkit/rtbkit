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
struct test2_thread {
    test2_thread(boost::barrier & barrier, X & val, int iter, int tnum)
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

        //barrier.wait();
        
        //cerr << "started thread" << tnum << endl;
        
        for (unsigned i = 0;  i < iter;  ++i)
            atomic_add(val, 1);

        //cerr << "finished thread" << tnum << endl;
    }
};

template<class X>
void test2_type()
{
    cerr << "testing type " << demangle(typeid(X).name()) << endl;
    int nthreads = 8, iter = 1000000;
    boost::barrier barrier(nthreads);
    X val = 0;
    boost::thread_group tg;
    for (unsigned i = 0;  i < nthreads;  ++i)
        tg.create_thread(test2_thread<X>(barrier, val, iter, i));

    tg.join_all();

    BOOST_CHECK_EQUAL(val, (iter * nthreads) & (X)-1);
}

BOOST_AUTO_TEST_CASE( test2 )
{
    test2_type<uint8_t>();
    test2_type<uint16_t>();
    test2_type<uint32_t>();
    test2_type<uint64_t>();
}
