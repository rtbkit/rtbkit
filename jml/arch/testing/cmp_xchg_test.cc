/* bitops_test.cc
   Jeremy Barnes, 20 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of the bit operations class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/arch/cmp_xchg.h"
#include "jml/arch/demangle.h"
#include "jml/arch/cpuid.h"
#include "jml/arch/format.h"

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

struct uint128_t {
    uint128_t(uint64_t l = 0, uint64_t h = 0)
        : l(l), h(h)
    {
    }

    uint128_t & operator = (uint64_t val)
    {
        h = 0;  l = val;
        return *this;
    }

    bool operator == (const uint128_t & val) const
    {
        return h == val.h && l == val.l;
    }

    bool operator == (const uint64_t & val) const
    {
        return h == 0 && l == val;
    }

    uint128_t & operator += (const uint128_t & other)
    {
        asm ("add   %[otherl], %[l]\n\t"
             "adc   %[otherh], %[h]\n\t"
             : [l] "+r" (l), [h] "+r" (h)
             : [otherl] "r" (other.l), [otherh] "r" (other.h) : "cc");
        return *this;
    }

    uint128_t & operator &= (const uint128_t & other)
    {
        asm ("and   %[otherl], %[l]\n\t"
             "and   %[otherh], %[h]\n\t"
             : [l] "+r" (l), [h] "+r" (h)
             : [otherl] "r" (other.l), [otherh] "r" (other.h) : "cc");
        return *this;
    }

    template<class X>
    uint128_t operator + (const X & other)
    {
        uint128_t result = *this;
        result += other;
        return result;
    }

#if 0
    uint128_t & operator += (uint64_t other)
    {
        asm ("add   %[other], %[l]\n\t"
             "adc   $0, %[h]\n\t"
             : [l] "+r" (l), [h] "+r" (h)
             : [other] "r" (other) : "cc");
        return *this;
    }
#endif

    uint64_t l, h;
};

template<class X>
uint128_t operator & (const X & x, const uint128_t & val)
{
    uint128_t result = val;
    result &= x;
    return result;
}

std::ostream & operator << (std::ostream & stream, const uint128_t & val)
{
    return stream << format("0x%016llx%016llx",
                            (long long)val.h, (long long)val.l);
}

template<class X>
void test1_type()
{
    cerr << "testing type " << demangle(typeid(X).name()) << endl;
    X x1 = 0, x2 = 1;
    BOOST_CHECK(cmp_xchg(x1, x2, (X)3) == false);
    BOOST_CHECK_EQUAL(x1, 0);
    BOOST_CHECK_EQUAL(x2, 0);
    BOOST_CHECK(cmp_xchg(x1, x2, (X)3) == true);
    BOOST_CHECK_EQUAL(x1, 3);
    BOOST_CHECK_EQUAL(x2, 0);
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

#if (defined(JML_INTEL_ISA) && JML_BITS == 64)
    if (cpu_info().cx16)
        test1_type<uint128_t>();
#endif
}

template<class X>
struct test2_thread {
    test2_thread(boost::barrier & barrier, X & val, int iter)
        : barrier(barrier), val(val), iter(iter)
    {
    }

    boost::barrier & barrier;
    X & val;
    int iter;

    void operator () ()
    {
        barrier.wait();
        
        for (unsigned i = 0;  i < iter;  ++i) {
            X last = val, next;
            do {
                next = last + 1;
            } while (!cmp_xchg(val, last, next));
        }
    }
};

template<class X>
void test2_type()
{
    cerr << "testing type " << demangle(typeid(X).name()) << endl;
    int nthreads = 2, iter = 1000000;
    boost::barrier barrier(nthreads);
    X val = 0;
    boost::thread_group tg;
    for (unsigned i = 0;  i < nthreads;  ++i)
        tg.create_thread(test2_thread<X>(barrier, val, iter));

    tg.join_all();

    //cerr << "val = " << (uint64_t)val << endl;
    //cerr << "leftover = " << ((iter * nthreads) & (X)-1) << endl;

    BOOST_CHECK_EQUAL(val, (iter * nthreads) & (X)-1);

}

BOOST_AUTO_TEST_CASE( test2 )
{
    test2_type<uint8_t>();
    test2_type<uint16_t>();
    test2_type<uint32_t>();
    test2_type<uint64_t>();

#if (defined(JML_INTEL_ISA) && JML_BITS == 64)
    if (cpu_info().cx16)
        test2_type<uint128_t>();
#endif
}
