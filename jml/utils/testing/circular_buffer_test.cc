/* circular_buffer_test.cc
   Jeremy Barnes, 7 December 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Test for the circular buffer class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/utils/circular_buffer.h"
#include "jml/utils/string_functions.h"
#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include <boost/tuple/tuple.hpp>
#include "jml/arch/exception_handler.h"
#include "jml/arch/demangle.h"
#include <set>
#include "live_counting_obj.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

void circular_buffer_offset_test(Circular_Buffer<int> & buf)
{
    const Circular_Buffer<int> & cbuf = buf;
    BOOST_CHECK(buf.empty());
    BOOST_CHECK_EQUAL(buf.begin(), buf.end());
    BOOST_CHECK_EQUAL(buf.begin(), cbuf.end());
    BOOST_CHECK_EQUAL(cbuf.begin(), cbuf.end());
    BOOST_CHECK_EQUAL(buf.end() - buf.begin(), 0);
    BOOST_CHECK_EQUAL(buf.end() - cbuf.begin(), 0);
    BOOST_CHECK_EQUAL(buf.begin() - buf.end(), 0);
    BOOST_CHECK_EQUAL(buf.begin() - cbuf.end(), 0);
    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(buf.front(), Exception);
        BOOST_CHECK_THROW(buf.back(), Exception);
        BOOST_CHECK_THROW(buf[0], Exception);
    }

    buf.push_back(1);
    BOOST_CHECK_EQUAL(buf.size(), 1);
    BOOST_CHECK(buf.capacity() >= 1);
    BOOST_CHECK_EQUAL(buf[0], 1);
    BOOST_CHECK_EQUAL(buf[-1], 1);
    BOOST_CHECK_EQUAL(buf.front(), 1);
    BOOST_CHECK_EQUAL(buf.back(), 1);
    BOOST_CHECK_EQUAL(boost::next(buf.begin()), buf.end());
    BOOST_CHECK_EQUAL(boost::next(buf.begin()), cbuf.end());
    BOOST_CHECK_EQUAL(boost::next(cbuf.begin()), cbuf.end());
    BOOST_CHECK_EQUAL(buf.begin(), boost::prior(buf.end()));
    BOOST_CHECK_EQUAL(buf.begin(), boost::prior(cbuf.end()));
    BOOST_CHECK_EQUAL(cbuf.begin(), boost::prior(cbuf.end()));
    BOOST_CHECK_EQUAL(buf.end() - buf.begin(), 1);
    BOOST_CHECK_EQUAL(buf.end() - cbuf.begin(), 1);
    BOOST_CHECK_EQUAL(buf.begin() - buf.end(), -1);
    BOOST_CHECK_EQUAL(buf.begin() - cbuf.end(), -1);
    BOOST_CHECK_EQUAL(cbuf.begin() + 1, buf.end());
    BOOST_CHECK_EQUAL(buf.end() - 1, cbuf.begin());

    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(buf[-2], Exception);
        BOOST_CHECK_THROW(buf[1], Exception);
    }

    buf.reserve(2);
    BOOST_CHECK_EQUAL(buf.size(), 1);
    BOOST_CHECK(buf.capacity() >= 2);
    BOOST_CHECK_EQUAL(buf[0], 1);
    BOOST_CHECK_EQUAL(*buf.begin(), 1);

    buf.push_back(2);
    BOOST_CHECK_EQUAL(buf.size(), 2);
    BOOST_CHECK(buf.capacity() >= 2);

    BOOST_CHECK_EQUAL(buf[0], 1);
    BOOST_CHECK_EQUAL(buf[1], 2);
    BOOST_CHECK_EQUAL(buf[-1], 2);
    BOOST_CHECK_EQUAL(buf[-2], 1);
    BOOST_CHECK_EQUAL(buf.front(), 1);
    BOOST_CHECK_EQUAL(buf.back(), 2);

    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(buf[-3], Exception);
        BOOST_CHECK_THROW(buf[2], Exception);
    }

    buf.push_back(3);
    buf.push_back(4);
    buf.push_back(5);
    buf.push_back(6);

    BOOST_CHECK_EQUAL(buf.size(), 6);
    BOOST_CHECK_EQUAL(buf[0], 1);
    BOOST_CHECK_EQUAL(buf[1], 2);
    BOOST_CHECK_EQUAL(buf[2], 3);
    BOOST_CHECK_EQUAL(buf[3], 4);
    BOOST_CHECK_EQUAL(buf[4], 5);
    BOOST_CHECK_EQUAL(buf[5], 6);
    
    buf.erase_element(1);

    BOOST_CHECK_EQUAL(buf.size(), 5);
    BOOST_CHECK_EQUAL(buf[0], 1);
    BOOST_CHECK_EQUAL(buf[1], 3);
    BOOST_CHECK_EQUAL(buf[2], 4);
    BOOST_CHECK_EQUAL(buf[3], 5);
    BOOST_CHECK_EQUAL(buf[4], 6);
}

BOOST_AUTO_TEST_CASE( circular_buffer_test )
{
    Circular_Buffer<int> buf;

    BOOST_CHECK_EQUAL(buf.capacity(), 0);

    circular_buffer_offset_test(buf);
    Circular_Buffer<int> buf2;
    buf2 = buf;

    BOOST_CHECK_EQUAL(buf2, buf);

    std::copy(buf.begin(), buf.end(), buf2.begin());
    
    BOOST_CHECK_EQUAL(buf2, buf);

    Circular_Buffer<int> buf3;
    buf3.push_back(1);
    buf3.push_back(2);
    buf3.push_back(3);
    buf3.push_back(4);

    BOOST_CHECK_EQUAL(buf3.capacity(), 4);
    BOOST_CHECK_EQUAL(buf3.size(), 4);
    BOOST_CHECK_EQUAL(buf3[0], 1);
    BOOST_CHECK_EQUAL(buf3[1], 2);
    BOOST_CHECK_EQUAL(buf3[2], 3);
    BOOST_CHECK_EQUAL(buf3[3], 4);
    BOOST_CHECK_EQUAL(buf3[-1], 4);
    BOOST_CHECK_EQUAL(buf3[-2], 3);
    BOOST_CHECK_EQUAL(buf3[-3], 2);
    BOOST_CHECK_EQUAL(buf3[-4], 1);

    buf3.pop_front();

    BOOST_CHECK_EQUAL(buf3.size(), 3);
    BOOST_CHECK_EQUAL(buf3.capacity(), 4);
    BOOST_CHECK_EQUAL(buf3[0], 2);
    BOOST_CHECK_EQUAL(buf3[1], 3);
    BOOST_CHECK_EQUAL(buf3[2], 4);
    BOOST_CHECK_EQUAL(buf3[-1], 4);
    BOOST_CHECK_EQUAL(buf3[-2], 3);
    BOOST_CHECK_EQUAL(buf3[-3], 2);
    
    buf3.pop_front();

    BOOST_CHECK_EQUAL(buf3.size(), 2);
    BOOST_CHECK_EQUAL(buf3.capacity(), 4);
    BOOST_CHECK_EQUAL(buf3[0], 3);
    BOOST_CHECK_EQUAL(buf3[1], 4);
    BOOST_CHECK_EQUAL(buf3[-1], 4);
    BOOST_CHECK_EQUAL(buf3[-2], 3);
    
    buf3.push_back(5);

    BOOST_CHECK_EQUAL(buf3.size(), 3);
    BOOST_CHECK_EQUAL(buf3.capacity(), 4);
    BOOST_CHECK_EQUAL(buf3[0], 3);
    BOOST_CHECK_EQUAL(buf3[1], 4);
    BOOST_CHECK_EQUAL(buf3[2], 5);
    BOOST_CHECK_EQUAL(buf3[-1], 5);
    BOOST_CHECK_EQUAL(buf3[-2], 4);
    BOOST_CHECK_EQUAL(buf3[-3], 3);
}

BOOST_AUTO_TEST_CASE( circular_buffer_offset_tests )
{
    for (unsigned sz = 1;  sz < 8;  ++sz) {
        
        cerr << endl << endl << "size " << sz << endl;

        for (unsigned i = 0;  i < sz;  ++i) {
            
            cerr << "start " << i << endl;

            Circular_Buffer<int> buf;
            buf.reserve(sz);
            buf.clear(i);
            
            circular_buffer_offset_test(buf);
        }
    }
}

#if 1

template<class Vector>
void check_basic_ops_type(Vector & vec)
{
    cerr << "checking " << demangle(typeid(Vector).name()) << endl;

    vec.clear();
    BOOST_CHECK_EQUAL(vec.size(), 0);

    vec.push_back(1);
    BOOST_CHECK_EQUAL(vec.size(), 1);
    BOOST_CHECK_EQUAL(vec.front(), 1);
    BOOST_CHECK_EQUAL(vec.back(), 1);

    Vector copy = vec;
    BOOST_CHECK_EQUAL(copy, vec);
    copy.reserve(2);
    BOOST_CHECK_EQUAL(copy, vec);
    
    vec.push_back(2);
    BOOST_CHECK_EQUAL(vec.size(), 2);
    BOOST_CHECK_EQUAL(vec.front(), 1);
    BOOST_CHECK_EQUAL(vec.back(), 2);

    vec.push_back(3);
    BOOST_CHECK_EQUAL(vec.size(), 3);
    BOOST_CHECK_EQUAL(vec.front(), 1);
    BOOST_CHECK_EQUAL(vec.back(), 3);

    vec.push_back(4);
    BOOST_CHECK_EQUAL(vec.size(), 4);
    BOOST_CHECK_EQUAL(vec[0], 1);
    BOOST_CHECK_EQUAL(vec[1], 2);
    BOOST_CHECK_EQUAL(vec[2], 3);
    BOOST_CHECK_EQUAL(vec[3], 4);
    BOOST_CHECK_EQUAL(vec.front(), 1);
    BOOST_CHECK_EQUAL(vec.back(), 4);

    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(vec.at(4), std::exception);
    }

    vec.pop_back();
    BOOST_CHECK_EQUAL(vec.size(), 3);
    BOOST_CHECK_EQUAL(vec.front(), 1);
    BOOST_CHECK_EQUAL(vec.back(), 3);

    vec.pop_back();
    BOOST_CHECK_EQUAL(vec.size(), 2);
    BOOST_CHECK_EQUAL(vec.front(), 1);
    BOOST_CHECK_EQUAL(vec.back(), 2);

    vec.pop_back();
    BOOST_CHECK_EQUAL(vec.size(), 1);
    BOOST_CHECK_EQUAL(vec.front(), 1);
    BOOST_CHECK_EQUAL(vec.back(), 1);

    vec.pop_back();
    BOOST_CHECK_EQUAL(vec.size(), 0);
}

BOOST_AUTO_TEST_CASE( check_basic_ops )
{
    constructed = destroyed = 0;

    Circular_Buffer<int> v1;

    check_basic_ops_type(v1);

#if 0
    Circular_Buffer<int, 1, uint16_t> v2;

    check_basic_ops_type(v2);

    Circular_Buffer<uint16_t, 2, uint16_t> v3;

    check_basic_ops_type(v3);

    Circular_Buffer<uint16_t, 3, uint16_t> v4;

    check_basic_ops_type(v4);

    Circular_Buffer<uint64_t, 4, uint16_t> v5;

    check_basic_ops_type(v5);

    BOOST_CHECK_EQUAL(constructed, destroyed);
#endif

    Circular_Buffer<Obj> v6;

    check_basic_ops_type(v6);

    v6.clear();

    BOOST_CHECK_EQUAL(constructed, destroyed);
}

#if 0
template<class Vector>
void check_insert_erase_type(Vector & vec)
{
    vec.clear();
    BOOST_CHECK_EQUAL(vec.size(), 0);

    vector<int> v1;
    v1.push_back(1);
    v1.push_back(2);
    v1.push_back(3);
    v1.push_back(4);
    v1.push_back(5);

    vector<int> v2;
    v2.push_back(6);
    v2.push_back(7);
    v2.push_back(8);
    v2.push_back(9);
    v2.push_back(10);

    vec.insert(vec.begin(), v1.begin(), v1.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(vec.begin(), vec.end(), v1.begin(), v1.end());

    vec.erase(vec.begin(), vec.end());

    BOOST_CHECK_EQUAL(vec.size(), 0);
    BOOST_CHECK(vec.begin() == vec.end());

    vec.insert(vec.begin(), v1.begin(), v1.end());
    vec.insert(vec.end(),   v2.begin(), v2.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(vec.begin(), vec.begin() + 5,
                                  v1.begin(), v1.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(vec.begin() + 5, vec.end(),
                                  v2.begin(), v2.end());

    vec.clear();

    //cerr << "vec 1 = " << vec << endl;

    vec.insert(vec.begin(), v1.begin(), v1.end());

    //cerr << "vec 2 = " << vec << endl;

    vec.insert(vec.begin() + 2, v2.begin(), v2.end());

    //cerr << "vec 3 = " << vec << endl;

    BOOST_CHECK_EQUAL(*vec.erase(vec.begin() + 7), 4);

    BOOST_CHECK_EQUAL(vec[0], 1);
    BOOST_CHECK_EQUAL(vec[1], 2);
    BOOST_CHECK_EQUAL(vec[2], 6);
    BOOST_CHECK_EQUAL(vec[3], 7);
    BOOST_CHECK_EQUAL(vec[4], 8);
    BOOST_CHECK_EQUAL(vec[5], 9);
    BOOST_CHECK_EQUAL(vec[6], 10);
    BOOST_CHECK_EQUAL(vec[7], 4);
    BOOST_CHECK_EQUAL(vec[8], 5);
}

BOOST_AUTO_TEST_CASE( check_insert_erase )
{
    constructed = destroyed = 0;

    Circular_Buffer<int> v1;

    check_insert_erase_type(v1);

#if 0
    Circular_Buffer<int, 1, uint16_t> v2;

    check_insert_erase_type(v2);

    Circular_Buffer<uint16_t, 2, uint16_t> v3;

    check_insert_erase_type(v3);

    Circular_Buffer<uint16_t, 3, uint16_t> v4;

    check_insert_erase_type(v4);

    Circular_Buffer<uint64_t, 4, uint16_t> v5;

    check_insert_erase_type(v5);
#endif

    BOOST_REQUIRE_EQUAL(constructed, destroyed);

    Circular_Buffer<Obj> v6;

    check_insert_erase_type(v6);

    v6.clear();

    BOOST_CHECK_EQUAL(constructed, destroyed);
}
#endif

BOOST_AUTO_TEST_CASE( check_swap_finishes )
{
    constructed = destroyed = 0;

    Circular_Buffer<Obj> v1, v2;
    v1.push_back(1);
    v2.push_back(2);

    v1.swap(v2);

    BOOST_CHECK(v1.size() == 1);
    BOOST_CHECK(v2.size() == 1);
    BOOST_CHECK(v1[0] == 2);
    BOOST_CHECK(v2[0] == 1);

    v1.clear();  v2.clear();

    BOOST_CHECK_EQUAL(constructed, destroyed);
}

BOOST_AUTO_TEST_CASE( check_swap_bounds )
{
    constructed = destroyed = 0;

    Circular_Buffer<Obj> v1, v2;
    v1.push_back(1);
    v1.push_back(2);
    v2.push_back(3);

    BOOST_CHECK_EQUAL(v1.size(), 2);
    BOOST_CHECK_EQUAL(v2.size(), 1);
    BOOST_CHECK_EQUAL(v1[0],     1);
    BOOST_CHECK_EQUAL(v1[1],     2);
    BOOST_CHECK_EQUAL(v2[0],     3);

    v1.swap(v2);

    BOOST_CHECK_EQUAL(v1.size(), 1);
    BOOST_CHECK_EQUAL(v2.size(), 2);
    BOOST_CHECK_EQUAL(v1[0],     3);
    BOOST_CHECK_EQUAL(v2[0],     1);
    BOOST_CHECK_EQUAL(v2[1],     2);

    v2.swap(v1);

    BOOST_CHECK_EQUAL(v1.size(), 2);
    BOOST_CHECK_EQUAL(v2.size(), 1);
    BOOST_CHECK_EQUAL(v1[0],     1);
    BOOST_CHECK_EQUAL(v1[1],     2);
    BOOST_CHECK_EQUAL(v2[0],     3);

    v1.clear();  v2.clear();

    BOOST_CHECK_EQUAL(constructed, destroyed);
}

BOOST_AUTO_TEST_CASE( check_reserve )
{
    constructed = destroyed = 0;

    Circular_Buffer<Obj> v;
    v.resize(3);
    v.reserve(4);
    v.clear();

    BOOST_CHECK_EQUAL(constructed, destroyed);
}

BOOST_AUTO_TEST_CASE( check_resize )
{
    constructed = destroyed = 0;

    Circular_Buffer<Obj> v;

    v.resize(0);

    BOOST_CHECK_EQUAL(v.size(), 0);

    v.resize(1);
    v[0] = 10;

    BOOST_CHECK_EQUAL(v.size(), 1);
    BOOST_CHECK_EQUAL(v[0], 10);

    v.resize(2);
    v[1] = 20;

    BOOST_CHECK_EQUAL(v.size(), 2);
    BOOST_CHECK_EQUAL(v[0], 10);
    BOOST_CHECK_EQUAL(v[1], 20);

    v.resize(3);
    v[2] = 30;

    BOOST_CHECK_EQUAL(v.size(), 3);
    BOOST_CHECK_EQUAL(v[0], 10);
    BOOST_CHECK_EQUAL(v[1], 20);
    BOOST_CHECK_EQUAL(v[2], 30);

    v.resize(4);
    v[3] = 40;

    BOOST_CHECK_EQUAL(v.size(), 4);
    BOOST_CHECK_EQUAL(v[0], 10);
    BOOST_CHECK_EQUAL(v[1], 20);
    BOOST_CHECK_EQUAL(v[2], 30);
    BOOST_CHECK_EQUAL(v[3], 40);

    v.resize(3);

    BOOST_CHECK_EQUAL(v.size(), 3);
    BOOST_CHECK_EQUAL(v[0], 10);
    BOOST_CHECK_EQUAL(v[1], 20);
    BOOST_CHECK_EQUAL(v[2], 30);

    v.resize(2);

    BOOST_CHECK_EQUAL(v.size(), 2);
    BOOST_CHECK_EQUAL(v[0], 10);
    BOOST_CHECK_EQUAL(v[1], 20);

    v.resize(1);

    BOOST_CHECK_EQUAL(v.size(), 1);
    BOOST_CHECK_EQUAL(v[0], 10);

    v.resize(0);

    BOOST_CHECK_EQUAL(v.size(), 0);

    BOOST_CHECK_EQUAL(constructed, destroyed);
}
#endif

BOOST_AUTO_TEST_CASE(check_delete_element)
{
}
