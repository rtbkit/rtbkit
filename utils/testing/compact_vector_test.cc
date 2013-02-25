/* compact_vector_test.cc
   Jeremy Barnes, 3 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Test for the compact_vector class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK


#include "jml/utils/filter_streams.h"
#include "jml/utils/compact_vector.h"
#include "jml/arch/exception.h"
#include "jml/arch/exception_handler.h"
#include "jml/arch/arch.h"

#include <boost/test/unit_test.hpp>
#include "jml/utils/vector_utils.h"
#include "jml/utils/pair_utils.h"
#include "jml/utils/file_functions.h"
#include "jml/utils/string_functions.h"
#include <iostream>
#include <sstream>



using namespace std;
using namespace ML;

using boost::unit_test_framework::test_suite;

size_t constructed = 0, copied = 0, moved = 0, destroyed = 0;

int GOOD = 0xfeedbac4;
int BAD  = 0xdeadbeef;

struct Obj {
    Obj()
        : val(0)
    {
        //cerr << "default construct at " << this << endl;
        ++constructed;
        magic = GOOD;
    }

    Obj(int val)
        : val(val)
    {
        //cerr << "value construct at " << this << endl;
        ++constructed;
        magic = GOOD;
    }

   ~Obj()
    {
        //cerr << "destroying at " << this << endl;
        ++destroyed;
        if (magic == BAD)
            throw Exception("object destroyed twice");

        if (magic != GOOD)
            throw Exception("object never initialized in destructor");

        magic = BAD;
    }

    Obj(const Obj & other)
        : val(other.val)
    {
        //cerr << "copy construct at " << this << endl;
        ++constructed;
        ++copied;
        magic = GOOD;
    }

    Obj & operator = (Obj & other)
    {
        // cerr << "copy assign at " << this << endl;

        if (magic == BAD)
            throw Exception("assigned to destroyed object");

        if (magic != GOOD)
            throw Exception("assigned to object never initialized in assign");

        val = other.val;
        ++copied;

        return *this;
    }


    Obj(Obj && other) :
        val(std::move(other.val))
    {
        // cerr << "move construct at " << this << endl;
        ++constructed;
        ++moved;
        magic = GOOD;
    }

    Obj & operator=(Obj && other)
    {
        // cerr << "move assign at " << this << endl;

        if (magic == BAD)
            throw Exception("assigned to destroyed object");

        if (magic != GOOD)
            throw Exception("assigned to object never initialized in assign");

        val = std::move(other.val);
        ++moved;

        return *this;
    }

    int val;
    int magic;

    operator int () const
    {
        if (magic == BAD)
            throw Exception("read destroyed object");

        if (magic != GOOD)
            throw Exception("read from uninitialized object");

        return val;
    }
};

BOOST_AUTO_TEST_CASE( check_sizes )
{
    compact_vector<int, 1, uint16_t> vec1;
#if (JML_BITS == 32)
    BOOST_CHECK_EQUAL(sizeof(vec1), 8);
#else
    BOOST_CHECK_EQUAL(sizeof(vec1), 12);
#endif

    compact_vector<uint16_t, 3, uint16_t> vec2;
#if (JML_BITS == 32)
    BOOST_CHECK_EQUAL(sizeof(vec2), 8);
#else
    BOOST_CHECK_EQUAL(sizeof(vec2), 12);
#endif

    compact_vector<uint16_t, 5, uint16_t> vec3;
    BOOST_CHECK_EQUAL(sizeof(vec3), 12);
}

template<class Vector>
void check_basic_ops_type(Vector & vec)
{
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
        Set_Trace_Exceptions guard(false);
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

    compact_vector<int> v1;

    check_basic_ops_type(v1);

    compact_vector<int, 1, uint16_t> v2;

    check_basic_ops_type(v2);

    compact_vector<uint16_t, 2, uint16_t> v3;

    check_basic_ops_type(v3);

    compact_vector<uint16_t, 3, uint16_t> v4;

    check_basic_ops_type(v4);

    compact_vector<uint64_t, 4, uint16_t> v5;

    check_basic_ops_type(v5);

    BOOST_CHECK_EQUAL(constructed, destroyed);

    compact_vector<Obj, 4, uint16_t> v6;

    check_basic_ops_type(v6);

    v6.clear();

    BOOST_CHECK_EQUAL(constructed, destroyed);
}

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

    auto it1 = vec.insert(vec.begin(), v1.begin(), v1.end());
    BOOST_CHECK(it1 == vec.begin());

    BOOST_CHECK_EQUAL_COLLECTIONS(vec.begin(), vec.end(), v1.begin(), v1.end());

    vec.erase(vec.begin(), vec.end());

    BOOST_CHECK_EQUAL(vec.size(), 0);
    BOOST_CHECK(vec.begin() == vec.end());

    auto it2 = vec.insert(vec.begin(), v1.begin(), v1.end());
    BOOST_CHECK(it2 == vec.begin());

    auto it3 = vec.insert(vec.end(),   v2.begin(), v2.end());
    BOOST_CHECK(it3 == vec.begin() + v1.size());

    BOOST_CHECK_EQUAL_COLLECTIONS(vec.begin(), vec.begin() + 5,
                                  v1.begin(), v1.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(vec.begin() + 5, vec.end(),
                                  v2.begin(), v2.end());

    vec.clear();

    //cerr << "vec 1 = " << vec << endl;

    vec.insert(vec.begin(), v1.begin(), v1.end());

    //cerr << "vec 2 = " << vec << endl;

    auto it4 = vec.insert(vec.begin() + 2, v2.begin(), v2.end());
    BOOST_CHECK(it4 == vec.begin() + 2);

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

    compact_vector<int> v1;

    check_insert_erase_type(v1);

    compact_vector<int, 1, uint16_t> v2;

    check_insert_erase_type(v2);

    compact_vector<uint16_t, 2, uint16_t> v3;

    check_insert_erase_type(v3);

    compact_vector<uint16_t, 3, uint16_t> v4;

    check_insert_erase_type(v4);

    compact_vector<uint64_t, 4, uint16_t> v5;

    check_insert_erase_type(v5);

    BOOST_CHECK_EQUAL(constructed, destroyed);

    compact_vector<Obj, 4, uint16_t> v6;

    check_insert_erase_type(v6);

    v6.clear();

    BOOST_CHECK_EQUAL(constructed, destroyed);
}

BOOST_AUTO_TEST_CASE( check_swap_finishes )
{
    constructed = destroyed = 0;

    compact_vector<Obj, 1, uint16_t> v1, v2;
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

    compact_vector<Obj, 2, uint16_t> v1, v2;
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

    compact_vector<Obj, 3, unsigned> v;
    v.resize(3);
    v.reserve(4);
    v.clear();

    BOOST_CHECK_EQUAL(constructed, destroyed);
}

BOOST_AUTO_TEST_CASE( check_resize )
{
    constructed = destroyed = 0;

    compact_vector<Obj, 3, unsigned> v;

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

BOOST_AUTO_TEST_CASE( check_c11_ops )
{

    auto reset = [] { moved = copied = 0; };

    {
        compact_vector<Obj, 6, unsigned> v;

        reset();
        v.emplace_back(10);
        BOOST_CHECK_EQUAL(moved, 0);
        BOOST_CHECK_EQUAL(copied, 0);
        BOOST_CHECK_EQUAL(v.size(), 1);
        BOOST_CHECK_EQUAL(v[0], 10);

        reset();
        v.emplace_back(Obj(20));
        BOOST_CHECK_EQUAL(moved, 1);
        BOOST_CHECK_EQUAL(copied, 0);
        BOOST_CHECK_EQUAL(v.size(), 2);
        BOOST_CHECK_EQUAL(v[0], 10);
        BOOST_CHECK_EQUAL(v[1], 20);

        reset();
        Obj o1(30);
        v.emplace_back(o1);
        BOOST_CHECK_EQUAL(moved, 0);
        BOOST_CHECK_EQUAL(copied, 1);
        BOOST_CHECK_EQUAL(v.size(), 3);
        BOOST_CHECK_EQUAL(v[0], 10);
        BOOST_CHECK_EQUAL(v[1], 20);
        BOOST_CHECK_EQUAL(v[2], 30);

        reset();
        v.emplace(v.begin() + 1, 40);
        BOOST_CHECK_GT(moved, 0);
        BOOST_CHECK_EQUAL(copied, 0);
        BOOST_CHECK_EQUAL(v.size(), 4);
        BOOST_CHECK_EQUAL(v[0], 10);
        BOOST_CHECK_EQUAL(v[1], 40);
        BOOST_CHECK_EQUAL(v[2], 20);
        BOOST_CHECK_EQUAL(v[3], 30);

        reset();
        v.push_back(Obj(50));
        BOOST_CHECK_EQUAL(moved, 1);
        BOOST_CHECK_EQUAL(copied, 0);
        BOOST_CHECK_EQUAL(v.size(), 5);
        BOOST_CHECK_EQUAL(v[0], 10);
        BOOST_CHECK_EQUAL(v[1], 40);
        BOOST_CHECK_EQUAL(v[2], 20);
        BOOST_CHECK_EQUAL(v[3], 30);
        BOOST_CHECK_EQUAL(v[4], 50);

        reset();
        Obj obj(60);
        v.push_back(obj);
        BOOST_CHECK_EQUAL(moved, 0);
        BOOST_CHECK_EQUAL(copied, 1);
        BOOST_CHECK_EQUAL(v.size(), 6);
        BOOST_CHECK_EQUAL(v[0], 10);
        BOOST_CHECK_EQUAL(v[1], 40);
        BOOST_CHECK_EQUAL(v[2], 20);
        BOOST_CHECK_EQUAL(v[3], 30);
        BOOST_CHECK_EQUAL(v[4], 50);
        BOOST_CHECK_EQUAL(v[5], 60);
    }

    BOOST_CHECK_EQUAL(constructed, destroyed);

    {
        // Check that on move assignment that the moved-from version
        // is correctly resized
        compact_vector<Obj, 3, unsigned> v;
        v.push_back(1);
        v.push_back(2);

        compact_vector<Obj, 3, unsigned> u(std::move(v));

        BOOST_CHECK_EQUAL(v.size(), 0);
        BOOST_CHECK_EQUAL(u.size(), 2);
    }

    BOOST_CHECK_EQUAL(constructed, destroyed);
}
