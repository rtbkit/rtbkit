/* lightweight_hash_test.cc
   Jeremy Barnes, 10 December 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Test program for lightweight hash.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#undef NDEBUG

#include "jml/utils/lightweight_hash.h"
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

BOOST_AUTO_TEST_CASE(test1)
{
    Lightweight_Hash<int, int> h;
    const Lightweight_Hash<int, int> & ch = h;

    BOOST_CHECK_EQUAL(h.empty(), true);
    BOOST_CHECK_EQUAL(h.size(), 0);
    BOOST_CHECK_EQUAL(h.begin(), h.end());
    BOOST_CHECK_EQUAL(ch.begin(), ch.end());
    BOOST_CHECK_EQUAL(h.begin(), ch.end());

    h.reserve(16);
    BOOST_CHECK_EQUAL(h.capacity(), 16);
    BOOST_CHECK_EQUAL(h.size(), 0);
    BOOST_CHECK_EQUAL(h.begin(), h.end());
    BOOST_CHECK_EQUAL(ch.begin(), ch.end());
    BOOST_CHECK_EQUAL(h.begin(), ch.end());

    h[1] = 1;

    BOOST_CHECK_EQUAL(h[1], 1);
    BOOST_CHECK_EQUAL(h.size(), 1);
    BOOST_CHECK(h.begin() != h.end());
    BOOST_CHECK_EQUAL(h.begin()->first, 1);
    BOOST_CHECK_EQUAL(h.begin()->second, 1);
    BOOST_CHECK_EQUAL(boost::next(h.begin()), ch.end());
    BOOST_CHECK_EQUAL(h.begin(), boost::prior(ch.end()));

    h[2] = 2;

    BOOST_CHECK_EQUAL(h[1], 1);
    BOOST_CHECK_EQUAL(h[2], 2);
    BOOST_CHECK_EQUAL(h.size(), 2);
    BOOST_CHECK(h.capacity() >= 2);

    h.reserve(1024);
    BOOST_CHECK_EQUAL(h[1], 1);
    BOOST_CHECK_EQUAL(h[2], 2);
    BOOST_CHECK_EQUAL(h.size(), 2);
    BOOST_CHECK(h.capacity() >= 2);

    BOOST_CHECK_EQUAL(++++h.begin(), h.end());
}

// TODO: use live counting object to check that everything works OK

struct Entry {
    void * p1;
    void * p2;
    void * p3;
    bool val;
};

std::ostream & operator << (std::ostream & stream, Entry entry)
{
    return stream << "Entry";
}

BOOST_AUTO_TEST_CASE(test2)
{
    Lightweight_Hash<void *, Entry> h;
    BOOST_CHECK_THROW(h[(void *)0].p1, ML::Exception);
}

BOOST_AUTO_TEST_CASE(test3)
{
    int nobj = 100;

    vector<void *> objects;
        
    for (unsigned j = 0;  j < nobj;  ++j)
        objects.push_back(malloc(50));

    Lightweight_Hash<void *, Entry> h;
    
    for (unsigned i = 0;  i < nobj;  ++i) {
        h[objects[i]].val = true;
    }

    h.destroy();

    for (unsigned i = 0;  i < nobj;  ++i) {
        BOOST_CHECK(h.find(objects[i]) == h.end());
        h[objects[i]].val = true;
        BOOST_CHECK_EQUAL(h.size(), i + 1);
    }

    BOOST_CHECK_EQUAL(h.size(), nobj);

    for (unsigned j = 0;  j < nobj;  ++j)
        free(objects[j]);
}

BOOST_AUTO_TEST_CASE(test3_log)
{
    int nobj = 100;

    vector<void *> objects;
        
    for (unsigned j = 0;  j < nobj;  ++j)
        objects.push_back(malloc(50));
    
    Lightweight_Hash<void *, Entry,
                     std::pair<void *, Entry>,
                     std::pair<const void *, Entry>,
                     PairOps<void *, Entry>,
                     LogMemStorage<std::pair<void *, Entry> > > h;
    
    for (unsigned i = 0;  i < nobj;  ++i) {
        h[objects[i]].val = true;
    }

    h.destroy();

    for (unsigned i = 0;  i < nobj;  ++i) {
        BOOST_CHECK(h.find(objects[i]) == h.end());
        h[objects[i]].val = true;
        BOOST_CHECK_EQUAL(h.size(), i + 1);
    }

    BOOST_CHECK_EQUAL(h.size(), nobj);

    for (unsigned j = 0;  j < nobj;  ++j)
        free(objects[j]);
}

BOOST_AUTO_TEST_CASE(test_set)
{

    int nobj = 100;

    vector<int> objects;
        
    for (unsigned j = 0;  j < nobj;  ++j)
        objects.push_back(random());

    Lightweight_Hash_Set<int> s;

    BOOST_CHECK_EQUAL(s.size(), 0);

    for (unsigned i = 0;  i < nobj;  ++i) {
        BOOST_CHECK_EQUAL(s.count(objects[i]), 0);
        s.insert(objects[i]);
        BOOST_CHECK_EQUAL(s.size(), i + 1);
        BOOST_CHECK_EQUAL(s.count(objects[i]), 1);
    }

    vector<int> obj2(s.begin(), s.end());
    std::sort(objects.begin(), objects.end());
    std::sort(obj2.begin(), obj2.end());
    
    BOOST_CHECK_EQUAL_COLLECTIONS(objects.begin(), objects.end(),
                                  obj2.begin(), obj2.end());

    cerr << "before copy" << endl;
    s.dump(cerr);

    Lightweight_Hash_Set<int> s2 = s;

    cerr << "after copy" << endl;
    s2.dump(cerr);

    vector<int> obj3(s2.begin(), s2.end());
    std::sort(obj3.begin(), obj3.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(objects.begin(), objects.end(),
                                  obj3.begin(), obj3.end());
}
