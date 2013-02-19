/* pending_list_test.cc
   Jeremy Barnes, 28 February 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Test for the pending list.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "jml/arch/format.h"
#include "soa/service/pending_list.h"
#include "soa/types/id.h"
#include "jml/utils/pair_utils.h"


using namespace std;
using namespace ML;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_router_init_persistence )
{
    struct Value {
        Value(int i = 0)
            : i(i)
        {
        }

        int i;
    };

    PendingList<pair<Id, Id>, Value> pending;

    IsPrefixPair isPrefix;

    pair<Id, Id> none;


    auto h  = make_pair(Id("0"), Id());

    auto i  = make_pair(Id("1"), Id());
    auto i1 = make_pair(Id("1"), Id("1"));
    auto i2 = make_pair(Id("1"), Id("2"));

    auto j  = make_pair(Id("2"), Id());

    auto k  = make_pair(Id("3"), Id());
    auto k0 = make_pair(Id("3"), Id("0"));
    auto k1 = make_pair(Id("3"), Id("1"));
    auto k2 = make_pair(Id("3"), Id("2"));

    auto l  = make_pair(Id("4"), Id());

    auto m  = make_pair(Id("5"), Id());
    auto m1 = make_pair(Id("5"), Id("1"));
    auto m2 = make_pair(Id("5"), Id("2"));

    auto o  = make_pair(Id("6"), Id());

    BOOST_CHECK_LT(i, k);
    BOOST_CHECK_LT(k, m);
    BOOST_CHECK_LT(i, m);
    BOOST_CHECK_LT(i, i1);
    BOOST_CHECK_LT(i, i2);

    BOOST_CHECK_EQUAL(pending.completePrefix(k, isPrefix), none);

    pending.insert(k1, 1, Date());
    BOOST_CHECK_EQUAL(pending.completePrefix(k, isPrefix), k1);
    pending.insert(k2, 2, Date());
    BOOST_CHECK_EQUAL(pending.completePrefix(k, isPrefix), k1);
    pending.insert(k0, 0, Date());
    BOOST_CHECK_EQUAL(pending.completePrefix(k, isPrefix), k0);
    BOOST_CHECK_EQUAL(pending.completePrefix(i, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(m, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(h, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(j, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(l, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(o, isPrefix), none);

    pending.insert(m1, 1, Date());
    BOOST_CHECK_EQUAL(pending.completePrefix(k, isPrefix), k0);
    BOOST_CHECK_EQUAL(pending.completePrefix(i, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(m, isPrefix), m1);
    BOOST_CHECK_EQUAL(pending.completePrefix(h, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(j, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(l, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(o, isPrefix), none);

    pending.insert(m2, 2, Date());
    BOOST_CHECK_EQUAL(pending.completePrefix(k, isPrefix), k0);
    BOOST_CHECK_EQUAL(pending.completePrefix(i, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(m, isPrefix), m1);
    BOOST_CHECK_EQUAL(pending.completePrefix(h, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(j, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(l, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(o, isPrefix), none);

    pending.insert(i2, 2, Date());
    BOOST_CHECK_EQUAL(pending.completePrefix(k, isPrefix), k0);
    BOOST_CHECK_EQUAL(pending.completePrefix(i, isPrefix), i2);
    BOOST_CHECK_EQUAL(pending.completePrefix(m, isPrefix), m1);
    BOOST_CHECK_EQUAL(pending.completePrefix(h, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(j, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(l, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(o, isPrefix), none);

    pending.insert(i1, 1, Date());
    BOOST_CHECK_EQUAL(pending.completePrefix(k, isPrefix), k0);
    BOOST_CHECK_EQUAL(pending.completePrefix(i, isPrefix), i1);
    BOOST_CHECK_EQUAL(pending.completePrefix(m, isPrefix), m1);
    BOOST_CHECK_EQUAL(pending.completePrefix(h, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(j, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(l, isPrefix), none);
    BOOST_CHECK_EQUAL(pending.completePrefix(o, isPrefix), none);
}

