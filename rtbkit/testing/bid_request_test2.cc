/* bid_request_test.cc
   Sunil Rottoo, 13 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Acceptance test for the bid request
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "rtbkit/common/bid_request.h"


using namespace std;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_user_id )
{
    UserIds uid;
    uid.add(Id("uid1"), "datacratic");
    BOOST_CHECK_EQUAL(uid.size(),1);
    BOOST_CHECK_EQUAL(uid.count("datacratic"),1);
    // make sure we cannot add a duplicate
    BOOST_CHECK_THROW(uid.add(Id("uid1"), "datacratic"), ML::Exception);
    // make sure we add multiple ids for the same domain
    uid.add(Id("uid2"), "datacratic");
    BOOST_CHECK_EQUAL(uid.size(), 2);
    // make sure that set will erase previous values
    uid.set(Id("uid1"), "datacratic");
    BOOST_CHECK_EQUAL(uid.size(), 1);
    BOOST_CHECK_EQUAL(uid[0].second, Id("uid1"));
    BOOST_CHECK_EQUAL(uid[0].first, "datacratic");
    uid.add(Id("uid3"), "datacratic");
    BOOST_CHECK_EQUAL(uid.size(),2);
    uid.add(Id("uid2"), "adx");
    BOOST_CHECK_EQUAL(uid.size(), 3);
#if 0
    cerr << "The contents of uids " << endl;
    for ( auto iddom: uid)
    {
        cerr << iddom.first << "," << iddom.second << endl;
    }
#endif
    auto domainMatch = [&](const pair<string,Id> &elt, const std::string &dom)
    {
        return elt.first == dom ;
    };
    Id lastUid ;
    vector<string> allDoms = {"datacratic"};
    auto found = search(uid.begin(), uid.end(), allDoms.begin(), allDoms.end(),
                        domainMatch);
    while ( found != uid.end())
    {
        lastUid = found->second;
        //cerr << "matching domain " << found->first << " with id " << found->second << endl;
        found = search(++found, uid.end(), allDoms.begin(), allDoms.end(),
                       domainMatch);
    }
    BOOST_CHECK_EQUAL(lastUid, Id("uid3"));
    // make sure we take the last one
    unsigned numErased = uid.eraseDomain("datacratic");
    BOOST_CHECK_EQUAL(numErased, 2);
    BOOST_CHECK_EQUAL(uid.size(), 1);
    // make sure we get zero if not present
    numErased = uid.eraseDomain("datacratic");
    BOOST_CHECK_EQUAL(numErased, 0);
}
