#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <soa/utils/fnv_hash.h>
#include <iostream>

using namespace std;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE( fnv_hash )
{
    BOOST_CHECK_EQUAL(fnv_hash32("abcdef"),2670544664);
    BOOST_CHECK_EQUAL(fnv_hash32a("abcdef"),4282878506);
    BOOST_CHECK_EQUAL(fnv_hash64("abcdef"),2594670854942755800ULL);
    BOOST_CHECK_EQUAL(fnv_hash64a("abcdef"),15567776504244095498ULL);
}
