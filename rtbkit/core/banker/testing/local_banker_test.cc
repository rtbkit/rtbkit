/* master_banker_test.cc
   Wolfgang Sourdeau, 13 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Unit tests for the MasterBanker class
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <memory>
#include <boost/test/unit_test.hpp>
#include "rtbkit/common/account_key.h"
#include "jml/arch/timers.h"

#include "rtbkit/core/banker/local_banker.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_local_banker )
{
    LocalBanker banker(ROUTER);
    banker.init("http://127.0.0.1:27890");
    banker.start();
    AccountKey key({"test", "account"});
    banker.addAccount(key);
    

    ML::sleep(2.0);
    banker.shutdown();
}
