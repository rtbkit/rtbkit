/* mongo_basic_test.cc
   Sunil Rottoo, 2 September 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   Test for our Mongo class.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "soa/service/testing/mongo_temporary_server.h"
#include <boost/test/unit_test.hpp>
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/string_functions.h"
#include "jml/arch/atomic_ops.h"
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/function.hpp>
#include <iostream>
#include "jml/arch/atomic_ops.h"
#include "jml/arch/timers.h"
#include <linux/futex.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "jml/arch/futex.h"

using namespace std;
using namespace ML;

using namespace Mongo;

BOOST_AUTO_TEST_CASE( test_mongo_connection )
{
    MongoTemporaryServer mongo;

    sleep(5.0);
    cerr << "Shutting down the mongo server " << endl;
    mongo.shutdown();
}
