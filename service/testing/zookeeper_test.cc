/* zookeeper_test.cc
   Jeremy Barnes, 2 July 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Test of zookeeper interface.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/pair_utils.h"
#include "jml/utils/string_functions.h"
#include "jml/utils/file_functions.h"
#include "jml/arch/exception_handler.h"
#include "jml/arch/timers.h"
#include "soa/service/zookeeper.h"

#include <sstream>
#include <iostream>
#include <fstream>


using namespace std;
using namespace ML;

using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_zookeeper_cant_connect )
{
    ML::set_default_trace_exceptions(false);

    ZookeeperConnection zk;
    BOOST_CHECK_THROW(zk.connect("localhost:2182", 0.5), std::exception);
}


BOOST_AUTO_TEST_CASE( test_zookeeper )
{
    ZookeeperConnection zk;
    zk.connect("localhost:2181");

    string nodeName = zk.createNode("/hello", "mum", false, true).first;
    cerr << "nodeName = " << nodeName << endl;

    auto children = zk.getChildren("/");
    cerr << "children = " << children << endl;

    cerr << zk.getChildren("/zookeeper/quota");

    BOOST_CHECK_THROW(children = zk.getChildren("/bonus/man"), std::exception);
}
