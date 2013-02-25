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
#include "soa/service/testing/zookeeper_temporary_server.h"

#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;
using namespace ML;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_zookeeper )
{
    ML::set_default_trace_exceptions(false);

    ZooKeeper::TemporaryServer server;
    std::string uri = ML::format("localhost:%d", server.getPort());

    std::cerr << "trying to connect to port " << server.getPort() << std::endl;
    {
        ZookeeperConnection zk;
        BOOST_CHECK_THROW(zk.connect(uri, 0.5), std::exception);
    }

    std::cerr << "starting zookeeper..." << std::endl;
    server.start();

    std::cerr << "trying to connect to port " << server.getPort() << std::endl;
    ZookeeperConnection zk;
    zk.connect(uri);

    std::string nodeName = zk.createNode("/hello", "mum", false, true).first;
    std::cerr << "nodeName = " << nodeName << endl;

    auto children = zk.getChildren("/");
    std::cerr << "children = " << children << endl;

    std::cerr << zk.getChildren("/zookeeper/quota");

    BOOST_CHECK_THROW(children = zk.getChildren("/bonus/man"), std::exception);
}

