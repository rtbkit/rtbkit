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

#include <iostream>

using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_zookeeper )
{
    ML::set_default_trace_exceptions(false);

    ZooKeeper::TemporaryServer server;
    std::string uri = ML::format("localhost:%d", server.getPort());

    std::vector<int> pids;

    int n = 100;
    for(int i = 0; i != n; ++i) {
        int pid = fork();
        if(pid == -1) {
            throw ML::Exception(errno, "fork");
        }

        if(pid == 0) {
            ML::sleep(1);
            std::cerr << getpid() << " trying to connect to " << uri << std::endl;
            ZookeeperConnection zk;
            zk.connect(uri);
            for(;;) {
                auto node = zk.readNode("/hello");
                std::cerr << getpid() << " node=" << node << std::endl;
                if(node == "world") {
                    break;
                }

                ML::sleep(1);
            }

            zk.createNode(ML::format("/%d", getpid()), "hello", true, false);
            for(;;) {
                ML::sleep(1);
            }
        }
        else {
            pids.push_back(pid);
        }
    }

    std::cerr << "starting zookeeper..." << std::endl;
    server.start();

    ZookeeperConnection zk;
    zk.connect(uri);

    auto node = zk.createNode("/hello", "world", true, false);
    std::cerr << "nodeName = " << node.first << std::endl;

    signal(SIGCHLD, SIG_DFL);

    for(;;) {
        auto children = zk.getChildren("/");
        std::cerr << "children = " << children << std::endl;

        if(children.size() == 2 + n) {
            break;
        }

        ML::sleep(1);
    }

    for(int i = 0; i != n; ++i) {
        int pid = pids[i];
        int res = kill(pid, SIGTERM);
        if (res == -1) {
            throw ML::Exception(errno, "cannot kill child process");
        }
    }

    for(int i = 0; i != n; ++i) {
        int pid = pids[i];
        int status = 0;
        int res = waitpid(pid, &status, 0);
        if (res == -1) {
            throw ML::Exception(errno, "failed to wait for child process to shutdown");
        }
    }
}

