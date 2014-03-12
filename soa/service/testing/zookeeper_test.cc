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

#include <thread>
#include <iostream>
#include <set>
#include <sys/prctl.h>

using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_zookeeper_connection )
{
    ML::set_default_trace_exceptions(false);

    ZooKeeper::TemporaryServer server;
    std::string uri = ML::format("localhost:%d", server.getPort());

    // avoid aborting test when killing a child process
    signal(SIGCHLD, SIG_DFL);

    std::thread client([=] {
        ZookeeperConnection zk;
        std::cerr << "starting client..." << std::endl;
        zk.connect(uri, 1.0);
    });

    ML::sleep(5.0);

    std::cerr << "starting zookeeper..." << std::endl;
    server.start();
    client.join();
}

BOOST_AUTO_TEST_CASE( test_zookeeper_crash )
{
    ML::set_default_trace_exceptions(false);

    ZooKeeper::TemporaryServer server;
    std::string uri = ML::format("localhost:%d", server.getPort());

    // avoid aborting test when killing a child process
    signal(SIGCHLD, SIG_DFL);

    std::cerr << "starting zookeeper..." << std::endl;
    server.start();

    std::thread client([=] {
        ZookeeperConnection zk;

        std::cerr << "starting client..." << std::endl;
        zk.connect(uri);

        for(;;) {
            auto text = zk.readNode("/hello");
            if(text == "world") {
                break;
            }

            ML::sleep(0.5);
        }
    });

    ML::sleep(1.0);

    std::cerr << "crash!" << std::endl;
    server.shutdown();

    ML::sleep(1.0);

    std::cerr << "restarting zookeeper..." << std::endl;
    server.start();

    ZookeeperConnection zk;
    zk.connect(uri);
    zk.createNode("/hello", "world", true, false);

    zk.readNode("/hello", [](int type, int state, std::string const & path, void * data) {
        std::cerr << "event type=" << type << " path=" << path << std::endl;
    }, 0);

    client.join();

    std::cerr << "crash & restart..." << std::endl;
    server.shutdown();
    server.start();

    while(zk.readNode("/hello") != "world") {
        ML::sleep(0.5);
    }
}

BOOST_AUTO_TEST_CASE( test_zookeeper )
{
    ML::set_default_trace_exceptions(false);

    ZooKeeper::TemporaryServer server;
    std::string uri = ML::format("localhost:%d", server.getPort());

    // avoid aborting test when killing a child process
    signal(SIGCHLD, SIG_DFL);

    std::cerr << "starting zookeeper..." << std::endl;
    server.start();

    ZookeeperConnection zk;
    zk.connect(uri);

    auto node = zk.createNode("/hello", "world", true, false);
    std::cerr << "nodeName = " << node.first << std::endl;

    int forked = 0;
    int killed = 0;

    std::vector<int> tasks;
    tasks.push_back(100);
    tasks.push_back(-10);
    tasks.push_back(+10);
    tasks.push_back(-20);
    tasks.push_back(+20);
    tasks.push_back(-50);
    tasks.push_back(+50);

    std::vector<int> pids;

    for(int task : tasks) {
        if(task > 0) {
            for(int i = 0; i != task; ++i) {
                int pid = fork();
                if(pid == -1) {
                    throw ML::Exception(errno, "fork");
                }

                if(pid == 0) {
                    pid = getpid();
                    std::cerr << "process created pid=" << pid << std::endl;

                    int res = prctl(PR_SET_PDEATHSIG, SIGHUP);
                    if(res == -1) {
                        throw ML::Exception(errno, "prctl failed");
                    }

                    ML::sleep(1);

                    std::cerr << pid << " trying to connect to " << uri << std::endl;
                    ZookeeperConnection zk;
                    zk.connect(uri);

                    for(;;) {
                        auto node = zk.readNode("/hello");
                        std::cerr << pid << " node=" << node << std::endl;
                        if(node == "world") {
                            break;
                        }

                        ML::sleep(1);
                    }

                    zk.createNode(ML::format("/%d", pid), "hello", true, false);
                    for(;;) {
                        ML::sleep(1);
                    }
                }
                else {
                    pids.push_back(pid);
                    ++forked;
                }
            }
        }
        else {
            for(int i = 0; i != -task; ++i) {
                int k = rand() % pids.size();
                int pid = pids[k];

                std::cerr << "killing pid=" << pid << std::endl;

                int res = kill(pid, SIGTERM);
                if(res == -1) {
                    throw ML::Exception(errno, "cannot kill child process");
                }

                int status = 0;
                res = waitpid(pid, &status, 0);
                if (res == -1) {
                    throw ML::Exception(errno, "failed to wait for child process to shutdown");
                }
 
                std::swap(pids[k], pids.back());
                pids.pop_back();

                ++killed;
            }
        }
    }

    bool ok = false;
    while(!ok) {
        auto children = zk.getChildren("/");
        std::cerr << "children = " << children << std::endl;

        if(children.size() == 2 + pids.size()) {
            ok = true;
            std::set<std::string> keys(children.begin(), children.end());
            for(int pid : pids) {
                ok &= keys.count(std::to_string(pid));
            }
        }

        ML::sleep(1);
    }

    for(int pid : pids) {
        int res = kill(pid, SIGTERM);
        if (res == -1) {
            throw ML::Exception(errno, "cannot kill child process");
        }

        ++killed;
    }

    for(int pid : pids) {
        int status = 0;
        int res = waitpid(pid, &status, 0);
        if (res == -1) {
            throw ML::Exception(errno, "failed to wait for child process to shutdown");
        }
    }

    std::cerr << "number of process forked: " << forked << std::endl;
    std::cerr << "number of process killed: " << killed << std::endl;

    BOOST_CHECK_EQUAL(forked, 180);
    BOOST_CHECK_EQUAL(killed, 180);
}

