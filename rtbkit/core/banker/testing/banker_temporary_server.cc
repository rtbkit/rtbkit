/* banker_temporary_server.h                                        -*- C++ -*-
   Jeremy Barnes, 19 October 2012
   Wolfgang Sourdeau, 20 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   A temporary server for testing of banker-based services. Starts one up in a
   temporary directory and gives the uri to connect to. */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>

#include <iostream>

#include <jml/arch/exception.h>
#include <jml/arch/timers.h>
#include "rtbkit/core/banker/master_banker.h"

#include "banker_temporary_server.h"

using namespace std;

namespace {
    bool mustTerminate(false);

    void handleSIGTERM(int) {
        cerr << "handling sigterm" << endl;
        mustTerminate = true;
    }
}

namespace RTBKIT {

BankerTemporaryServer::
BankerTemporaryServer(const Redis::Address & redisAddress,
                      const std::string & zookeeperUri,
                      const std::string & zookeeperPath)
    : redisAddress_(redisAddress),
      zookeeperUri_(zookeeperUri),
      zookeeperPath_(zookeeperPath),
      serverPid_(-1)
{
    start();
}

BankerTemporaryServer::
~BankerTemporaryServer()
{
    shutdown();
}

void
BankerTemporaryServer::
start()
{
    // 2.  Start the server
    int pid = fork();
    if (pid == -1)
        throw ML::Exception(errno, "fork");
    if (pid == 0) {
        signal(SIGTERM, handleSIGTERM);
        signal(SIGKILL, SIG_DFL);

        auto redis = make_shared<Redis::AsyncConnection>(redisAddress_);
        redis->test();

        // cerr << "tested redis" << endl;

        auto proxies = std::make_shared<ServiceProxies>();
        proxies->useZookeeper(zookeeperUri_, zookeeperPath_);
        cerr << "zookeeperPath: " << zookeeperPath_ << endl;

        auto banker = make_shared<MasterBanker>(proxies, "masterBanker");

        // cerr << "initializing banker" << endl;

        banker->init(make_shared<RedisBankerPersistence>(redis));

        // cerr << "binding banker" << endl;

        auto addr = banker->bindTcp();

        cerr << "MasterBanker: addrs = " << addr.first << "," << addr.second
             << endl;
        bankerAddress = addr.second;

        // cerr << "running banker" << endl;
        banker->start();
        proxies->config->dump(cerr);
    
        while (!mustTerminate) {
            ML::sleep(1);
        }

        /* force the destruction of the banker instance */
        banker.reset();

        // Exit without running destructors, which will greatly confuse things
        _exit(0);
    }
    else {
        cerr << "MasterBanker: pid = " << pid << endl;
        serverPid_ = pid;

#if 0
        // 3.  Connect to the server to make sure it works
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == -1)
            throw ML::Exception(errno, "socket");

        struct sockaddr_in name;
        name.sin_family = AF_INET;
        inet_aton("127.0.0.1", &name.sin_addr);
        name.sin_port = htons(9876);

        cerr << "Attempting to connect to daemon process..." << endl;

        int res;
        // Wait for it to start up
        for (unsigned i = 0; i < 1000;  ++i) {
            res = connect(sock, (const sockaddr *)&name, sizeof(name));
            if (res == 0) break;
            if (res == -1 && errno != ECONNREFUSED)
                throw ML::Exception(errno, "connect");
            
            ML::sleep(0.01);
        }

        if (res != 0)
            throw ML::Exception("banker didn't start up in 10 seconds");
        cerr << "Master Banker daemon is ready" << endl;
        ::close(sock);
#endif
    }
}

void
BankerTemporaryServer::
shutdownWithSignal(int signum)
{
    cerr << "shutdownWithSignal; serverPid_ = " << serverPid_ << endl;

    if (serverPid_ == -1)
        return;

    // Stop boost test framework from interpreting this as a problem...
    sighandler_t oldHandler = signal(SIGCHLD, SIG_DFL);

    int res = kill(serverPid_, signum);
    if (res == -1)
        throw ML::Exception(errno, "exterminate banker");

    cerr << "done kill" << endl;

    int status = 0;
    res = waitpid(serverPid_, &status, 0);
    if (res == -1)
        throw ML::Exception(errno, "wait for banker shutdown");

    serverPid_ = -1;
    bankerAddress = "";

    signal(SIGCHLD, oldHandler);
}

void
BankerTemporaryServer::
shutdown()
{
    shutdownWithSignal(SIGTERM);
}

void
BankerTemporaryServer::
exterminate()
{
    shutdownWithSignal(SIGKILL);
}

} // namespace RTBKIT

