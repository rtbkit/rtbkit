/* banker_temporary_server.h                                        -*- C++ -*-
   Jeremy Barnes, 19 October 2012
   Wolfgang Sourdeau, 20 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   A temporary server for testing of banker-based services. Starts one up in a
   temporary directory and gives the uri to connect to. */

// #include "jml/utils/environment.h"
// #include "jml/utils/file_functions.h"
// #include "jml/arch/timers.h"
// #include "soa/service/redis.h"
// #include <sys/stat.h>
// #include <sys/types.h>
// #include <sys/wait.h>
// #include <sys/un.h>
// #include <sys/socket.h>
// #include <signal.h>

#include <string>
#include <vector>

#include <boost/noncopyable.hpp>


namespace RTBKIT {

struct BankerTemporaryServer : boost::noncopyable {
    BankerTemporaryServer(const Redis::Address & redisAddress,
                          const std::string & zookeeperUri,
                          const std::string & zookeeperPath = "CWD");
    ~BankerTemporaryServer();

    void start();
    void shutdownWithSignal(int signal);
    void shutdown(); /* SIGTERM */
    void exterminate(); /* SIGKILL */

    Redis::Address redisAddress_;
    std::string zookeeperUri_;
    std::string zookeeperPath_;
    int serverPid_;
    std::string bankerAddress;
};

} // namespace RTBKIT

