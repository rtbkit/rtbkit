/* mongo_temporary_server.h                                        -*- C++ -*-
   Sunil Rottoo, 2 September 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

   A temporary server for testing of mongo-based services.  Starts one up in a
   temporary directory and gives the uri to connect to.

   NOTE: This is not a self-contained util. mongod need to be installed prior
         to using mongo_temporary_server.
*/

#pragma once

#include "jml/utils/environment.h"
#include "jml/utils/file_functions.h"
#include "jml/arch/timers.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/un.h>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <signal.h>
#include <boost/noncopyable.hpp>
#include <boost/filesystem.hpp>
#include "soa/service/message_loop.h"
#include "soa/service/runner.h"
#include "soa/service/sink.h"
namespace Mongo {

struct MongoTemporaryServer : boost::noncopyable {

    MongoTemporaryServer(std::string uniquePath = "");
    ~MongoTemporaryServer();
    
    void testConnection();
    void start();
    void suspend();
    void resume();
    void shutdown();

private:
    enum State { Inactive, Stopped, Suspended, Running };
    State state;
    std::string uniquePath_;
    std::string socketPath_;
    std::string logfile_;
    int serverPid;
    Datacratic::MessageLoop loop_;
    Datacratic::Runner runner_;
};

} // namespace Mongo
