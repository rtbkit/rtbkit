/* zookeeper_temporary_server.h                                        -*- C++ -*-
   Eric Robert, 25 February 2013
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   A temporary server for testing zookeeper services.
*/

#pragma once

#include "jml/utils/environment.h"
#include "jml/utils/file_functions.h"
#include "jml/arch/timers.h"
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/un.h>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <signal.h>

namespace ZooKeeper {

struct TemporaryServer : boost::noncopyable
{
    TemporaryServer(std::string uniquePath = "")
        : server(-1)
    {
        if (uniquePath == "") {
            ML::Env_Option<std::string> tmpDir("TMP", "./tmp");
            uniquePath = ML::format("%s/zookeeper-temporary-server-%d", tmpDir.get(), getpid());
        }

        static int uid;
        port = 4096 + (getpid() % 16768) + uid++;

        std::cerr << "starting zookeeper temporary server under " << uniquePath << std::endl;
        this->uniquePath = uniquePath;
    }

    ~TemporaryServer() {
        shutdown();
    }

    void start() {
        if (uniquePath == "" || uniquePath == "." || uniquePath == "..") {
            throw ML::Exception("unacceptable path");
        }

        createDirectory();
        createConfig();
        createServer();
    }

    void suspend()
    {
        if(server != -1)
        {
            // Stop boost test framework from interpreting this as a problem...
            signal(SIGCHLD, SIG_DFL);

            std::cerr << "suspending server with pid " << server << std::endl;
            int res = kill(server, SIGSTOP);
            if(res == -1)
            {
                throw ML::Exception("failed to suspend zookeeper");
            }
        }
        else
        {
            throw ML::Exception("server was not started!");
        }
    }

    void resume() 
    {
        if(server == -1)
            throw ML::Exception("server was not started!");

        int res = kill(server, SIGCONT);
        if(res == -1)
        {
            throw ML::Exception("failed to continue zookeeper");
        }
    }

    void shutdown() {
        if (server == -1)
            return;

        // Stop boost test framework from interpreting this as a problem...
        signal(SIGCHLD, SIG_DFL);

        int res = kill(server, SIGTERM);
        if (res == -1) {
            throw ML::Exception(errno, "cannot kill zookeeper");
        }

        int status = 0;
        res = waitpid(server, &status, 0);
        if (res == -1) {
            throw ML::Exception(errno, "failed to wait for zookeeper to shutdown");
        }

        server = -1;

        if (uniquePath != "") {
            std::cerr << "removing " << uniquePath << std::endl;
            int res = system(("rm -rf " + uniquePath).c_str());
            if (res) {
                throw ML::Exception(errno, "failed to remove zookeeper path");
            }
        }
    }

    int getPort() const {
        return port;
    }

private:
    void createDirectory() {
        struct stat stats;
        int res = stat(uniquePath.c_str(), &stats);
        if (res != -1) {
            res = system(ML::format("rm -rf %s", uniquePath).c_str());
            if (res == -1) throw ML::Exception(errno, "Unable to clean up old path");
        }

        res = system(ML::format("mkdir -p %s", uniquePath).c_str());
        if (res == -1) {
            throw ML::Exception(errno, "couldn't create directory");
        }

        res = system(ML::format("mkdir -p %s/data", uniquePath).c_str());
        if (res == -1) {
            throw ML::Exception(errno, "couldn't create data directory");
        }
    }

    void createConfig() {
        std::string filename = uniquePath + "/zoo.cfg";
        std::ofstream file(filename);
        if (!file) {
            throw ML::Exception("couldn't create zoo.cfg");
        }

        std::cerr << "zookeeper is using port " << port << std::endl;

        file << "tickTime=1000" << std::endl;
        file << "dataDir=" << uniquePath << "/data" << std::endl;
        file << "clientPort=" << port << std::endl;
        file << "dataLogDir=" << uniquePath << std::endl;
        file << "initLimit=10" << std::endl;
        file << "syncLimit=5" << std::endl;
        file << "maxClientCnxns=4096" << std::endl;
    }

    void createServer() {
        int pid = fork();
        if (pid == -1) {
            throw ML::Exception(errno, "fork");
        }

        if (pid == 0) {
            signal(SIGTERM, SIG_DFL);
            signal(SIGKILL, SIG_DFL);

            int res = prctl(PR_SET_PDEATHSIG, SIGHUP);
            if(res == -1) {
                throw ML::Exception(errno, "prctl failed");
            }

            std::string home = getenv("HOME");
            std::string path = home + "/local/bin/zookeeper/bin/zkServer.sh";
            std::string file = uniquePath + "/zoo.cfg";
            std::string logs = "ZOO_LOG_DIR=" + uniquePath;

            char const * args[] = { path.c_str(), "start-foreground", file.c_str(), (char *) 0 };
            char const * envp[] = { logs.c_str(), (char *) 0 };

            std::cerr << "running zookeeper from " << file << std::endl;

            res = execvpe(path.c_str(), (char **) args, (char **) envp);
            if (res == -1) {
                throw ML::Exception(errno, "zookeeper failed to start");
            }

            throw ML::Exception(errno, "zookeeper server didn't start");
        }
        else {
            server = pid;
        }

        ML::sleep(1);
    }

    int server;
    int port;
    std::string uniquePath;
};

} // namespace ZooKeeper

