/* redis_temporary_server.h                                        -*- C++ -*-
   Jeremy Barnes, 19 October 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   A temporary server for testing of redis-based services.  Starts one up in a
   temporary directory and gives the uri to connect to.
*/

#include "jml/utils/environment.h"
#include "jml/utils/file_functions.h"
#include "jml/arch/timers.h"
#include "soa/service/redis.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/un.h>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <signal.h>

namespace Redis {

struct RedisTemporaryServer : boost::noncopyable {

    RedisTemporaryServer(std::string uniquePath = "")
        : state(Inactive)
    {
        static int index;
        ++index;

        using namespace std;
        if (uniquePath == "") {
            ML::Env_Option<std::string> tmpDir("TMP", "./tmp");
            std::string dir = tmpDir;
            if(dir[0] == '/') dir.insert(0, 1, '.');
            uniquePath = ML::format("%s/redis-temporary-server-%d-%d",
                                    dir, getpid(), index);
            cerr << "starting redis temporary server under unique path "
                 << uniquePath << endl;
        }

        this->uniquePath = uniquePath;
        start();
    }

    ~RedisTemporaryServer()
    {
        shutdown();
    }

    void start()
    {
        using namespace std;

        // Check the unique path
        if (uniquePath.empty() || uniquePath == "."  || uniquePath == "..")
            throw ML::Exception("unacceptable unique path");

        // 1.  Create the directory

#if 0
        char cwd[1024];

        if (uniquePath[0] != '/')
            uniquePath = getcwd(cwd, 1024) + string("/") + uniquePath;

        cerr << "uniquePath = " << uniquePath << endl;
#endif

        // First check that it doesn't exist
        struct stat stats;
        int res = stat(uniquePath.c_str(), &stats);
        if (res != -1 || (errno != EEXIST && errno != ENOENT))
            throw ML::Exception(errno, "unique path " + uniquePath
                                + " already exists");
        
        res = system(ML::format("mkdir -p %s", uniquePath).c_str());
        if (res == -1)
            throw ML::Exception(errno, "couldn't mkdir");
        
        string unixPath = uniquePath + "/redis-socket";

        int UNIX_PATH_MAX=108;

        if (unixPath.size() >= UNIX_PATH_MAX)
            throw ML::Exception("unix socket path is too long");

        // Create unix socket
        res = mknod(unixPath.c_str(), 0777 | S_IFIFO, 0);
        if (res == -1)
            throw ML::Exception(errno, "couldn't create unix socket for Redis");

        //ML::set_permissions(unixPath, "777", "");

        // 2.  Start the server
        int pid = fork();
        if (pid == -1)
            throw ML::Exception(errno, "fork");
        if (pid == 0) {
            int res = prctl(PR_SET_PDEATHSIG, SIGHUP);
            if(res == -1) {
                throw ML::Exception(errno, "prctl failed");
            }

            signal(SIGTERM, SIG_DFL);
            signal(SIGKILL, SIG_DFL);

            cerr << "running redis" << endl;
            res = execlp("redis-server",
                             "redis-server",
                             "--port", "0",
                             "--unixsocket", "./redis-socket",
                             "--dir", uniquePath.c_str(),
                             (char *)0);
            if (res == -1)
                throw ML::Exception(errno, "redis failed to start");

            throw ML::Exception(errno, "redis failed to start");
        }
        else {
            serverPid = pid;
        }

        {
            // 3.  Connect to the server to make sure it works
            int sock = socket(AF_UNIX, SOCK_STREAM, 0);
            if (sock == -1)
                throw ML::Exception(errno, "socket");

            sockaddr_un addr;
            addr.sun_family = AF_UNIX;
            strcpy(addr.sun_path, unixPath.c_str());

            // Wait for it to start up
            for (unsigned i = 0;  i < 1000;  ++i) {
                res = connect(sock, (const sockaddr *)&addr, SUN_LEN(&addr));
                if (res == 0) break;
                if (res == -1 && errno != ECONNREFUSED && errno != ENOENT)
                    throw ML::Exception(errno, "connect");
            
                ML::sleep(0.01);
            }

            if (res != 0)
                throw ML::Exception("redis didn't start up in 10 seconds");

            close(sock);
        }

        cerr << "address is " << unixPath << endl;

        this->address_ = Address::unix(unixPath);
        state = Running;
    }

    void suspend() {
        if (serverPid == -1)
            return;

        signal(SIGCHLD, SIG_DFL);

        int res = kill(serverPid, SIGSTOP);
        if (res == -1) {
            throw ML::Exception(errno, "suspend redis");
        }
        state = Suspended;
    }

    void resume() {
        if (serverPid == -1)
            return;

        if (state != Suspended) {
            throw ML::Exception("Server has not been suspended");
        }

        int res = kill(serverPid, SIGCONT);
        if (res == -1) {
            throw ML::Exception(errno, "resuming redis");
        }

        state = Running;
    }


    void shutdown()
    {
        if (serverPid == -1)
            return;

        // Stop boost test framework from interpreting this as a problem...
        signal(SIGCHLD, SIG_DFL);

        int res = kill(serverPid, SIGTERM);
        if (res == -1)
            throw ML::Exception(errno, "kill redis");

        int status = 0;
        res = waitpid(serverPid, &status, 0);
        if (res == -1)
            throw ML::Exception(errno, "wait for redis shutdown");

        this->address_ = Address();

        serverPid = -1;

        if (uniquePath != "") {
            using namespace std;
            cerr << "removing " << uniquePath << endl;
            int rmstatus = system(("rm -rf " + uniquePath).c_str());
            if (rmstatus)
                throw ML::Exception(errno, "removing redis path");

            state = Stopped;
        }
    }

    Address address() const
    {
        return address_;
    }

    operator Address() const
    {
        return address();
    }

private:
    enum State { Inactive, Stopped, Suspended, Running };
    State state;
    Address address_;
    std::string uniquePath;
    int serverPid;
};

} // namespace Redis

