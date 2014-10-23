/* mongo_temporary_server.h                                        -*- C++ -*-
   Sunil Rottoo, 2 September 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

   A temporary server for testing of mongo-based services.  Starts one up in a
   temporary directory and gives the uri to connect to.
*/

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
namespace Mongo {

struct MongoTemporaryServer : boost::noncopyable {

    MongoTemporaryServer(std::string uniquePath = "")
        : state(Inactive)
    {
        static int index;
        ++index;

        using namespace std;
        if (uniquePath == "") {
            ML::Env_Option<std::string> tmpDir("TMP", "./tmp");
            uniquePath = ML::format("%s/mongo-temporary-server-%d-%d",
                                    tmpDir.get(), getpid(), index);
            cerr << "starting mongo temporary server under unique path "
                 << uniquePath << endl;
        }

        this->uniquePath = uniquePath;
        start();
    }

    ~MongoTemporaryServer()
    {
        shutdown();
    }

    void start()
    {
        using namespace std;

        // Check the unique path
        if (uniquePath == "" || uniquePath[0] == '/' || uniquePath == "."
            || uniquePath == "..")
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
        cerr << "creating directory " << uniquePath << endl;
        res = system(ML::format("mkdir -p %s", uniquePath).c_str());
        if (res == -1)
            throw ML::Exception(errno, "couldn't mkdir");
        
        string unixPath = uniquePath + "/mongo-socket";
        string logfile = uniquePath + "/output.log";
        int UNIX_PATH_MAX=108;

        if (unixPath.size() >= UNIX_PATH_MAX)
            throw ML::Exception("unix socket path is too long");

        // Create unix socket directory
        boost::filesystem::path unixdir(unixPath);
        if( !boost::filesystem::create_directory(unixdir))
            throw ML::Exception(errno, "couldn't create unix socket directory for Mongo");

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

            cerr << "running mongo with database files at " << uniquePath << endl;
            cerr << "logging at " << logfile << endl;
            res = execlp("mongod",
                             "mongod",
                             "--port", "28335",
                         "--logpath",logfile.c_str(),
                         "--bind_ip", "127.0.0.1",
                             "--dbpath", uniquePath.c_str(),
                         "--unixSocketPrefix",unixPath.c_str(),
                         "--nojournal",
                             (char *)0);
            if (res == -1)
                throw ML::Exception(errno, "mongo failed to start");

            throw ML::Exception(errno, "mongo failed to start");
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
            // Wait for it to start up
            namespace fs = boost::filesystem;
            fs::directory_iterator endItr;

            for (unsigned i = 0;  i < 2000;  ++i) {
                // read the directory to wait for the socket file to appear
                bool found = false;
                for( fs::directory_iterator itr(unixdir) ; itr!=endItr ; ++itr)
                {
                    strcpy(addr.sun_path, itr->path().string().c_str());
                    found = true;
                }
                if(found)
                {
                    res = connect(sock, (const sockaddr *)&addr, SUN_LEN(&addr));
                    if (res == 0) break;
                    if (res == -1 && errno != ECONNREFUSED && errno != ENOENT)
                        throw ML::Exception(errno, "connect");
                }
                else
                    ML::sleep(0.1);
            }

            if (res != 0)
                throw ML::Exception("mongod didn't start up in 100 seconds");
            close(sock);
        }

        cerr << "address is " << unixPath << endl;

        state = Running;
    }

    void suspend() {
        if (serverPid == -1)
            return;

        signal(SIGCHLD, SIG_DFL);

        int res = kill(serverPid, SIGSTOP);
        if (res == -1) {
            throw ML::Exception(errno, "suspend mongo");
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
            throw ML::Exception(errno, "resuming mongo");
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
            throw ML::Exception(errno, "kill mongod");

        int status = 0;
        res = waitpid(serverPid, &status, 0);
        if (res == -1)
            throw ML::Exception(errno, "wait for mongod shutdown");

        serverPid = -1;

        if (uniquePath != "") {
            using namespace std;
            cerr << "removing " << uniquePath << endl;
            int rmstatus = system(("rm -rf " + uniquePath).c_str());
            if (rmstatus)
                throw ML::Exception(errno, "removing mongod path");

            state = Stopped;
        }
    }


private:
    enum State { Inactive, Stopped, Suspended, Running };
    State state;
    std::string uniquePath;
    int serverPid;
};

} // namespace Mongo
