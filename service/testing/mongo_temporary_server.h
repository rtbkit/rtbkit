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
#include "soa/service/message_loop.h"
#include "soa/service/runner.h"
#include "soa/service/sink.h"
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

        this->uniquePath_ = uniquePath;
        start();
    }

    ~MongoTemporaryServer()
    {
        shutdown();
    }
    
    void testConnection()
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
            boost::filesystem::path socketdir(socketPath_);
            int res=0;
            for (unsigned i = 0;  i < 1000;  ++i) {
                // read the directory to wait for the socket file to appear
                bool found = false;
                for( fs::directory_iterator itr(socketdir) ; itr!=endItr ; ++itr)
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
                    ML::sleep(0.01);
            }

            if (res != 0)
                throw ML::Exception("mongod didn't start up in 10 seconds");
            close(sock);
            std::cerr << "Connection to mongodb socket established " << std::endl;
    }

    void start()
    {
        using namespace std;
        namespace fs = boost::filesystem;
        // Check the unique path
        if (uniquePath_ == "" || uniquePath_[0] == '/' || uniquePath_ == "."
            || uniquePath_ == "..")
            throw ML::Exception("unacceptable unique path");

        // 1.  Create the directory

        // First check that it doesn't exist
        struct stat stats;
        int res = stat(uniquePath_.c_str(), &stats);
        if (res != -1 || (errno != EEXIST && errno != ENOENT))
            throw ML::Exception(errno, "unique path " + uniquePath_
                                + " already exists");
        cerr << "creating directory " << uniquePath_ << endl;
        if(!fs::create_directory(fs::path(uniquePath_)))
            throw ML::Exception("could not create unique path %s",uniquePath_.c_str());

        socketPath_ = uniquePath_ + "/mongo-socket";
        logfile_ = uniquePath_ + "/output.log";
        int UNIX_PATH_MAX=108;

        if (socketPath_.size() >= UNIX_PATH_MAX)
            throw ML::Exception("unix socket path is too long");

        // Create unix socket directory
        boost::filesystem::path unixdir(socketPath_);
        if( !boost::filesystem::create_directory(unixdir))
            throw ML::Exception(errno, "couldn't create unix socket directory for Mongo");
        auto onStdOut = [&] (string && message) {
             cerr << "received message on stdout: /" + message + "/" << endl;
            //  receivedStdOut += message;
        };
        auto stdOutSink = make_shared<Datacratic::CallbackInputSink>(onStdOut);

        loop_.addSource("runner", runner_);
        loop_.start();

        cerr << "about to run command using runner " << endl;
        runner_.run({"/usr/bin/mongod",
                    "--port", "28356",
                    "--logpath",logfile_.c_str(),"--bind_ip",
                    "localhost","--dbpath",uniquePath_.c_str(),"--unixSocketPrefix",
                    socketPath_.c_str(),"--nojournal"}, nullptr, nullptr,stdOutSink);
        // connect to the socket to make sure everything is working fine
        testConnection();
        string payload("db.addUser('testuser','testpw',true)");
        execute(loop_,{"/usr/bin/mongo","localhost:28356"}, nullptr, nullptr, payload);
        state = Running;
    }

    void suspend() {
        runner_.kill(SIGSTOP);
        state = Suspended;
    }

    void resume() {
        runner_.kill(SIGCONT);
        state = Running;
    }


    void shutdown()
    {
        namespace fs = boost::filesystem;
        if(runner_.childPid() < 0) 
            return;
        runner_.kill();
        runner_.waitTermination();
        if (uniquePath_ != "") {
            using namespace std;
            cerr << "removing " << uniquePath_ << endl;
            // throws an exception on error
            fs::remove_all(fs::path(uniquePath_));
            state = Stopped;
        }
    }


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
