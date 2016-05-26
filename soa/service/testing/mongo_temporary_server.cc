/**
 * mongo_temporary_server.cc
 * Mich, 2014-12-15
 * Copyright (c) 2014 Datacratic Inc.  All rights reserved.
 **/


#include <sys/socket.h>
#include <netinet/in.h>
#include "jml/utils/exc_assert.h"
#include "mongo_temporary_server.h"

using namespace std;
using namespace Mongo;
namespace fs = boost::filesystem;
using namespace Datacratic;

MongoTemporaryServer::
MongoTemporaryServer(const string & uniquePath, const int portNum)
    : state(Inactive), uniquePath_(uniquePath)
{
    static int index(0);
    ++index;

    if (uniquePath_.empty()) {
        ML::Env_Option<string> tmpDir("TMP", "./tmp");
        uniquePath_ = ML::format("%s/mongo-temporary-server-%d-%d",
                                 tmpDir.get(), getpid(), index);
        cerr << ("starting mongo temporary server under unique path "
                 + uniquePath_ + "\n");
    }

    if (portNum == 0) {
        int freePort = 0;
        for (int i = 0; i < 100; ++ i) {
            struct sockaddr_in addr;
            addr.sin_family = AF_INET;
            auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
            freePort = rand() % 15000 + 5000; // range 15000 - 20000
            addr.sin_port = htons(freePort);
            addr.sin_addr.s_addr = INADDR_ANY;
            int res = ::bind(sockfd, (struct sockaddr *) &addr, sizeof(addr));
            if (res == 0) {
                close(sockfd);
                break;
            }
            freePort = 0;
        }
        if (freePort == 0) {
            throw ML::Exception("Failed to find free port");
        }
        this->portNum = freePort;
    }
    else {
        this->portNum = portNum;
    }

    start();
}

MongoTemporaryServer::
~MongoTemporaryServer()
{
    shutdown();
}

void
MongoTemporaryServer::
testConnection()
{
    // 3.  Connect to the server to make sure it works
    int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock == -1) {
        throw ML::Exception(errno, "socket");
    }

    sockaddr_un addr;
    addr.sun_family = AF_UNIX;

    // Wait for it to start up
    fs::directory_iterator endItr;
    fs::path socketdir(socketPath_);
    bool connected(false);
    for (unsigned i = 0; i < 100 && !connected;  ++i) {
        // read the directory to wait for the socket file to appear
        for (fs::directory_iterator itr(socketdir); itr != endItr; ++itr) {
            ::strcpy(addr.sun_path, itr->path().string().c_str());
            int res = ::connect(sock,
                                (const sockaddr *) &addr, SUN_LEN(&addr));
            if (res == 0) {
                connected = true;
            }
            else if (res == -1) {
                if (errno != ECONNREFUSED && errno != ENOENT) {
                    throw ML::Exception(errno, "connect");
                }
            }
        }
        ML::sleep(0.1);
    }

    if (!connected) {
        throw ML::Exception("mongod didn't start up in 10 seconds");
    }
    ::close(sock);
}

void
MongoTemporaryServer::
start()
{
    // Check the unique path
    if (uniquePath_ == "" || uniquePath_[0] == '/' || uniquePath_ == "."
            || uniquePath_ == "..") {
        throw ML::Exception("unacceptable unique path");
    }

    // 1.  Create the directory

    // First check that it doesn't exist
    struct stat stats;
    int res = ::stat(uniquePath_.c_str(), &stats);
    if (res != -1 || (errno != EEXIST && errno != ENOENT)) {
        throw ML::Exception(errno, "unique path " + uniquePath_
                            + " already exists");
    }
    cerr << "creating directory " << uniquePath_ << endl;
    if (!fs::create_directory(fs::path(uniquePath_))) {
        throw ML::Exception("could not create unique path " + uniquePath_);
    }

    socketPath_ = uniquePath_ + "/mongo-socket";
    logfile_ = uniquePath_ + "/output.log";
    int UNIX_PATH_MAX=108;

    if (socketPath_.size() >= UNIX_PATH_MAX) {
        throw ML::Exception("unix socket path is too long");
    }

    // Create unix socket directory
    fs::path unixdir(socketPath_);
    if (!fs::create_directory(unixdir)) {
        throw ML::Exception(errno,
                            "couldn't create unix socket directory for Mongo");
    }
    auto onStdOut = [&] (string && message) {
         cerr << "received message on stdout: /" + message + "/" << endl;
        //  receivedStdOut += message;
    };
    auto stdOutSink = make_shared<Datacratic::CallbackInputSink>(onStdOut);

    loop_.addSource("runner", runner_);
    loop_.start();

    auto onTerminate = [&] (const RunResult & result) {
    };
    runner_.run({"/usr/bin/mongod",
                 "--bind_ip", "localhost", "--port", to_string(portNum),
                 "--logpath", logfile_, "--dbpath", uniquePath_,
                 "--unixSocketPrefix", socketPath_, "--nojournal"},
                onTerminate, nullptr, stdOutSink);
    // connect to the socket to make sure everything is working fine
    testConnection();
    string payload("db.createUser({user: 'testuser', pwd: 'testpw',"
                                   "roles: ['userAdmin', 'dbAdmin']})");
    RunResult runRes = execute({"/usr/bin/mongo",
                                "localhost:" + to_string(portNum)},
                               nullptr, nullptr, payload);
    ExcAssertEqual(runRes.processStatus(), 0);
    execute({"/usr/bin/mongo", "localhost:" + to_string(portNum)},
                               nullptr, nullptr, "db.getUsers()");

    state = Running;
}

void
MongoTemporaryServer::
suspend()
{
    runner_.kill(SIGSTOP);
    state = Suspended;
}

void
MongoTemporaryServer::
resume()
{
    runner_.kill(SIGCONT);
    state = Running;
}

void
MongoTemporaryServer::
shutdown()
{
    if (runner_.childPid() < 0) {
        return;
    }
    runner_.kill();
    runner_.waitTermination();
    if (uniquePath_ != "") {
        cerr << "removing " << uniquePath_ << endl;
        // throws an exception on error
        fs::remove_all(fs::path(uniquePath_));
        state = Stopped;
    }
}
