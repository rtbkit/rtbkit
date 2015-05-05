/**
 * mongo_temporary_server.cc
 * Mich, 2014-12-15
 * Copyright (c) 2014 Datacratic Inc.  All rights reserved.
 **/


#include "jml/utils/exc_assert.h"
#include "mongo_temporary_server.h"

using namespace std;
using namespace Mongo;
namespace fs = boost::filesystem;
using namespace Datacratic;

MongoTemporaryServer::
MongoTemporaryServer(const string & uniquePath)
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

    runner_.run({"/usr/bin/mongod",
                 "--bind_ip", "localhost", "--port", "28356",
                 "--logpath", logfile_, "--dbpath", uniquePath_,
                 "--unixSocketPrefix", socketPath_, "--nojournal"},
                nullptr, nullptr, stdOutSink);
    // connect to the socket to make sure everything is working fine
    testConnection();
    string payload("db.addUser('testuser','testpw',true)");
    RunResult runRes = execute({"/usr/bin/mongo", "localhost:28356"},
                            nullptr, nullptr, payload);
    ExcAssertEqual(runRes.processStatus(), 0);
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
