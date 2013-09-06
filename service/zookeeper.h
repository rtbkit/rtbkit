/* zookeeper.h                                                     -*- C++ -*-
   Jeremy Barnes, 17 August 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#pragma once

#include <zookeeper/zookeeper.h>

#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include "jml/utils/guard.h"

#include <set>
#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <unordered_map>


namespace Datacratic {

//forward declaration
struct ZookeeperCallback;

typedef void (* ZookeeperCallbackType)(int type, int state, std::string const & path, 
                                       void * data);
struct CallbackInfo
{
    CallbackInfo(ZookeeperCallback *cb=nullptr):callback(cb),valid(false)
    {
    }

    ZookeeperCallback *callback;
    bool valid;
};
typedef std::unordered_map<uintptr_t, CallbackInfo> ZookeeperCallbackMap;

class ZookeeperCallbackManager 
{
public:

    static ZookeeperCallbackManager & instance();

    ZookeeperCallbackManager():id_(0)
    {}

    uintptr_t  createCallback(ZookeeperCallbackType watch, 
                              std::string const & path, void * data) ;

    // returns a pointer to a callback with the specified id. nullptr if not found
    // note that if found the callback will be removed from the list of callbacks
    // it is the responsibility of the caller to invoke call on the pointer which
    // frees up the memory
    ZookeeperCallback *popCallback(uintptr_t id) ;

    // Marks callback id with the specified value
    bool mark(uintptr_t id, bool valid);
    // sends the specified event to all callbacks and deletes all callbacks
    void sendEvent(int type, int state) ;

    // at this point used for tests for want of a better mechanism
    uintptr_t getId() const
    {
        return id_;
    }
private:
    // global lock for access to the linked list of callbacks
    std::mutex lock;
    ZookeeperCallbackMap callbacks_;
    uintptr_t id_;
};


struct ZookeeperCallback  {

    uintptr_t id;
    ZookeeperCallbackManager *mgr;
    ZookeeperCallbackType callback;
    std::string path;
    void * user;
    // we want to make sure only the ZookeeperManager can create callbacks
    friend class ZookeeperCallbackManager;
protected:
    ZookeeperCallback(uint64_t id, ZookeeperCallbackManager *mgr, 
                      ZookeeperCallbackType callback, 
                      std::string path, void * user) : 
        id(id), mgr(mgr), callback(callback), path(path), user(user) 
    {
    }
public:
    void call(int type, int state) 
    {
        callback(type, state, path, user);
        delete this;
    }
};


/*****************************************************************************/
/* ZOOKEEPER CONNECTION                                                      */
/*****************************************************************************/

struct ZookeeperConnection {


    ZookeeperConnection();
    ~ZookeeperConnection() { close(); }

    static std::string printEvent(int eventType);

    static std::string printState(int state);

    /** Connect synchronously. */
    void connect(const std::string & host,
                 double timeoutInSeconds = 5.0);

    /** Connect with a session id and password 
     * 
     *  precondition: password.size() <= 16
     */ 
    void connectWithCredentials(const std::string & host,
                                int64_t sessionId,
                                const char *password,
                                double timeoutInSeconds = 5.0);

    std::pair<int64_t, const char *> sessionCredentials() const;

    void reconnect();

    void close();

    enum CheckResult {
        CR_RETRY,  ///< Retry the operation
        CR_DONE    ///< Finish the operation
    };

    /** Check the result of an operation.  Will either return
        RETRY if the operation should be redone, DONE if the
        operation completed, or will throw an exception if there
        was an error.
    */
    CheckResult checkRes(int returnCode, int & retries,
                         const char * operation, const char * path);

    std::pair<std::string, bool>
    createNode(const std::string & path,
               const std::string & value,
               bool ephemeral,
               bool sequence,
               bool mustSucceed = true,
               bool createPath = false);

    /** Delete the given node.  If throwIfNodeMissing is false, then a missing
        node will not be considered an error.  Returns if the node was deleted
        or not, and throws an exception in the case of an error.
    */
    bool deleteNode(const std::string & path, bool throwIfNodeMissing = true);

    /** Create nodes such that the given path exists. */
    void createPath(const std::string & path);

    /** Remove the entire path including all children. */
    void removePath(const std::string & path);

    /** Return if the node exists or not. */
    bool nodeExists(const std::string & path,
                    ZookeeperCallbackType watcher = 0,
                    void * watcherData = 0);

    std::string readNode(const std::string & path,
                         ZookeeperCallbackType watcher = 0,
                         void * watcherData = 0);

    void writeNode(const std::string & path, const std::string & value);

    std::vector<std::string>
    getChildren(const std::string & path,
                bool failIfNodeMissing = true,
                ZookeeperCallbackType watcher = 0,
                void * watcherData = 0);

    static void eventHandlerFn(zhandle_t * handle,
                               int event,
                               int state,
                               const char * path,
                               void * context);

    /** Remove trailing slash so it can be used as a path. */
    static std::string fixPath(const std::string & path);

    std::mutex connectMutex;
    std::condition_variable cv;
    std::string host;
    int recvTimeout;
    std::shared_ptr<clientid_t> clientId;
    zhandle_t * handle;

    struct Node {
        Node(std::string const & path) : path(path) {
        }

        Node(std::string const & path, std::string const & value) : path(path), value(value) {
        }

        bool operator<(Node const & other) const {
            return path < other.path;
        }

        std::string path;
        mutable std::string value;
    };

    std::set<Node> ephemerals;

    ZookeeperCallbackManager &callbackMgr_;

private:
    void connectImpl(const std::string &host, 
                     double timeoutInSeconds,
                     clientid_t *clientId);
    
};

} // namespace Datacratic
