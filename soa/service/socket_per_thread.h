/* socket_per_thread.h                                             -*- C++ -*-
   Jeremy Barnes, 14 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   ZeroMQ Socket per thread.
*/

#ifndef __zmq__socket_per_thread_h__
#define __zmq__socket_per_thread_h__


#include "zmq.hpp"
#include <boost/thread/tss.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread/locks.hpp>
#include "jml/compiler/compiler.h"
#include "jml/arch/spinlock.h"
#include "jml/arch/exception.h"
#include <set>


namespace Datacratic {


/*****************************************************************************/
/* SOCKET PER THREAD                                                         */
/*****************************************************************************/

/** A simple structure that creates on-demand a different zeromq socket per
    thread that will connect to a given endpoint.  Using this you can make
    sure that you don't share sockets amongst multiple threads.

    Note that if you use this from the same thread that will be shutting down
    the zeromq context, then you need to call cleanupAllForThread() before
    you try to terminate zeromq, otherwise it will hang on shutdown.
*/

struct SocketPerThread : boost::noncopyable {

    /** Default constructor.  You must call init() before accessing it. */
    SocketPerThread();

    /** Constructor to create a socket of the given type within the given
        context that will connect to the given URI on demand. */
    SocketPerThread(zmq::context_t & context,
                    int type,
                    const std::string & uri,
                    bool allowForceClose = true);

    ~SocketPerThread();

    /** Initialize to create a socket of the given type within the given
        context that will connect to the given URI on demand. */
    void init(zmq::context_t & context,
              int type,
              const std::string & uri,
              bool allowForceClose = true);

    void shutdown();

    zmq::context_t * context;   ///< Owning zeromq context
    int type;                   ///< Type to create
    std::string uri;            ///< URI to connect to
    bool allowForceClose;       ///< Cleanup open sockets on destruction?
    mutable int numOpen;        ///< Num of open connections to detect misuse

    /** Return (creating if necessary) the socket for this thread. */
    inline zmq::socket_t & operator () () const
    {
        if (state != READY)
            throw ML::Exception("socket not ready: %d", state);

        if (JML_UNLIKELY(!entries.get())) {
            initForThisThread();
        }

        return *entries->sock;
    }

    /** Initialize the socket for this thread. */
    void initForThisThread() const;
    
    /** Prematurely pretend that this thread has exited for this socket. */
    void cleanupThisThread();

    /** Prematurely pretend that this thread has exited for all of the
        SocketPerThread instances that are open.  Commonly called just
        before shutdown in the main thread.
    */
    static void cleanupAllForThread();
    
private:
    enum {
        NOTINITIALIZED = 12321,
        READY = 349244,
        FINISHED = 293845
    };
    int state;

    /** How we store the actual sockets internally. */
    struct Entry {
        zmq::socket_t * sock;
        SocketPerThread * owner;
    };

    /** All threads that are alive */
    std::set<Entry *> allThreads;

    /** Lock to protect allThreads. */
    typedef ML::Spinlock Lock;
    typedef boost::unique_lock<Lock> Guard;
    mutable Lock allThreadsLock;

    /** Remove the given entry from allThreads. */
    void removeThreadEntry(Entry * entry);

    /** Add the given entry to allThreads. */
    void addThreadEntry(Entry * entry);

    /** Function called whenever a thread exits to clean up its entry. */
    static void onFreeEntry(Entry * entry);
    
    /** Thread-specific pointer to our entry that contains our socket. */
    mutable boost::thread_specific_ptr<Entry> entries;

    /** Global thread-specific pointer to a set of all active sockets for this
        thread.
    */
    static boost::thread_specific_ptr<std::set<SocketPerThread *> > allForThread;
};

} // namespace Datacratic



#endif /* __zmq__socket_per_thread_h__ */

