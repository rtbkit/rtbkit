/* transport.h                                                     -*- C++ -*-
   Jeremy Barnes, 23 February 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.
   
   Transport abstraction for endpoints.
*/

#ifndef __rtb__transport_h__
#define __rtb__transport_h__

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>
#include <boost/thread/locks.hpp>
#include <ace/SOCK_Stream.h>
#include <ace/Synch.h>
#include "jml/arch/exception.h"
#include "jml/arch/demangle.h"
#include "jml/utils/guard.h"
#include "jml/arch/format.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/cmp_xchg.h"
#include "jml/arch/spinlock.h"
#include "soa/types/date.h"
#include "soa/jsoncpp/json.h"
#include <boost/type_traits/is_convertible.hpp>
#include <boost/enable_shared_from_this.hpp>

namespace Datacratic {


struct ConnectionHandler;
struct EndpointBase;

extern boost::function<void (const char *, float)> onLatencyEvent;

/*****************************************************************************/
/* TRANSPORT BASE                                                            */
/*****************************************************************************/

/** Deals with the underlying socket infrastructure. */

struct TransportBase : public std::enable_shared_from_this<TransportBase> {
    friend class EndpointBase;
    friend class ActiveEndpoint;
    friend class PassiveEndpoint;

    TransportBase(); // throws
    TransportBase(EndpointBase * endpoint);

    virtual ~TransportBase();
    
    /* Event callbacks */
    virtual int handleInput();
    virtual int handleOutput();
    virtual int handlePeerShutdown();
    virtual int handleTimeout();
    virtual int handleError(const std::string & error);
    virtual int handleAsync(const boost::function<void ()> & callback,
                            const char * name, Date dateSet);

    virtual ssize_t send(const char * buf, size_t len, int flags) = 0;
    virtual ssize_t recv(char * buf, size_t buf_size, int flags) = 0;

    // closeWhenHandlerFinished() should be used in almost all cases instead
    // of this, except when writing test code, in which case asyncClose()
    // should be called instead.
    virtual int closePeer() = 0;

    virtual int getHandle() const = 0;
    
    EndpointBase * get_endpoint() { return endpoint_; }

    /** Recycles the current transport with the endpoint, disassociating
        the connection handler in the process.

        All this does is set a flag; the actual work will be done after
        the current handler has finished.
    */
    void recycleWhenHandlerFinished();

    /** Close the current transport once the current handler is finished.
        Can only be called from within a handler. */
    void closeWhenHandlerFinished();

    /** Close the transport asynchronously whenever the current handler
        (if any) has finished running.  Can be called from outsied a
        handler.
    */
    void closeAsync();

    /** Associate the given connection handler with this transport.  Any
        activity on the transport will be handled by the slave.  The
        transport takes ownership of the connection handler and will
        arrange for it to be freed once it is no longer needed.

        Not to be called within a handler.
    */
    void associate(std::shared_ptr<ConnectionHandler> newSlave);

    /** Associates a new handler once the current handler is finished. */
    void
    associateWhenHandlerFinished(std::shared_ptr<ConnectionHandler> newSlave,
                                 const std::string & whereFrom);

    void startReading();
    void stopReading();
    void startWriting();
    void stopWriting();

    /** Schedule a timeout at the given absolute time.  Only one timer is
        available per connection. */
    void scheduleTimerAbsolute(Date timeout,
                               size_t cookie = 0,
                               void (*freecookie) (size_t) = 0);

    /** Schedule a timeout at the given number of seconds from now.  Again,
        only one timer is available per connection.
    */
    void scheduleTimerRelative(double secondsFromNow,
                               size_t cookie = 0,
                               void (*freecookie) (size_t) = 0);
    
    /** Cancel the timer for this connection if it exists. */
    void cancelTimer();

    /** Run the given function from a worker thread in the context of this
        handler.
    */
    void doAsync(const boost::function<void ()> & callback,
                 const std::string & name);


    /** Return the hostname of the connected entity. */
    virtual std::string getPeerName() const = 0;

protected:
    long long lockThread;   ///< For debug for the moment
    const char * lockActivity;

public:
    bool hasSlave() const { return !!slave_; }

    void doError(const std::string & error);

    /** To enable memory leak detection. */
    static long created;
    static long destroyed;

    virtual std::string status() const;

    // DEBUG: record what happens on the socket
    bool debug;

    struct Activity {
        Activity(Date time, std::string what)
            : time(time), what(what)
        {
        }

        Activity(const Json::Value & val)
        {
            fromJson(val);
        }

        Date time;
        std::string what;

        void fromJson(const Json::Value & val);
    };

    struct Activities {
        
        Activities()
        {
        }

        Activities(const std::vector<Activity> & acts)
            : activities(acts)
        {
        }

        ~Activities();

        void add(const std::string & act)
        {
            Guard guard(lock);

            //using namespace std;
            //cerr << Date::now().print(4) << " " << this << " " << act << endl;

            if (activities.size() > 200)
                activities.erase(activities.begin(),
                                 activities.end() - 100);

            activities.push_back(Activity(Date::now(), act));
        }

        void limit(int maxSize)
        {
            Guard guard(lock);

            if (activities.size() > maxSize)
                activities.erase(activities.begin(),
                                 activities.end() - maxSize);
        }

        size_t size() const
        {
            Guard guard(lock);
            return activities.size();
        }

        void clear()
        {
            Guard guard(lock);
            activities.clear();
        }

        std::vector<Activity> takeCopy()
        {
            Guard guard(lock);
            return activities;
        }

        void dump() const;

        Json::Value toJson(int first = 0, int last = -1) const;

        void fromJson(const Json::Value & val);

    private:
        std::vector<Activity> activities;
        
        typedef boost::lock_guard<ML::Spinlock> Guard;
        mutable ML::Spinlock lock;
    };
    
    Activities activities;

    bool debugOn() const { return debug; }

    void addActivity(const std::string & act)
    {
        if (!debug) return;
        //assertLockedByThisThread();
        checkMagic();
        activities.add(act);
    }

    void addActivityS(const char * act)
    {
        if (!debug) return;
        //assertLockedByThisThread();
        checkMagic();
        activities.add(act);
    }

    void addActivity(const char * fmt, ...)
    {
        if (!debug) return;
        //assertLockedByThisThread();
        checkMagic();

        va_list ap;
        va_start(ap, fmt);
        ML::Call_Guard cleanupAp([&] () { va_end(ap); });
        activities.add(ML::vformat(fmt, ap));
    }

    void dumpActivities() const
    {
        activities.dump();
    }

    struct InHandlerGuard {
        InHandlerGuard(TransportBase * t, const char * where)
            : t(0), where(0)
        {
            //using namespace std;
            //cerr << "InHandlerGuard constructur for " << where
            //     << " t = " << t << endl;
            t->checkMagic();
            init(t, where);
        }
        
        InHandlerGuard(TransportBase & t, const char * where)
            : t(0), where(0)
        {
            //using namespace std;
            //cerr << "InHandlerGuard constructur for " << where
            //     << " t = " << &t << endl;
            t.checkMagic();
            init(&t, where);
        }
        
        ~InHandlerGuard()
        {
            //using namespace std;
            //cerr << "InHandlerGuard destructor for " << where
            //     << " t = " << t << endl;

            if (!t) return;
            t->assertLockedByThisThread();
            t->lockThread = 0;
            t->lockActivity = 0;
        }

        void init(TransportBase * t, const char * where)
        {
            using namespace std;
#if 0
            cerr << "InHandlerGuard init for " << where
                 << " with " << t << " lockActivity "
                 << t->lockActivity << endl;
#endif
            this->t = t;
            this->where = where;

            long long me = ACE_OS::thr_self();
            long long locker = t->lockThread;

            for (;;) {
                if (locker == me) {
                    // recursively locked
                    this->t = 0;
                    return;
                }
                if (locker)
                    throw ML::Exception("attempting to enter handler %s: "
                                        "already locked by thread %lld "
                                        " (not my thread %lld) doing %s",
                                        where, locker, me, t->lockActivity);
                if (ML::cmp_xchg(t->lockThread, locker, me)) break;
            }

            t->lockActivity = where;
            t->assertLockedByThisThread();
        }

        void reset()
        {
            t = 0;
        }
        
        TransportBase * t;
        const char * where;
    };

    struct AsyncEntry {
        AsyncEntry()
            : name("none")
        {
        }

        AsyncEntry(const boost::function<void ()> & callback,
                   const std::string & name)
            : callback(callback), name(name), date(Date::now())
        {
        }

        boost::function<void ()> callback;
        std::string name;
        Date date;
    };

    /** A node in the list of things to do asynchronously. */
    struct AsyncNode : public AsyncEntry {
        AsyncNode(const boost::function<void ()> & callback,
                  const std::string & name)
            : AsyncEntry(callback, name), next(0)
        {
        }

        AsyncNode * next;
    };

    bool hasAsync() const
    {
        return asyncHead_;
    }

    /** Head of list of asynchronous entry nodes. */
    AsyncNode * asyncHead_;

    /** Push an asynchronous entry onto the list.  Thread safe and lock
        free.
    */
    void pushAsync(const boost::function<void ()> & fn,
                   const std::string & name);

    /** Return the current async list in order and reset it to empty.
        Thread safe and lock free.
    */
    std::vector<AsyncEntry> popAsync();
    
    bool isZombie() const { return zombie_; }

    bool locked() const
    {
        return lockThread;
    }

    bool lockedByThisThread() const
    {
        return lockThread == ACE_OS::thr_self();
    }

    /** Check that we're locked by this thread; throw otherwise */
    void assertLockedByThisThread() const
    {
        if (!lockedByThisThread())
            throw ML::Exception("should be locked by this thread %lld "
                                "but instead locked by %lld in %s",
                                (long long)ACE_OS::thr_self(),
                                lockThread, lockActivity);
        checkMagic();
    }

    /** Check that we're not locked by another thread; throw otherwise */
    void assertNotLockedByAnotherThread() const
    {
        if (locked() && !lockedByThisThread())
            throw ML::Exception("already locked by thread %lld "
                                " (not my thread %lld) doing %s",
                                lockThread,
                                (long long)ACE_OS::thr_self(),
                                lockActivity);
        checkMagic();
    }

    /** Check that the magic number is correct and that therefore the
        transport is still alive. */
    void checkMagic() const;

    ConnectionHandler & slave()
    {
        if (!slave_) {
            activities.dump();
            throw ML::Exception("transport %p of type %s"
                                "has no associated slave", this,
                                ML::type_name(*this).c_str());
        }
        return *slave_;
    }

    bool hasTimeout() const
    {
        return timeout_.isSet();
    }

    Date nextTimeout() const
    {
        return timeout_.timeout;
    }

    /** Called by something that creates transports to say that it now has
        a connection associated with it.  This causes a bunch of internal
        setup to happen.
    */
    void hasConnection();

private:
    std::shared_ptr<ConnectionHandler> slave_;
    EndpointBase * endpoint_;

    /** If this is non-null, the connection handler will be changed to this
        once the current handler is finished.
    */
    std::shared_ptr<ConnectionHandler> newSlave_;
    std::string newSlaveFrom_;

    /**  If this flag is true, the transport will be recycled once the
         current handler is finished. */
    bool recycle_;

    /** If this flag is true, the transport will be closed once the current
        handler is finished. */
    bool close_;

    /** Current set of flags */
    short flags_;

    /** FD used for epoll when multiplexing events */
    int epollFd_;

    /** FD used for timeouts. */
    int timerFd_;

    /** FD used for events */
    int eventFd_;

    /** Do we have a connection at the moment? */
    bool hasConnection_;

    /** Structure to hold a timeout value. */
    struct Timeout {
        Timeout()
            : timeout(Date::notADate()), timeoutCookie(0), freeTimeoutCookie(0)
        {
        }

        ~Timeout()
        {
            cancel();
        }
        

        Date timeout;               ///< Next timeout for transport
        size_t timeoutCookie;
        void (*freeTimeoutCookie) (size_t);

        bool isSet() const
        {
            return timeout.isADate();
        }

        void cancel()
        {
            timeout = Date::notADate();
            if (freeTimeoutCookie)
                freeTimeoutCookie(timeoutCookie);
            timeoutCookie = 0;
        }

        /** Set the timeout.  Returns true if the timeout is now earlier than
            it was, which means that the event loop should be woken up
            again. */
        bool set(Date date, size_t cookie, void (*freeCookie) (size_t))
        {
            if (!date.isADate())
                throw ML::Exception("Transport::Timeout::set(): not a date");

            Date oldTimeout = timeout;

            cancel();

            timeout = date;
            timeoutCookie = cookie;
            freeTimeoutCookie = freeCookie;

            return (!oldTimeout.isADate() || date < oldTimeout);
        }

    };

    /** Current timeout. */
    Timeout timeout_;

    /** Magic to check that we're still alive. */
    int magic_;

    /** If true, we're a zombie transport (not yet dead, but waiting for
        all references to be removed so that we can die).
    */
    bool zombie_;

    /** Function used act on any flags set and return the appropriate code. */
    int endEventHandler(const char * handler, InHandlerGuard & guard);

    /** Function called when we got an exception from a handler. */
    void handleException(const std::string & handler,
                         const std::exception & exc);
    void handleUnknownException(const std::string & handler);

    /** Disassociate the current connection handler.  This will cause the
        endpoint to be notified that the connection is now free.  It returns
        the current connection handler.

        The connection handler will be freed as soon as the returned shared
        pointer goes out of scope.

        Not to be called from within a handler.
    */
    std::shared_ptr<ConnectionHandler> disassociate();

    /** Perform the actual event handling.  Called from within a worker
        thread that has selected on the epoll instance.

        Returns -1 if the connection should be closed.
    */
    int handleEvents();
};


/*****************************************************************************/
/* SOCKET TRANSPORT                                                          */
/*****************************************************************************/

struct SocketTransport
    : public TransportBase {

    SocketTransport();  // throws
    SocketTransport(EndpointBase * endpoint);

    virtual ~SocketTransport();

    virtual int getHandle() const
    {
        return peer().get_handle();
    }

    virtual std::string getPeerName() const { return peerName_; }

    virtual ssize_t send(const char * buf, size_t len, int flags);
    virtual ssize_t recv(char * buf, size_t buf_size, int flags);
    virtual int closePeer();

    ACE_SOCK_Stream & peer() { return peer_; }
    const ACE_SOCK_Stream & peer() const { return peer_; }

    ACE_SOCK_Stream peer_;
    std::string peerName_;
};



} // namespace Datacratic

#endif /* __rtb__transport_h__ */

