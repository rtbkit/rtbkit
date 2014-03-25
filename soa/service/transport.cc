/* transport.cc
   Jeremy Barnes, 24 February 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Information on the transport.
*/

#include "transport.h"

#include "soa/service//http_endpoint.h"
#include "jml/arch/cmp_xchg.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/format.h"
#include "jml/arch/exception.h"
#include "jml/arch/demangle.h"
#include "jml/arch/backtrace.h"
#include "jml/utils/environment.h"
#include <iostream>
#include <sys/epoll.h>
#include <sys/timerfd.h>
#include <sys/eventfd.h>
#include <poll.h>


using namespace std;
using namespace ML;


namespace Datacratic {

boost::function<void (const char *, float)> onLatencyEvent;

ML::Env_Option<bool> DEBUG_TRANSPORTS("DEBUG_TRANSPORTS", false);


/*****************************************************************************/
/* TRANSPORT BASE                                                            */
/*****************************************************************************/

long TransportBase::created = 0;
long TransportBase::destroyed = 0;


TransportBase::
TransportBase()
{
    throw Exception("TransportBase constructor requires an endpoint");
}

TransportBase::
TransportBase(EndpointBase * endpoint)
    : lockThread(0), lockActivity(0), debug(DEBUG_TRANSPORTS),
      asyncHead_(0),
      endpoint_(endpoint),
      recycle_(0), close_(0), flags_(0),
      hasConnection_(false), zombie_(false)
{
    atomic_add(created, 1);

    if (!endpoint)
        throw ML::Exception("transport requires an endpoint");

    magic_ = 0x12345678;

    addActivityS("created");

    epollFd_ = epoll_create(1024);
    if (epollFd_ == -1)
        throw ML::Exception(errno, "couldn't create epoll fd");
    timerFd_ = timerfd_create(CLOCK_REALTIME, TFD_NONBLOCK);
    if (timerFd_ == -1)
        throw ML::Exception(errno, "timer FD couldn't be created");
    eventFd_ = eventfd(0, EFD_NONBLOCK);
    if (eventFd_ == -1)
        throw ML::Exception(errno, "event FD couldn't be created");

    /* Add all of these other FDs to the event FD */
    struct epoll_event data;
    data.data.u64 = 0;
    data.data.fd = timerFd_;
    data.events = EPOLLIN;
    int res = epoll_ctl(epollFd_, EPOLL_CTL_ADD, timerFd_, &data);
    if (res == -1)
        throw ML::Exception(errno, "epoll_ctl ADD timerFd");
    data.data.fd = eventFd_;
    res = epoll_ctl(epollFd_, EPOLL_CTL_ADD, eventFd_, &data);
    if (res == -1)
        throw ML::Exception(errno, "epoll_ctl ADD eventFd");

    //dumpActivities();

}

TransportBase::
~TransportBase()
{
    int res = close(epollFd_);
    if (res == -1)
        cerr << "closing epoll fd: " << strerror(errno) << endl;
    res = close(timerFd_);
    if (res == -1)
        cerr << "closing timer fd: " << strerror(errno) << endl;
    res = close(eventFd_);
    if (res == -1)
        cerr << "closing event fd: " << strerror(errno) << endl;

    popAsync();
    assertNotLockedByAnotherThread();
    checkMagic();
    atomic_add(destroyed, 1);
    magic_ = 0;
    lockThread = 1;
    lockActivity = "DESTROYED";
}

void
TransportBase::
checkMagic() const
{
    if (magic_ != 0x12345678) {
        cerr << "dead transport: " << endl;
        //activities.dump();
        throw Exception("attempt to access dead transport %p: magic %d",
                        this, magic_);
    }
}

int
TransportBase::
handleInput()
{
    if (close_) return -1;

    InHandlerGuard guard(this, "input");

    if (close_) return -1;

    addActivityS("input");

    try {
        slave().handleInput();
    } catch (const std::exception & exc) {
        handleException("input", exc);
    } catch (...) {
        handleUnknownException("input");
    }

    return endEventHandler("input", guard);
}

int
TransportBase::
handleOutput()
{
    if (close_) return -1;

    InHandlerGuard guard(this, "output");

    if (close_) return -1;

    addActivityS("output");

    try {
        slave().handleOutput();
    } catch (const std::exception & exc) {
        handleException("output", exc);
    } catch (...) {
        handleUnknownException("output");
    }

    return endEventHandler("output", guard);
}

int
TransportBase::
handlePeerShutdown()
{
    if (close_) return -1;

    InHandlerGuard guard(this, "peerShutdown");

    if (close_) return -1;

    addActivityS("peerShutdown");

    try {
        slave().handlePeerShutdown();
    } catch (const std::exception & exc) {
        handleException("peerShutdown", exc);
    } catch (...) {
        handleUnknownException("peerShutdown");
    }

    return endEventHandler("peerShutdown", guard);
}

int
TransportBase::
handleTimeout()
{
    if (close_) return -1;

    Date before = Date::now();

    InHandlerGuard guard(this, "timeout");

    if (close_) return -1;

    Date afterLock = Date::now();

    addActivity("timeout %zd in %.1fms", timeout_.timeoutCookie,
                afterLock.secondsSince(before) * 1000);

    try {
        slave().handleTimeout(timeout_.timeout, timeout_.timeoutCookie);

        //Date afterHandler = Date::now();

    } catch (const std::exception & exc) {
        handleException("timeout", exc);
    } catch (...) {
        handleUnknownException("timeout");
    }

    return endEventHandler("timeout", guard);
}

int
TransportBase::
handleError(const std::string & error)
{
    if (close_) return -1;

    InHandlerGuard guard(this, "error");

    if (close_) return -1;

    addActivityS("error");

    try {
        slave().handleError(error);
    } catch (const std::exception & exc) {
        handleException("error", exc);
    } catch (...) {
        handleUnknownException("error");
    }

    return endEventHandler("error", guard);
}

int
TransportBase::
handleAsync(const boost::function<void ()> & callback, const char * name,
            Date dateSet)
{
    Date now = Date::now();

    double delay = now.secondsSince(dateSet);

    if (onLatencyEvent)
        onLatencyEvent("asyncCallback", delay);

    static Date lastMessage;
    static int messagesSinceLastMessage = 0;

    if (delay > 0.01) {
        if (lastMessage.secondsSince(now) > 0.5) {
            lastMessage = now;
            cerr << "big delay on async callback " << name << ": "
                 << delay * 1000 << "ms"
                 << endl;
            if (messagesSinceLastMessage > 0)
                cerr << "also " << messagesSinceLastMessage << endl;

            cerr << "dateSet: " << dateSet.print(5) << endl;
            cerr << "now: " << now.print(5) << endl;
            activities.dump();
            //* (char *)0 = 0;  // cause segfault for gdb
            messagesSinceLastMessage = 0;
        }
        else ++messagesSinceLastMessage;
    }
    
    if (close_) return -1;

    addActivity("handleAsync: %s", name);

    InHandlerGuard guard(this, name);

    if (close_) return -1;

    addActivityS(name);

    try {
        callback();
    } catch (const std::exception & exc) {
        handleException(name, exc);
    } catch (...) {
        handleUnknownException(name);
    }

    return endEventHandler(name, guard);
}

void
TransportBase::
associate(std::shared_ptr<ConnectionHandler> newSlave)
{
    addActivity("associate with " + newSlave->status());
    
    //assertLockedByThisThread();
    assertNotLockedByAnotherThread();

    if (slave_) {
        assertLockedByThisThread();

        if (slave_.get() == newSlave.get())
            throw Exception("re-associating the same slave of type "
                            + ML::type_name(*slave_));
        slave().onDisassociate();
    }

    slave_ = newSlave;
    slave().setTransport(this);

    auto finishAssociate = [=] ()
        {
            this->slave().onGotTransport();
        };

    /* If we're already locked, then finish the association here.  Otherwise,
       do it asynchronously so that we can do it in the handler context.
    */
    if (lockedByThisThread()) {
        InHandlerGuard guard(this, "associate");
        finishAssociate();
        endEventHandler("associate", guard);
    }
    else {
        doAsync(finishAssociate, "associate");
    }
    
    addActivityS("associate done");
}

std::shared_ptr<ConnectionHandler>
TransportBase::
disassociate()
{
    addActivityS("disassociate");

    if (!slave_)
        throw Exception("TransportBase::disassociate(): no slave");

    cancelTimer();

    slave().onDisassociate();
    std::shared_ptr<ConnectionHandler> result = slave_;
    slave_.reset();
    return result;
}

void
TransportBase::
recycleWhenHandlerFinished()
{
    if (!lockedByThisThread())
        throw Exception("not in handler");

    addActivityS("recycleWhenHandlerFinished");

    this->recycle_ = true;
}

void
TransportBase::
closeWhenHandlerFinished()
{
    //backtrace();

    if (!lockedByThisThread())
        throw Exception("not in handler");

    addActivityS("closeWhenHandlerFinished");

    this->close_ = true;
    this->recycle_ = false;
}

void
TransportBase::
closeAsync()
{
    auto doClose = [=] () { this->closeWhenHandlerFinished(); };
    doAsync(doClose, "closeAsync");
}

void
TransportBase::
associateWhenHandlerFinished(std::shared_ptr<ConnectionHandler> slave,
                             const std::string & whereFrom)
{
    if (!slave)
        throw Exception("associateWhenHandlerFinished(): null slave");

    if (newSlave_)
        throw Exception("associateWhenHandlerFinished(): attempt to "
                        "double replace slave %s from %s "
                        "with new slave %s from %s",
                        newSlaveFrom_.c_str(),
                        type_name(*slave_).c_str(),
                        whereFrom.c_str(),
                        type_name(*slave).c_str());

    if (!lockedByThisThread()) {
        throw Exception("associateWhenHandlerFinished(): needs to be in "
                        "handler");
        //associate(slave);
    }
    else {
        newSlave_ = slave;
        newSlaveFrom_ = whereFrom;
    }
}

void
TransportBase::
handleException(const std::string & handler, const std::exception & exc)
{
    addActivityS("exception");

    if (slave_)
        slave().onHandlerException(handler, exc);
}

void
TransportBase::
handleUnknownException(const std::string & handler)
{
    handleException(handler, ML::Exception("unknown exception"));
}

int
TransportBase::
endEventHandler(const char * handler, InHandlerGuard & guard)
{
    addActivity("endHandler %s with close %d, recycle %d, "
                "newHandler %p", handler, close_, recycle_,
                newSlave_.get());

    if (close_) {
        try {
            if (hasSlave()) disassociate();
        } catch (const std::exception & exc) {
            cerr << "error: disassociate() threw exception: "
                 << exc.what();
        }

        return -1;  // handle_close will be called once all out of the way
    }
    else if (recycle_) {
        try {
            if (hasSlave()) disassociate();
            //cerr << "done disassociate for " << handler << " for "
            //     << this << " with recycle" << endl;
        } catch (const std::exception & exc) {
            cerr << "error: disassociate() threw exception: "
                 << exc.what();
            return -1;
        }
        endpoint_->notifyRecycleTransport(shared_from_this());
        recycle_ = false;
        return 0;
    }
    else if (newSlave_) {
        auto saved = newSlave_;
        newSlave_.reset();
        associate(saved);
    }

    return 0;
}

void
TransportBase::
doError(const std::string & error)
{
    slave().doError(error);
}

struct TransportTimer {
    TransportTimer(TransportBase * transport,
                   const char * event,
                   double maxTime = 0.0)
        : start(maxTime == 0.0 ? Date() : Date::now()),
          transport(transport), event(event),
          maxTime(maxTime)
    {
        if (maxTime != 0.0)
            statusBefore = transport->status();
    }

    ~TransportTimer()
    {
        if (maxTime == 0.0) return;
        double elapsed = Date::now().secondsSince(start);
        if (elapsed > maxTime) {
            cerr << "transport operation " << event << " took "
                 << elapsed * 1000 << "ms with status "
                 << transport->status() << " before "
                 << statusBefore << endl;
        }
    }

    Date start;
    TransportBase * transport;
    const char * event;
    double maxTime;
    std::string statusBefore;
};

int pollFlagsToEpoll(int flags)
{
    int result = 0;
    if (flags & POLLIN) result |= EPOLLIN;
    if (flags & POLLOUT) result |= EPOLLOUT;
    if (flags & POLLPRI) result |= EPOLLPRI;
    if (flags & POLLERR) result |= EPOLLERR;
    if (flags & POLLHUP) result |= EPOLLHUP;
    if (flags & POLLRDHUP) result |= EPOLLRDHUP;
    return result;
}

void
TransportBase::
hasConnection()
{
    addActivity("hasConnection; handle %d; epoll fd %d",
                getHandle(), epollFd_);

    if (getHandle() < 0)
        throw ML::Exception("hasConnection without a connection");

    struct epoll_event data;
    data.data.u64 = 0;
    data.data.fd = getHandle();
    data.events = pollFlagsToEpoll(flags_);
    int res = epoll_ctl(epollFd_, EPOLL_CTL_ADD, getHandle(), &data);
    if (res == -1)
        throw ML::Exception(errno, "epoll_ctl ADD getHandle()");

    hasConnection_ = true;

    //cerr << "transport " << getHandle() << " "
    //     << status() << " has a connection" << endl;
}

std::string
epollFlags(int mask)
{
    return ML::format("%s%s%s%s%s%s%s%s",
                      (mask & EPOLLIN ? "I" : ""),
                      (mask & EPOLLOUT ? "O" : ""),
                      (mask & EPOLLPRI ? "P" : ""),
                      (mask & EPOLLERR ? "E" : ""),
                      (mask & EPOLLHUP ? "H" : ""),
                      (mask & EPOLLRDHUP ? "R" : ""),
                      (mask & EPOLLET ? "E" : ""),
                      (mask & EPOLLONESHOT ? "1" : ""));
}

std::string
pollFlags(int mask)
{
    return ML::format("%s%s%s%s%s%s%s",
                      (mask & POLLIN ? "I" : ""),
                      (mask & POLLOUT ? "O" : ""),
                      (mask & POLLPRI ? "P" : ""),
                      (mask & POLLERR ? "E" : ""),
                      (mask & POLLHUP ? "H" : ""),
                      (mask & POLLRDHUP ? "R" : ""),
                      (mask & POLLNVAL ? "N" : ""));
}

int
TransportBase::
handleEvents()
{
    int rc = 0;

    addActivity("handleEvents");
        
    while (!isZombie() && rc != -1) {
        struct pollfd items[3] = {
            { eventFd_, POLLIN, 0 },
            { timerFd_, POLLIN, 0 },
            { getHandle(), flags_, 0 }
        };

        int res = poll(items, 3, 0);
        
#if 0
        cerr << "handleevents for " << getHandle() << " " << status()
             << " got " << res << " events" << " and has async "
             << hasAsync() << " " << Date::now().print(4) << endl;
        cerr << "flags_ = " << pollFlags(flags_) << endl;

        for (unsigned i = 0;  i < 3;  ++i) {
            if (!items[i].revents) continue;
            string fdname;
            if (i == 0) fdname = "wakeup";
            else if (i == 1) fdname = "timer";
            else if (i == 2) fdname = "connection";
            
            int mask = items[i].revents;

            cerr << "    " << fdname << " has flags "
                 << pollFlags(mask) << endl;
        }
#endif

        if (res == 0 && !hasAsync()) break;
        
        if (items[0].revents) {
            // Clear the wakeup if there was one
            uint64_t nevents;
            int res = eventfd_read(eventFd_, &nevents);
            if (res == -1)
                throw ML::Exception(errno, "eventfd_read");
            //cerr << "    got wakeup" << endl;
        }
        if (rc != -1 && items[2].revents & POLLERR) {
            // Connection finished or has an error; check which one
            int error = 0;
            socklen_t error_len = sizeof(int);
            int res = getsockopt(getHandle(), SOL_SOCKET, SO_ERROR,
                                 &error, &error_len);
            if (res == -1 || error_len != sizeof(int))
                throw ML::Exception(errno, "getsockopt(SO_ERROR)");
            
            {
                TransportTimer timer(this, "error");
                rc = handleError(strerror(error));
            }
        }
        if (rc != -1 && items[1].revents & POLLIN) {
            // Timeout...
            
            // First read the number of timeouts to reset the count
            uint64_t numTimeouts;
            int res = read(timerFd_, &numTimeouts, 8);
            if (res == -1)
                throw ML::Exception(errno, "reading from timerfd");
            if (res != 8)
                throw ML::Exception("read wrong num bytes from timerfd");
            
            if (timeout_.isSet()) {
                // Now call the handler
                TransportTimer timer(this, "timeout");
                rc = handleTimeout();
            }

            //cerr << "    got timeout" << endl;
        }
        if (rc != -1
            && (items[2].revents & POLLIN)
            && (flags_ & POLLIN)) {
            //cerr << "    got input" << endl;
            TransportTimer timer(this, "input");
            rc = handleInput();
        }
        if (rc != -1
            && (items[2].revents & POLLOUT)
            && (flags_ & POLLOUT)) {
            //cerr << "    got output" << endl;
            TransportTimer timer(this, "output");
            rc = handleOutput();
        }
        if (rc != -1
            && (items[2].revents & POLLRDHUP)
            && (flags_ & POLLRDHUP)) {
            //cerr << "    got output" << endl;
            TransportTimer timer(this, "peerShutdown");
            rc = handlePeerShutdown();
        }

        if (hasAsync() && rc != -1) {
            std::vector<AsyncEntry> async
                = popAsync();
            
            for (unsigned i = 0;  i < async.size();  ++i) {
                //cerr << "    got async " << async[i].name << endl;
                TransportTimer timer(this, async[i].name.c_str());
                rc = handleAsync(async[i].callback,
                                 async[i].name.c_str(),
                                 async[i].date);
                if (rc == -1) break;
            }
        }
    }

    //cerr << "finished handling events for " << this << " with rc " << rc
    //     << endl;

    if (rc == -1) {

        //cerr << "    closing connection" << endl;
        std::shared_ptr<TransportBase> tr
            = shared_from_this();
    
        {
            InHandlerGuard guard(this, "close");

            addActivityS("close");

            if (hasSlave()) {
                cerr << "    slave" << endl;
                slave().onCleanup();
                slave_.reset();
            }
        }

        if (hasConnection_) {
            int res = epoll_ctl(epollFd_, EPOLL_CTL_DEL, getHandle(), 0);
            if (res == -1)
                throw ML::Exception("TransportBase::close(): epoll_ctl DEL %d: %s",
                                    getHandle(), strerror(errno));
        }
        cancelTimer();

        //closePeer();
        
        endpoint_->notifyCloseTransport(tr);
    }
    else if (hasConnection_) {
        // Change the epoll event set

        struct epoll_event data;
        data.data.u64 = 0;
        data.data.fd = getHandle();
        data.events = pollFlagsToEpoll(flags_);
        
        //cerr << "setting flags to " << epollFlags(flags_) << endl;

        int res = epoll_ctl(epollFd_, EPOLL_CTL_MOD, getHandle(), &data);
        
        // TODO: no setting of mask if not necessary
        if (res == -1)
            throw ML::Exception(errno, "TransportBase::close(): epoll_ctl MOD");
    }

    return rc;
}

std::string
TransportBase::
status() const
{
    string result = format("Transport %p", this)
        + " of type " + ML::type_name(*this);
    if (hasSlave())
        result += " with slave " + slave_->status();
    else result += " with no slave";
    return result;
}

void
TransportBase::
startReading()
{
    //cerr << "starting reading " << this << endl;
    assertLockedByThisThread();
    flags_ |= POLLIN;
}

void
TransportBase::
stopReading()
{
    //cerr << "stopping reading " << this << endl;
    assertLockedByThisThread();
    flags_ &= ~POLLIN;
}

void
TransportBase::
startWriting()
{
    //cerr << "starting writing " << this << endl;
    assertLockedByThisThread();
    flags_ |= POLLOUT;
}

void
TransportBase::
stopWriting()
{
    //cerr << "stopping writing " << this << endl;
    assertLockedByThisThread();
    flags_ &= ~POLLOUT;
}

void
TransportBase::
scheduleTimerAbsolute(Date timeout, size_t cookie,
                      void (*freecookie) (size_t))
{
    timeout_.set(timeout, cookie, freecookie);
    long seconds = timeout.wholeSecondsSinceEpoch();
    long nanoseconds = timeout.fractionalSeconds() * 1000000000.0;
    itimerspec spec = { { 0, 0 }, { seconds, nanoseconds } };
    int res = timerfd_settime(timerFd_, TFD_TIMER_ABSTIME, &spec, 0);
    if (res == -1)
        throw ML::Exception(errno, "timerfd_settime absolute");
}

void 
TransportBase::
scheduleTimerRelative(double secondsFromNow,
                      size_t cookie,
                      void (*freecookie) (size_t))
{
    assertLockedByThisThread();
    
    if (secondsFromNow < 0)
        throw ML::Exception("attempting to schedule timer in the past: %f",
                            secondsFromNow);

    timeout_.set(Date::now().plusSeconds(secondsFromNow),
                 cookie, freecookie);
    long seconds = secondsFromNow;
    long nanoseconds = 1000000000.0 * (secondsFromNow - seconds);
    itimerspec spec = { { 0, 0 }, { seconds, nanoseconds } };
    int res = timerfd_settime(timerFd_, 0, &spec, 0);
    if (res == -1)
        throw ML::Exception(errno, "timerfd_settime relative");
}
    
void
TransportBase::
cancelTimer()
{
    timeout_.cancel();

    itimerspec spec = { { 0, 0 }, { 0, 0 } };
    int res = timerfd_settime(timerFd_, 0, &spec, 0);
    if (res == -1)
        throw ML::Exception(errno, "timerfd_settime");
}

void
TransportBase::
doAsync(const boost::function<void ()> & callback, const std::string & name)
{
    addActivity("doAsync: %s lockedByThisThread %d", name.c_str(),
                lockedByThisThread());
    pushAsync(callback, name);
}

void
TransportBase::
pushAsync(const boost::function<void ()> & fn, const std::string & name)
{
    std::auto_ptr<AsyncNode> node(new AsyncNode(fn, name));
    
    AsyncNode * current = asyncHead_;
    
    for (;;) {
        node->next = current;
        if (ML::cmp_xchg(asyncHead_, current, node.get())) break;
    }

    node.release();
    
    if (!lockedByThisThread())
        eventfd_write(eventFd_, 1);
}

/** Return the current async list in order and reset it to empty.  Thread safe and lock
    free. */
std::vector<TransportBase::AsyncEntry>
TransportBase::
popAsync()
{
    AsyncNode * current = asyncHead_;
    
    for (;;) {
        if (ML::cmp_xchg(asyncHead_, current, (AsyncNode *)0)) break;
    }
    
    std::vector<AsyncEntry> result;
    
    for (; current; ) {
        result.push_back(*current);
        auto next = current->next;
        delete current;
        current = next;
    }
    
    // TODO: should just iterate in reverse order...
    std::reverse(result.begin(), result.end());
    
    return result;
}

TransportBase::Activities::
~Activities()
{
    Guard guard(lock);
    activities.clear();
}

void
TransportBase::Activities::
dump() const
{
    Guard guard(lock);
    if (activities.empty()) return;
    Date firstTime = activities.front().time, lastTime = firstTime;
    for (unsigned i = 0;  i < activities.size();  ++i) {
        Date time = activities[i].time;
        cerr << format("%3d %s %7.3f %7.3f %s\n",
                       i,
                       time.print(4).c_str(),
                       time.secondsSince(firstTime),
                       time.secondsSince(lastTime),
                       activities[i].what.c_str());
        lastTime = time;
    }
}

Json::Value
TransportBase::Activities::
toJson(int first, int last) const
{
    Guard guard(lock);
    if (last == -1) last = size();

    if (first < 0 || last < first || last > size())
        throw Exception("Activities::toJson(): "
                        "range %d-%d incompatible with 0-%d",
                        first, last, (int)size());

    Json::Value result;

    if (first == last) return result;

    Date firstTime = activities[first].time, lastTime = firstTime;
    for (unsigned i = first;  i < last;  ++i) {
        Date time = activities[i].time;
        result[i - first][0u] = time.print(4);
        result[i - first][1u] = time.secondsSince(firstTime);
        result[i - first][2u] = time.secondsSince(lastTime);
        result[i - first][3u] = activities[i].what;
        lastTime = time;
    }

    return result;
}

void
TransportBase::Activity::
fromJson(const Json::Value & val)
{
    if (val.size() != 4)
        throw Exception("not an activity in JSON: %s", val.toString().c_str());
    
    time = Date(val[0u]);
    what = val[3].asString();
}

void
TransportBase::Activities::
fromJson(const Json::Value & val)
{
    vector<Activity> activities;

    for (unsigned i = 0;  i < val.size();  ++i) {
        activities.push_back(val[i]);
    }

    activities.swap(activities);
}


/*****************************************************************************/
/* SOCKET TRANSPORT                                                          */
/*****************************************************************************/

SocketTransport::
SocketTransport()
{
}

SocketTransport::
SocketTransport(EndpointBase * endpoint)
    : TransportBase(endpoint)
{
}

SocketTransport::
~SocketTransport()
{
    if (getHandle() != -1) {
        cerr << "warning: closing TransportBase " << this
             << " of type " << status() << "with open socket "
             << getHandle() << endl;
        backtrace();
    }
}

ssize_t
SocketTransport::
send(const char * buf, size_t len, int flags)
{
    return peer().send(buf, len, flags);
}
   
ssize_t
SocketTransport::
recv(char * buf, size_t buf_size, int flags)
{
    return peer().recv(buf, buf_size, flags);
}

int
SocketTransport::
closePeer()
{
    addActivityS("closePeer");
    return peer().close();
}

} // namespace Datacratic

