/* carbon_connector.cc
   Jeremy Barnes, 3 August 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "soa/service/carbon_connector.h"
#include "ace/INET_Addr.h"
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include <iostream>
#include "jml/arch/cmp_xchg.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/timers.h"
#include "jml/arch/futex.h"
#include "jml/utils/floating_point.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/exc_assert.h"
#include <boost/tuple/tuple.hpp>
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <poll.h>


using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* MULTI AGGREGATOR                                                          */
/*****************************************************************************/

MultiAggregator::
MultiAggregator()
    : doShutdown(false), doDump(false), dumpInterval(0.0)
{
}

MultiAggregator::
MultiAggregator(const std::string & path,
                const OutputFn & output,
                double dumpInterval,
                std::function<void ()> onStop)
    : doShutdown(false), doDump(false)
{
    open(path, output, dumpInterval, onStop);
}

MultiAggregator::
~MultiAggregator()
{
    shutdown();
}

void
MultiAggregator::
open(const std::string & path,
     const OutputFn & output,
     double dumpInterval,
     std::function<void ()> onStop)
{
    shutdown();

    doShutdown = doDump = false;
    if (dumpInterval < 1.0) {
        dumpInterval = 1.0;
    }
    this->dumpInterval = dumpInterval;
    this->onStop = onStop;

    if (path == "") prefix = "";
    else prefix = path + ".";

    if (output)
        outputFn = output;
    else
        outputFn = [&] (const std::vector<StatReading> & values)
            {
                this->doStat(values);
            };

    dumpingThread.reset
        (new std::thread(std::bind(&MultiAggregator::runDumpingThread,
                                       this)));
}

void
MultiAggregator::
stop()
{
    shutdown();
}

void
MultiAggregator::
doStat(const std::vector<StatReading> & values) const
{
    outputFn(values);
}

StatAggregator * createNewCounter()
{
    return new CounterAggregator();
}

StatAggregator * createNewStableLevel()
{
    return new GaugeAggregator(GaugeAggregator::StableLevel);
}

StatAggregator * createNewLevel()
{
    return new GaugeAggregator(GaugeAggregator::Level);
}

StatAggregator * createNewOutcome(std::vector<int> percentiles)
{
    return new GaugeAggregator(GaugeAggregator::Outcome, std::move(percentiles));
}

void
MultiAggregator::
record(const std::string & stat,
       StatEventType type,
       float value,
       std::initializer_list<int> extra)
{
    switch (type) {
    case ET_HIT:
        recordHit(stat);
        break;
    case ET_COUNT:
        recordCount(stat, value);
        break;
    case ET_STABLE_LEVEL:
        recordStableLevel(stat, value);
        break;
    case ET_LEVEL:
        recordLevel(stat, value);
        break;
    case ET_OUTCOME:
        recordOutcome(stat, value, extra);
        break;
    default:
        cerr << "warning: unknown stat type" << endl;
    }
}

void
MultiAggregator::
recordHit(const std::string & stat)
{
    getAggregator(stat, createNewCounter).record(1.0);
}

void
MultiAggregator::
recordCount(const std::string & stat, float quantity)
{
    getAggregator(stat, createNewCounter).record(quantity);
}

void
MultiAggregator::
recordStableLevel(const std::string & stat, float value)
{
    getAggregator(stat, createNewStableLevel).record(value);
}

void
MultiAggregator::
recordLevel(const std::string & stat, float value)
{
    getAggregator(stat, createNewLevel).record(value);
}
    
void
MultiAggregator::
recordOutcome(const std::string & stat, float value,
              std::vector<int> percentiles)
{
    getAggregator(stat, createNewOutcome, std::move(percentiles)).record(value);
}


void
MultiAggregator::
dump()
{
    {
        std::lock_guard<std::mutex> lock(m);
        doDump = true;
    }
    
    cond.notify_all();
}

void
MultiAggregator::
dumpSync(std::ostream & stream) const
{
    std::unique_lock<Lock> guard(this->lock);

    for (auto & s: stats) {
        auto vals = s.second->read(s.first);
        for (auto v: vals) {
            stream << v.name << ":\t" << v.value << endl;
        }
    }
}

void
MultiAggregator::
shutdown()
{
    if (dumpingThread) {
        if (onPreShutdown) onPreShutdown();
        
        {
            std::lock_guard<std::mutex> lock(m);
            doShutdown = true;
        }

        cond.notify_all();
        
        dumpingThread->join();
        dumpingThread.reset();

        if (onPostShutdown) onPostShutdown();

        if (onStop) onStop();
    }
}

void
MultiAggregator::
runDumpingThread()
{
    size_t current = 0;
    Date nextWakeup = Date::now();

    for (;;) {
        std::unique_lock<std::mutex> lock(m);

        nextWakeup.addSeconds(1.0);
        if (cond.wait_until(lock, nextWakeup.toStd(), [&] { return doShutdown.load(); }))
            break;

        // Get the read lock to extract a list of stats to dump
        vector<Stats::iterator> toDump;
        {
            std::unique_lock<Lock> guard(this->lock);
            toDump.reserve(stats.size());
            for (auto it = stats.begin(), end = stats.end(); it != end;  ++it)
                toDump.push_back(it);
        }

        ++current;
        bool dumpNow = doDump.exchange(false) ||
                       (current % static_cast<size_t>(dumpInterval)) == 0;

        // Now dump them without the lock held. Note that we still need to call
        // read every second even if we're not flushing to carbon.
        for (auto it = toDump.begin(), end = toDump.end(); it != end;  ++it) {

            try {
                auto stat = (*it)->second->read((*it)->first);

                // Hack: ensures that all timestamps are consistent and that we
                // will not have any gaps within carbon.
                for (auto& s : stat) s.timestamp = nextWakeup;

                if (dumpNow) doStat(std::move(stat));
            } catch (const std::exception & exc) {
                cerr << "error writing stat: " << exc.what() << endl;
            }
        }
    }
}


/*****************************************************************************/
/* CARBON CONNECTOR                                                        */
/*****************************************************************************/

CarbonConnector::
CarbonConnector()
{
}

CarbonConnector::
CarbonConnector(const std::string & carbonAddr,
                const std::string & path,
                double dumpInterval,
                std::function<void ()> onStop)
{
    open(carbonAddr, path, dumpInterval, onStop);
}

CarbonConnector::
CarbonConnector(const std::vector<std::string> & carbonAddrs,
                const std::string & path,
                double dumpInterval,
                std::function<void ()> onStop)
{
    open(carbonAddrs, path,dumpInterval,  onStop);
}

CarbonConnector::
~CarbonConnector()
{
    doShutdown();
}

void
CarbonConnector::
open(const std::string & carbonAddr,
     const std::string & path,
     double dumpInterval,
     std::function<void ()> onStop)
{
    return open(vector<string>({carbonAddr}), path, dumpInterval, onStop);
}

void
CarbonConnector::
open(const std::vector<std::string> & carbonAddrs,
     const std::string & path,
     double dumpInterval,
     std::function<void ()> onStop)
{
    stop();

    int numConnections = 0;

    connections.clear();
    for (unsigned i = 0;  i < carbonAddrs.size();  ++i) {
        connections.push_back(std::make_shared<Connection>
                                    (carbonAddrs[i]));
        string error = connections.back()->connect();
        if (connections.back()->fd == -1) {
            cerr << "error connecting to Carbon at " << carbonAddrs[i]
                 << ": " << error << endl;
        }
        else ++numConnections;
    }

    if (numConnections == 0)
        throw ML::Exception("unable to connect to any Carbon instances");

    this->onPostShutdown = std::bind(&CarbonConnector::doShutdown, this);

    MultiAggregator::open(path, OutputFn(), dumpInterval, onStop);
}

void
CarbonConnector::
doShutdown()
{
    stop();
    connections.clear();
}

void
CarbonConnector::
doStat(const std::vector<StatReading> & values) const
{
    if (connections.empty())
        return;

    std::string message;

    for (unsigned i = 0;  i < values.size();  ++i) {
        message += ML::format("%s%s %g %lld\n",
                              prefix.c_str(), values[i].name.c_str(),
                              values[i].value,
                              (unsigned long long)
                              values[i].timestamp.secondsSinceEpoch());
    }

    for (unsigned i = 0;  i < connections.size();  ++i)
        connections[i]->send(message);
}

CarbonConnector::Connection::
~Connection()
{
    close();

    if (reconnectionThread) {
        shutdown = 1;
        futex_wake(shutdown);
        reconnectionThread->join();
        reconnectionThread.reset();
    }
}

void
CarbonConnector::Connection::
close()
{
    if (fd != -1)
        ::close(fd);
    fd = -1;
}

std::string
CarbonConnector::Connection::
connect()
{
    if (fd != -1)
        throw ML::Exception("error connecting");

    ip = ACE_INET_Addr(addr.c_str());

    cerr << "connecting to Carbon at "
         << ip.get_host_addr() << ":" << ip.get_port_number()
         << " (" << ip.get_host_name() << ")" << endl;

    int tmpFd = socket(AF_INET, SOCK_STREAM, 0);
    int res = ::connect(tmpFd,
                        (sockaddr *)ip.get_addr(),
                        ip.get_addr_size());
    
    int saved_errno = errno;

    if (res == -1) {
        ::close(tmpFd);
        return ML::format("connect to carbon at %s:%d (%s): %s",
                          ip.get_host_addr(),
                          ip.get_port_number(),
                          ip.get_host_name(),
                          strerror(saved_errno));
    }

    fd = tmpFd;

    return "";
}

void
CarbonConnector::Connection::
send(const std::string & message)
{
    //cerr << "STAT: " << message << endl;
    //return;
    if (message.empty())
        return;

    //cerr << "sending to " << addr << " on " << fd << " " << message << endl;

    if (fd == -1) {
        if (reconnectionThreadActive) return;
        throw ML::Exception("send with fd -1 and no thread active");
    }
    
    size_t done = 0;

    for (;;) {
        int sendRes = ::send(fd, message.c_str() + done,
                             message.size() - done,
                             MSG_DONTWAIT | MSG_NOSIGNAL);
        
        if (sendRes > 0) {
            done += sendRes;
            if (done == message.size())
                return;  // done; normal case
            else if (done > message.size())
                throw ML::Exception("logic error sending message to Carbon");
            else continue;  // do the rest of the message
        }
        
        if (sendRes != -1)
            throw ML::Exception("invalid return code from send");
        
        // Error handling
        if (errno == EINTR)
            continue;  // retry
        else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // Would block (something that we don't want).  Select on the
            // socket for the timeout before giving up.
            struct pollfd events[] = {
                { fd, POLLOUT | POLLERR | POLLHUP | POLLNVAL, 0 }
            };

            int res = poll(events, 1, 500 /* 500ms max timeout */);
            if (res == 1 && events[0].revents == POLLOUT) {
                // Ready to send
                continue;  // we can now send it
            }
            else if (res == -1) {
                // error in epoll call
                int saved_errno = errno;
                cerr << "error on epoll with CarbonConnector " << addr
                     << ": " << strerror(saved_errno) << endl;
            }
            else if (res == 0) {
                // nothing ready... must be a timeout
                cerr << "timeout sending to CarbonConnector at " << addr
                     << endl;
            }
            else if (res == 1 && events[0].revents & ~POLLOUT) {
                // Disconnection or error... need to reconnect
                cerr << "disconnection sending to CarbonConnector at " << addr
                     << endl;
            }
            else {
                // Logic error; we should never get here
                throw ML::Exception("logic error in carbon connector");
            }
        } else {
            // Error in sending
            int saved_errno = errno;
            cerr << "error sending to CarbonConnector " << addr
                 << ": " << strerror(saved_errno) << endl;
        }
        break;
    }    

    reconnect();
}

void
CarbonConnector::Connection::
reconnect()
{
    close();

    cerr << "reconnecting to " << addr << endl;

    if (reconnectionThreadJoinable && reconnectionThread) {
        reconnectionThread->join();
        reconnectionThread.reset();
    }

    reconnectionThreadActive = false;
    reconnectionThreadJoinable = false;
    
    reconnectionThread
        = std::make_shared<std::thread>
            (std::bind(&Connection::runReconnectionThread, this));

}

void
CarbonConnector::Connection::
runReconnectionThread()
{
    cerr << "started reconnection thread" << endl;

    reconnectionThreadActive = true;

    // Close the current connection
    if (fd != -1)
        close();

    double meanWaitTime = 0.5;  // half a second

    while (!shutdown) {
        string error = connect();
        if (fd != -1) break;

        cerr << "error reconnecting to " << addr << ": " << error
             << endl;

        double r = (random() % 10001) / 10000.0;
        double waitTime = meanWaitTime * (0.5 + r);

        if (meanWaitTime < 8.0)
            meanWaitTime *= 2;
        
        // Wait for the given time before we attempt reconnection
        // again.
        futex_wait(shutdown, 0, waitTime);
    }

    reconnectionThreadActive = false;
    reconnectionThreadJoinable = true;
}

} // namespace Datacratic
