/* router_base.cc
   Jeremy Barnes, 29 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Base class for router.
*/

#include "router_base.h"
#include "rtbkit/core/banker/null_banker.h"


using namespace std;
using namespace ML;


namespace RTBKIT {

/*****************************************************************************/
/* AUCTION DEBUG INFO                                                        */
/*****************************************************************************/

void
AuctionDebugInfo::
addAuctionEvent(Date timestamp, std::string type,
                const std::vector<std::string> & args)
{
    Message message;
    message.timestamp = timestamp;
    message.type = type;
    //message.args = args;
    messages.push_back(message);
}

void
AuctionDebugInfo::
addSpotEvent(const Id & spot, Date timestamp, std::string type,
             const std::vector<std::string> & args)
{
    Message message;
    message.spot = spot;
    message.timestamp = timestamp;
    message.type = type;
    //message.args = args;
    messages.push_back(message);
}

void
AuctionDebugInfo::
dumpAuction() const
{
    for (unsigned i = 0;  i < messages.size();  ++i) {
        auto & m = messages[i];
        cerr << m.timestamp.print(6) << " " << m.spot << " " << m.type << endl;
    }
}

void
AuctionDebugInfo::
dumpSpot(Id spot) const
{
    dumpAuction();  // TODO
}

/*****************************************************************************/
/* ROUTER SHARED                                                             */
/*****************************************************************************/

RouterShared::
RouterShared(std::shared_ptr<zmq::context_t> zmqContext)
    : logger(zmqContext),
      doDebug(false),
      numAuctions(0), numBids(0), numNonEmptyBids(0),
      numAuctionsWithBid(0), numWins(0), numLosses(0),
      numImpressions(0), numClicks(0), numNoPotentialBidders(0),
      numNoBidders(0),
      simulationMode_(false)
{
}


/*****************************************************************************/
/* ROUTER BASE                                                               */
/*****************************************************************************/

void
RouterServiceBase::
throwException(const std::string & key, const std::string & fmt, ...)
{
    recordHit("error.exception");
    recordHit("error.exception.%s", key);

    string message;
    va_list ap;
    va_start(ap, fmt);
    try {
        message = vformat(fmt.c_str(), ap);
        va_end(ap);
    }
    catch (...) {
        va_end(ap);
        throw;
    }

    logRouterError("exception", key, message);
    throw ML::Exception("Router Exception: " + key + ": " + message);
}

void
RouterServiceBase::
debugAuctionImpl(const Id & auction, const std::string & type,
                 const std::vector<std::string> & args)
{
    Date now = Date::now();
    boost::unique_lock<ML::Spinlock> guard(shared->debugLock);
    AuctionDebugInfo & entry
        = shared->debugInfo.access(auction, now.plusSeconds(30.0));

    entry.addAuctionEvent(now, type, args);
}

void
RouterServiceBase::
debugSpotImpl(const Id & auction, const Id & spot, const std::string & type,
              const std::vector<std::string> & args)
{
    Date now = Date::now();
    boost::unique_lock<ML::Spinlock> guard(shared->debugLock);
    AuctionDebugInfo & entry
        = shared->debugInfo.access(auction, now.plusSeconds(30.0));

    entry.addSpotEvent(spot, now, type, args);
}

void
RouterServiceBase::
expireDebugInfo()
{
    boost::unique_lock<ML::Spinlock> guard(shared->debugLock);
    shared->debugInfo.expire();
}

void
RouterServiceBase::
dumpAuction(const Id & auction) const
{
    boost::unique_lock<ML::Spinlock> guard(shared->debugLock);
    auto it = shared->debugInfo.find(auction);
    if (it == shared->debugInfo.end()) {
        //cerr << "*** unknown auction " << auction << " in "
        //     << shared->debugInfo.size() << endl;
    }
    else it->second.dumpAuction();
}

void
RouterServiceBase::
dumpSpot(const Id & auction, const Id & spot) const
{
    boost::unique_lock<ML::Spinlock> guard(shared->debugLock);
    auto it = shared->debugInfo.find(auction);
    if (it == shared->debugInfo.end()) {
        //cerr << "*** unknown auction " << auction << " in "
        //     << shared->debugInfo.size() << endl;
    }
    else it->second.dumpSpot(spot);
}

Date
RouterServiceBase::
getCurrentTime() const
{
    if (shared->simulationMode_)
        return shared->simulatedTime_;
    else return Date::now();
}

void
RouterServiceBase::
setSimulatedTime(Date currentTime)
{
    if (!shared->simulationMode_)
        throw Exception("not in simulation mode");

    if (currentTime < shared->simulatedTime_) {
        cerr << "warning: simulated time is going backwards from "
             << shared->simulatedTime_.print(4) << " to "
             << currentTime.print(4) << endl;
    }

    shared->simulatedTime_ = currentTime;
}

} // namespace RTBKIT
