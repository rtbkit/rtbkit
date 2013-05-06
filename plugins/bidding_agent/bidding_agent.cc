/* bidding_agent.cc                                                   -*- C++ -*-
   Rémi Attab, 14 December 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Implementation details of the router proxy.
*/

#include "rtbkit/plugins/bidding_agent/bidding_agent.h"
#include "rtbkit/core/agent_configuration/agent_config.h"

#include "jml/arch/exception.h"
#include "jml/arch/timers.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/exc_check.h"
#include "jml/utils/exc_assert.h"
#include "jml/arch/futex.h"
#include "soa/service/zmq_utils.h"
#include "soa/service/process_stats.h"

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>

using namespace std;
using namespace Datacratic;
using namespace RTBKIT;
using namespace ML;

namespace RTBKIT {


/******************************************************************************/
/* OVERLOADED UTILITIES                                                       */
/******************************************************************************/

inline void
sendMesg(
        zmq::socket_t & sock,
        const Id & id,
        int options = 0)
{
    Datacratic::sendMesg(sock, id.toString(), options);
}

static Json::Value
jsonParse(const std::string & str)
{
    if (str.empty()) return Json::Value();
    return Json::parse(str);
}

/******************************************************************************/
/* ROUTER PROXY                                                               */
/******************************************************************************/

BiddingAgent::
BiddingAgent(std::shared_ptr<ServiceProxies> proxies,
            const std::string & name)
    : ServiceBase(name, proxies),
      agentName(name + "_" + to_string(getpid())),
      toRouters(getZmqContext()),
      toPostAuctionServices(getZmqContext()),
      toConfigurationAgent(getZmqContext()),
      toRouterChannel(65536),
      requiresAllCB(true)
{
}

BiddingAgent::
BiddingAgent(ServiceBase& parent,
            const std::string & name)
    : ServiceBase(name, parent),
      agentName(name + "_" + to_string(getpid())),
      toRouters(getZmqContext()),
      toPostAuctionServices(getZmqContext()),
      toConfigurationAgent(getZmqContext()),
      toRouterChannel(65536),
      requiresAllCB(true)
{
}

BiddingAgent::
~BiddingAgent()
{
    shutdown();
}

void
BiddingAgent::
init()
{
    auto messageHandler = [=] (
            const string & service, const vector<string>& msg)
        {
            try {
                handleRouterMessage(service, msg);
            }
            catch (const std::exception& ex) {
                recordHit("error");
                cerr << "Error handling auction message " << ex.what() << endl;
                for (size_t i = 0; i < msg.size(); ++i)
                    cerr << "\t" << i << ": " << msg[i] << endl;
                cerr << endl;
            }
        };

    toRouters.messageHandler = messageHandler;

    toPostAuctionServices.messageHandler = messageHandler;
    toConfigurationAgent.init(getServices()->config, agentName);
    toConfigurationAgent.connectToServiceClass
            ("rtbAgentConfiguration", "agents");

    toConfigurationAgent.connectHandler = [&] (const std::string&)
        {
            sendConfig();
        };

    toRouters.init(getServices()->config, agentName);
    toRouters.connectHandler = [=] (const std::string & connectedTo)
        {
            std::stringstream ss;
            ss << "BiddingAgent is connected to router "
                 << connectedTo << endl;
            cerr << ss.str() ;
            toRouters.sendMessage(connectedTo, "CONFIG", agentName);
        };
    toRouters.connectAllServiceProviders("rtbRequestRouter", "agents");
    toRouterChannel.onEvent = [=] (const RouterMessage & msg)
        {
            toRouters.sendMessage(msg.toRouter, msg.type, msg.payload);
        };
    toPostAuctionServices.init(getServices()->config, agentName);
    toPostAuctionServices.connectHandler = [=] (const std::string & connectedTo)
        {
            cerr << "BiddingAgent is connected to post auction service "
                 << connectedTo << endl;
            //toPostAuctionServices.sendMessage(connectedTo, "CONFIG", agentName);
        };
    toPostAuctionServices.connectAllServiceProviders("rtbPostAuctionService",
                                                     "agents");

    addSource("BiddingAgent::toRouters", toRouters);
    addSource("BiddingAgent::toPostAuctionServices", toPostAuctionServices);
    addSource("BiddingAgent::toConfigurationAgent", toConfigurationAgent);
    addSource("BiddingAgent::toRouterChannel", toRouterChannel);

    MessageLoop::init();
}

void
BiddingAgent::
shutdown()
{
    MessageLoop::shutdown();

    toConfigurationAgent.shutdown();
    toRouters.shutdown();
    //toPostAuctionService.shutdown();
}

void
BiddingAgent::
handleRouterMessage(const std::string & fromRouter,
                    const std::vector<std::string> & message)
{
    if (message.empty()) {
        cerr << "invalid empty message received" << endl;
        recordHit("errorEmptyMessage");
        return;
    }
    recordHit(message[0]);
    if (message[0].empty()) {
        cerr << "invalid message with empty type received" << endl;
        recordHit("errorEmptyMessageType");
        return;
    }

    bool invalid = false;

    switch (message[0][0]) {
    case 'A':
        if (message[0] == "AUCTION")
            handleBidRequest(fromRouter, message, onBidRequest);
        else invalid = true;
        break;

    case 'W':
        if (message[0] == "WIN")
            handleResult(message, onWin);
        else invalid = true;
        break;

    case 'L':
        if (message[0] == "LOSS")
            handleResult(message, onLoss);
        else invalid = true;
        break;

    case 'N':
        if (message[0] == "NOBUDGET")
            handleResult(message, onNoBudget);
        else if (message[0] == "NEEDCONFIG") sendConfig();
        else invalid = true;
        break;

    case 'T':
        if (message[0] == "TOOLATE")
            handleResult(message, onTooLate);
        else invalid = true;
        break;

    case 'I':
        if (message[0] == "INVALID")
            handleResult(message, onInvalidBid);
        else if (message[0] == "IMPRESSION")
            handleDelivery(message, onImpression);
        else invalid = true;
        break;

    case 'D':
        if (message[0] == "DROPPEDBID")
            handleResult(message, onDroppedBid);
        else invalid = true;
        break;

    case 'G':
        if (message[0] == "GOTCONFIG") { /* no-op */ }
        else invalid = true;
        break;

    case 'E':
        if (message[0] == "ERROR")
            handleError(message, onError);
        else invalid = true;
        break;

    case 'B':
        if (message[0] == "BYEBYE")   { /*no-op*/ }
        else invalid = true;
        break;

    case 'C':
        if (message[0] == "CLICK")
            handleDelivery(message, onClick);
        else invalid = true;
        break;

    case 'V':
        if (message[0] == "VISIT")
            handleDelivery(message, onVisit);
        else invalid = true;
        break;

    case 'P':
        if (message[0] == "PING0") {
            //cerr << "ping0: message " << message << endl;

            // Low-level ping (to measure network/message queue backlog);
            // we return straight away
            auto message_ = message;
            string received = message.at(1);
            message_.erase(message_.begin(), message_.begin() + 2);
            toRouters.sendMessage(fromRouter, "PONG0", received, Date::now(), message_);
        }
        else if (message[0] == "PING1") {
            // High-level ping (to measure whole stack backlog);
            // we pass through to the agent to process so we can measure
            // any backlog in the agent itself
            handlePing(fromRouter, message, onPing);
        }
        else invalid = true;
        break;

    default:
        invalid = true;
        break;
    }

    if (invalid) {
        recordHit("errorUnknownMessage");
        cerr << "Unknown message: {";
        for_each(message.begin(), message.end(), [&](const string& m) {
                    cerr << m << ", ";
                });
        cerr << "}" << endl;
    }
}

namespace {

/** This is actually for backwards compatibility when we moved the agents from
    pure (dirty) js to a c++ proxy class for protocol habndlingp.
*/
static string
eventName(const string& name)
{
    switch(name[0]) {
    case 'C':
        if (name == "CLICK") return "clicks";
        break;

    case 'D':
        if (name == "DROPPEDBID") return "droppedbids";
        break;

    case 'E':
        if (name == "ERROR") return "errors";
        break;

    case 'I':
        if (name == "INVALIDBID") return "invalidbids";
        if (name == "IMPRESSION") return "impressions";
        break;

    case 'L':
        if (name == "LOSS") return "losses";
        break;

    case 'N':
        if (name == "NOBUDGET") return "nobudgets";
        break;

    case 'P':
        if (name == "PING1") return "ping";
        break;

    case 'T':
        if (name == "TOOLATE") return "toolate";
        break;

    case 'V':
        if (name == "VISIT") return "visits";
        break;

    case 'W':
        if (name == "WIN") return "wins";
        break;
    }

    ExcAssert(false);
    return "unknown";
}

} // anonymous namespace


void
BiddingAgent::
checkMessageSize(const std::vector<std::string>& msg, int expectedSize)
{
    if (msg.size() >= expectedSize)
        return;

    string msgStr = boost::lexical_cast<string>(msg);
    throw ML::Exception("Message of wrong size: size=%d, expected=%d, msg=%s",
            msg.size(), expectedSize, msgStr.c_str());
}

void
BiddingAgent::
handleBidRequest(const std::string & fromRouter,
                 const std::vector<std::string>& msg, BidRequestCbFn& callback)
{
    ExcCheck(!requiresAllCB || callback, "Null callback for " + msg[0]);
    if (!callback) return;

    checkMessageSize(msg, 8);

    double timestamp = boost::lexical_cast<double>(msg[1]);
    Id id(msg[2]);

    string bidRequestSource = msg[3];

    std::shared_ptr<BidRequest> br(
            BidRequest::parse(bidRequestSource, msg[4]));

    Json::Value imp = jsonParse(msg[5]);
    double timeLeftMs = boost::lexical_cast<double>(msg[6]);
    Json::Value augmentations = jsonParse(msg[7]);

    Bids bids;
    bids.reserve(imp.size());

    for (size_t i = 0; i < imp.size(); ++i) {
        Bid bid;

        bid.spotIndex = imp[i]["spot"].asInt();
        for (const auto& creative : imp[i]["creatives"])
            bid.availableCreatives.push_back(creative.asInt());

        bids.push_back(bid);
    }


    recordHit("requests");

    ExcCheck(!requests.count(id), "seen multiple requests with same ID");
    {
        lock_guard<mutex> guard (requestsLock);

        requests[id].timestamp = Date::now();
        requests[id].fromRouter = fromRouter;
    }

    callback(timestamp, id, br, bids, timeLeftMs, augmentations);
}

void
BiddingAgent::
handleResult(const std::vector<std::string>& msg, ResultCbFn& callback)
{
    ExcCheck(!requiresAllCB || callback, "Null callback for " + msg[0]);
    if (!callback) return;

    checkMessageSize(msg, 6);

    recordHit(eventName(msg[0]));
    BidResult result = BidResult::parse(msg);

    if (result.result == BS_WIN) {
        recordLevel(MicroUSD(result.secondPrice), "winPrice");
        recordCount(MicroUSD(result.secondPrice), "winPriceTotal");
    }

    callback(result);

    if (result.result == BS_DROPPEDBID) {
        lock_guard<mutex> guard (requestsLock);
        requests.erase(Id(msg[3]));
    }
}

void
BiddingAgent::
handleError(const std::vector<std::string>& msg, ErrorCbFn& callback)
{
    ExcCheck(!requiresAllCB || callback, "Null callback for " + msg[0]);
    if (!callback) return;

    double timestamp = boost::lexical_cast<double>(msg[1]);
    string description = msg[2];

    vector<string> originalMessage;
    copy(msg.begin()+2, msg.end(),
            back_insert_iterator< vector<string> >(originalMessage));

    callback(timestamp, description, originalMessage);
}

void
BiddingAgent::
handleDelivery(const std::vector<std::string>& msg, DeliveryCbFn& callback)
{
    ExcCheck(!requiresAllCB || callback, "Null callback for " + msg[0]);
    if (!callback) return;

    checkMessageSize(msg, 12);

    DeliveryEvent ev = DeliveryEvent::parse(msg);
    recordHit(eventName(ev.event));

    callback(ev);
}

void
BiddingAgent::
doBid(Id id, const Bids & bids, const Json::Value & jsonMeta)
{
    Json::FastWriter jsonWriter;

    string response = jsonWriter.write(bids.toJson());
    boost::trim(response);

    string meta = jsonWriter.write(jsonMeta);
    boost::trim(meta);

    Date afterSend = Date::now();
    Date beforeSend;
    string fromRouter;
    {
        lock_guard<mutex> guard (requestsLock);

        auto it = requests.find(id);
        if (it != requests.end()) {
            beforeSend = it->second.timestamp;
            fromRouter = it->second.fromRouter;
            requests.erase(it);
        }
    }
    if (fromRouter.empty()) return;

    recordLevel((afterSend - beforeSend) * 1000.0, "timeTakenMs");

    toRouterChannel.push(RouterMessage(
                    fromRouter, "BID", { id.toString(), response, meta }));

    /** Gather some stats */
    for (const Bid& bid : bids) {
        if (bid.isNullBid()) recordHit("filtered.total");
        else {
            recordHit("bids");
            recordLevel(bid.price.value, "bidPrice." + bid.price.getCurrencyStr());
        }
    }
}

void
BiddingAgent::
handlePing(const std::string & fromRouter,
           const std::vector<std::string> & msg,
           PingCbFn& callback)
{
    recordHit(eventName(msg[0]));

    Date started = Date::parseSecondsSinceEpoch(msg.at(1));
    vector<string> payload(msg.begin() + 2, msg.end());

    if (callback)
        callback(fromRouter, started, payload);
    else
        doPong(fromRouter, started, Date::now(), payload);
}

void
BiddingAgent::
doPong(const std::string & fromRouter, Date sent, Date received,
       const std::vector<std::string> & payload)
{
    //cerr << "doPong with payload " << payload << " sent " << sent
    //     << " received " << received << endl;

    vector<string> message = {
        to_string(sent.secondsSinceEpoch()),
        to_string(received.secondsSinceEpoch())
    };

    message.insert(message.end(), payload.begin(), payload.end());
    toRouterChannel.push(RouterMessage(fromRouter, "PONG1", message));
}

void
BiddingAgent::
doConfig(const AgentConfig& config)
{
    doConfigJson(config.toJson());
}

void
BiddingAgent::
doConfigJson(Json::Value jsonConfig)
{
    Json::FastWriter jsonWriter;

    std::string newConfig = jsonWriter.write(jsonConfig);
    boost::trim(newConfig);

    sendConfig(newConfig);
}

void
BiddingAgent::
sendConfig(const std::string& newConfig)
{
    std::lock_guard<std::mutex> guard(configLock);

    if (!newConfig.empty()) config = newConfig;
    if (config.empty()) return;

    toConfigurationAgent.sendMessage("CONFIG", agentName, config);
}

} // namespace RTBKIT
