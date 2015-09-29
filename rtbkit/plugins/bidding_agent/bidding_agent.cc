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
/* UTILITIES                                                                  */
/******************************************************************************/


typedef uint64_t hash_t;

const hash_t prime = 0x100000001B3ull;
const hash_t basis = 0xCBF29CE484222325ull;


hash_t hash(const std::string & str)
{
    hash_t ret{basis};
    auto i = 0;
    
	while(str[i]){
		ret ^= str[i];
		ret *= prime;
		i++;
	}
 
	return ret;
} 

constexpr hash_t hash_compile_time(char const* str, hash_t last_value = basis)
{
    return *str ? hash_compile_time(str+1, (*str ^ last_value) * prime) : last_value;
}


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
             const std::string & name,
             double maxAddedLatency)
    : ServiceBase(name, proxies),
      MessageLoop(1 /* threads */, maxAddedLatency),
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
             const std::string & name,
             double maxAddedLatency)
    : ServiceBase(name, parent),
      MessageLoop(1 /* threads */, maxAddedLatency),
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

    // No need to init() message loop; it was done in the constructor
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


    auto newMessage = [&] {
        auto msg(message);
        msg.insert(msg.begin(), "CAMPAIGN_EVENT");
        return msg;
    }();
    
    switch (hash(message[0])) {
        case hash_compile_time("AUCTION") : handleBidRequest(fromRouter, message, onBidRequest); break;
        case hash_compile_time("WIN") :     handleResult(message, onWin); break;
        case hash_compile_time("LOSS") :    handleResult(message, onLoss); break;
        case hash_compile_time("LATEWIN") : handleResult(message, onLateWin ); break;
        case hash_compile_time("NOBUDGET") : handleResult(message, onNoBudget); break;
        case hash_compile_time("NEEDCONFIG") : sendConfig(); break;
        case hash_compile_time("TOOLATE") : handleResult(message, onTooLate); break;
        case hash_compile_time("INVALID") : handleResult(message, onInvalidBid); break;
        case hash_compile_time("CAMPAIGN_EVENT") :  {
            if(!onCampaignEvent) { 
                switch (hash(message[1])) {  
                     // Backward compatibility : replace by CAMPAIGN_EVENT
                    case hash_compile_time("VISIT") : {
                        handleDelivery(message, onVisit); 
                        break;
                    }
                    case hash_compile_time("IMPRESSION") : {
                        handleDelivery(message, onImpression); 
                        break;
                    }
                    case hash_compile_time("CLICK") : {
                        handleDelivery(message, onClick); 
                        break;
                    }
                    default : {
                        recordHit("errorUnknownMessage");
                        cerr << "Unknown message: {";
                        for_each(message.begin(), message.end(), [&](const string& m) {
                            cerr << m << ", ";
                        });
                        cerr << "}" << endl;
                    }
                }
            }
            else {
                handleDelivery(message, onCampaignEvent);
            }
            break;
        }
        case hash_compile_time("DROPPEDBID") : handleResult(message, onDroppedBid); break;
        case hash_compile_time("GOTCONFIG") : /* no-op */ ; break;
        case hash_compile_time("ERROR") : handleError(message, onError) ; break;
        case hash_compile_time("BYEBYE"): {
             if (onByebye) {
                 onByebye(fromRouter,Date::now());
              }
              else {
                 cerr << "eviction notification received. agent should join again";
              }
              break;
        }
        case hash_compile_time("PING0") : {
            //cerr << "ping0: message " << message << endl;

            // Low-level ping (to measure network/message queue backlog);
            // we return straight away
            auto message_ = message;
            string received = message.at(1);
            message_.erase(message_.begin(), message_.begin() + 2);
            toRouters.sendMessage(fromRouter, "PONG0", received, Date::now(), message_);
            break;
        } 
        case hash_compile_time("PING1") : {

            // High-level ping (to measure whole stack backlog);
            // we pass through to the agent to process so we can measure
            // any backlog in the agent itself
            handlePing(fromRouter, message, onPing);
            break;
        }
        default : {
             switch (hash(message[0])) {  
                 // Backward compatibility : replace by CAMPAIGN_EVENT
                 case hash_compile_time("VISIT") : {
                     handleDelivery(newMessage, onVisit); 
                     break;
                 }
                 case hash_compile_time("IMPRESSION") : {
                     handleDelivery(newMessage, onImpression); 
                     break;
                 }
                 case hash_compile_time("CLICK") : {
                     handleDelivery(newMessage, onClick); 
                     break;
                 }
                 default : {
                     recordHit("errorUnknownMessage");
                     cerr << "Unknown message: {";
                     for_each(message.begin(), message.end(), [&](const string& m) {
                        cerr << m << ", ";
                     });
                     cerr << "}" << endl;
                 }

             }
       }
   }
}

namespace {

/** This is actually for backwards compatibility when we moved the agents from
    pure (dirty) js to a c++ proxy class for protocol habndlingp.
*/
static string
eventName(const string& name)
{
    switch (hash(name)) {
        case hash_compile_time("CLICK") : return "clicks"; break;
        case hash_compile_time("DROPPEDBID") : return "droppedbids"; break;
        case hash_compile_time("ERROR") : return "errors"; break;
        case hash_compile_time("INVALID") : return "invalidbids"; break;
        case hash_compile_time("IMPRESSION") : return "impressions"; break;
        case hash_compile_time("LOSS") : return "losses"; break;
        case hash_compile_time("NOBUDGET") : return "nobudgets"; break;
        case hash_compile_time("PING1") : return "ping"; break;
        case hash_compile_time("TOOLATE") : return "toolate"; break;
        case hash_compile_time("VISIT") : return "visits"; break;
        case hash_compile_time("WIN") : return "wins"; break;
        case hash_compile_time("LATEWIN") : return "latewins"; break;
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
    throw ML::Exception("Message of wrong size: size=%zu, expected=%d, msg=%s",
                        msg.size(), expectedSize, msgStr.c_str());
}

void
BiddingAgent::
handleBidRequest(const std::string & fromRouter,
                 const std::vector<std::string>& msg, BidRequestCbFn& callback)
{
    ExcCheck(!requiresAllCB || callback, "Null callback for " + msg[0]);
    if (!callback) return;

    checkMessageSize(msg, 9);

    double timestamp = boost::lexical_cast<double>(msg[1]);
    Id id(msg[2]);

    string bidRequestSource = msg[3];

    std::shared_ptr<BidRequest> br(
            BidRequest::parse(bidRequestSource, msg[4]));

    Json::Value imp = jsonParse(msg[5]);
    double timeLeftMs = boost::lexical_cast<double>(msg[6]);
    Json::Value augmentations = jsonParse(msg[7]);
    WinCostModel wcm = WinCostModel::fromJson(jsonParse(msg[8]));

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

    {
        lock_guard<mutex> guard (requestsLock);
        ExcCheck(!requests.count(id), "seen multiple requests with same ID");

        requests[id].timestamp = Date::now();
        requests[id].fromRouter = fromRouter;
    }

    callback(timestamp, id, br, bids, timeLeftMs, augmentations, wcm);
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
        recordLevel(int64_t(MicroUSD(result.secondPrice)), "winPrice");
        recordCount(int64_t(MicroUSD(result.secondPrice)), "winPriceTotal");
        Bid bid = result.ourBid.bidForSpot(result.spotNum);
        recordLevel(int64_t(MicroUSD(bid.price)), "bidPriceOnWin");
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

    checkMessageSize(msg, 13);

    DeliveryEvent ev = DeliveryEvent::parse(msg);
    recordHit(eventName(ev.event));

    callback(ev);
}

void
BiddingAgent::
doBid(Id id, Bids bids, const Json::Value & jsonMeta, const WinCostModel & wcm)
{
    for (Bid& bid : bids) {
        if (bid.creativeIndex >= 0) {
            if (!bid.isNullBid()) {
                recordLevel(bid.price.value, "bidPrice." + bid.price.getCurrencyStr());
            }

            bid.price = agent_config.creatives[bid.creativeIndex].fees->applyFees(bid.price);
        }
    }

    Json::FastWriter jsonWriter;

    string response = jsonWriter.write(bids.toJson());
    boost::trim(response);

    string meta = jsonWriter.write(jsonMeta);
    boost::trim(meta);

    string model = jsonWriter.write(wcm.toJson());
    boost::trim(model);

    Date afterSend = Date::now();
    Date beforeSend;
    string fromRouter;

    {
        lock_guard<mutex> guard (requestsLock);

        auto it = requests.find(id);

        /** If the auction id isn't in the map then we previously received a
            DROPBID message we should simply forget this bid.
         */
        if (it == requests.end()) {
            cerr << "Ignoring bid (dropped auction id): " << id << endl;
            return;
        }

        beforeSend = it->second.timestamp;
        fromRouter = it->second.fromRouter;
        requests.erase(it);
    }
    if (fromRouter.empty()) return;

    recordLevel((afterSend - beforeSend) * 1000.0, "timeTakenMs");

    toRouterChannel.push(RouterMessage(
                    fromRouter, "BID", { id.toString(), response, model, meta }));

    /** Gather some stats */
    for (const Bid& bid : bids) {
        if (bid.isNullBid()) recordHit("noBid");
        else {
            recordHit("bids");
            recordLevel(bid.price.value, "bidPriceAugmented." + bid.price.getCurrencyStr());
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

    agent_config = AgentConfig::createFromJson(jsonConfig);

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
