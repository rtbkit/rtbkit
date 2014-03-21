/* standard_adserver_connector.cc
   Wolfgang Sourdeau, March 2013
   Copyright (c) 2013 Datacratic.  All rights reserved. */


#include "rtbkit/common/account_key.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/json_holder.h"

#include "soa/service/logs.h"

#include "standard_adserver_connector.h"


using namespace std;

using namespace boost::program_options;

using namespace RTBKIT;

Logging::Category adserverTrace("Standard Ad-Server connector");

/* STANDARDADSERVERARGUMENTS */
boost::program_options::options_description
StandardAdServerArguments::
makeProgramOptions()
{
    boost::program_options::options_description stdOptions
        = ServiceProxyArguments::makeProgramOptions();

    verbose = false;

    boost::program_options::options_description
        options("Standard-Ad-Server-Connector");
    options.add_options()
        ("win-port,w", value(&winPort), "listening port for wins")
        ("events-port,e", value(&eventsPort), "listening port for events")
        ("verbose,v", value(&verbose), "verbose mode");
    stdOptions.add(options);

    return stdOptions;
}

void
StandardAdServerArguments::
validate()
{
    ExcCheck(winPort > 0, "winPort is not set");
    ExcCheck(eventsPort > 0, "eventsPort is not set");
}

/* STANDARDADSERVERCONNECTOR */

StandardAdServerConnector::
StandardAdServerConnector(std::shared_ptr<ServiceProxies> & proxy,
                           const string & serviceName)
    : HttpAdServerConnector(serviceName, proxy),
      publisher_(proxy->zmqContext)
{
    initEventType(Json::Value());
}

StandardAdServerConnector::
StandardAdServerConnector(std::string const & serviceName, std::shared_ptr<ServiceProxies> const & proxies,
                          Json::Value const & json) :
    HttpAdServerConnector(serviceName, proxies),
    publisher_(getServices()->zmqContext) {
    int winPort = json.get("winPort", 18143).asInt();
    int eventsPort = json.get("eventsPort", 18144).asInt();
    verbose = json.get("verbose", false).asBool();
    initEventType(json);
    init(winPort, eventsPort, verbose);
}

void
StandardAdServerConnector::
initEventType(const Json::Value &json) {
    
    // Default value
    eventType["CLICK"] =  "CLICK";
    eventType["CONVERSION"] =  "CONVERSION";

    // User value
    if(json.isMember("eventType")) {
        auto item = json["eventType"];
        auto items = item.getMemberNames();
        
        for(auto i=items.begin(); i!=items.end(); ++i) {
            eventType[*i] = item[*i].asString();
        }
    }
}

void
StandardAdServerConnector::
init(StandardAdServerArguments & ssConfig)
{
    ssConfig.validate();
    init(ssConfig.winPort, ssConfig.eventsPort, ssConfig.verbose);
}

void
StandardAdServerConnector::
init(int winsPort, int eventsPort, bool verbose)
{
    if(!verbose) {
        adserverTrace.deactivate();
    }

    shared_ptr<ServiceProxies> services = getServices();

    auto win = &StandardAdServerConnector::handleWinRq;
    registerEndpoint(winsPort, bind(win, this, _1, _2, _3));

    auto delivery = &StandardAdServerConnector::handleDeliveryRq;
    registerEndpoint(eventsPort, bind(delivery, this, _1, _2, _3));

    HttpAdServerConnector::init(services->config);
    publisher_.init(services->config, serviceName_ + "/logger");
}


void
StandardAdServerConnector::
init(int winsPort, int eventsPort)
{
    init(winsPort, eventsPort, false);
}

void
StandardAdServerConnector::
start()
{
    bindTcp();

    publisher_.bindTcp(getServices()->ports->getRange("adServer.logger"));
    publisher_.start();
    HttpAdServerConnector::start();
}

void
StandardAdServerConnector::
shutdown()
{
    publisher_.shutdown();
    HttpAdServerConnector::shutdown();
}

HttpAdServerResponse
StandardAdServerConnector::
handleWinRq(const HttpHeader & header,
            const Json::Value & json, const std::string & jsonStr)
{
    HttpAdServerResponse response;

    Date timestamp = Date::fromSecondsSinceEpoch(json["timestamp"].asDouble());
    Date bidTimestamp;
    if (json.isMember("bidTimestamp")) {
        bidTimestamp
            = Date::fromSecondsSinceEpoch(json["bidTimestamp"].asDouble());
    }
    string auctionIdStr(json["auctionId"].asString());
    string adSpotIdStr(json["adSpotId"].asString());
    string accountKeyStr(json["accountId"].asString());
    double winPriceDbl(json["winPrice"].asDouble());
    double dataCostDbl(json["dataCost"].asDouble());

    Id auctionId(auctionIdStr);
    Id adSpotId(adSpotIdStr);
    AccountKey accountKey(accountKeyStr);
    USD_CPM winPrice(winPriceDbl);
    USD_CPM dataCost(dataCostDbl);

    UserIds userIds;
    string userIdStr;


    if (json.isMember("userIds")) {
        auto item =  json["userIds"];
        if(item.isMember("prov")){
             userIdStr = item["prov"].asString();
             userIds.add(Id(userIdStr), ID_PROVIDER);
        }
    }
    else {
        response.valid = false;
        response.error = "MISSING_USERID";
        response.details = "A win notice requires the userIds field";
    }
    
    const Json::Value & meta = json["winMeta"];

    writeUnitTestWinReqOutput(timestamp,
                                    bidTimestamp,
                                    auctionIdStr, 
                                    adSpotIdStr, 
                                    accountKeyStr, 
                                    winPrice,
                                    userIdStr,
                                    dataCost);
    if(response.valid) {
        publishWin(auctionId, adSpotId, winPrice, timestamp, meta, userIds,
                   accountKey, bidTimestamp);
        publisher_.publish("WIN", timestamp.print(3), auctionIdStr,
                           adSpotIdStr, accountKeyStr,
                           winPrice.toString(), dataCost.toString(), meta);
    }

    return response;
}

HttpAdServerResponse
StandardAdServerConnector::
handleDeliveryRq(const HttpHeader & header,
                 const Json::Value & json, const std::string & jsonStr)
{    
    HttpAdServerResponse response;
    string auctionIdStr, adSpotIdStr, userIdStr, event;
    Id auctionId, adSpotId, userId;
    UserIds userIds;
    
    Date timestamp = Date::fromSecondsSinceEpoch(json["timestamp"].asDouble());

    if (!json.isMember("auctionId") || !json.isMember("adSpotId")) {
        response.valid = false;
        response.error = "MISSING_ID";
        response.details = "A conversion notice requires the auctionId and adSpotId field";
    }
    
    auctionIdStr = json["auctionId"].asString();
    adSpotIdStr = json["adSpotId"].asString();
    auctionId = Id(auctionIdStr);
    adSpotId = Id(adSpotIdStr);
    
    if (json.isMember("userIds")) {
        auto item =  json["userIds"];
        if(item.isMember("prov")){
            userIdStr = item["prov"].asString();
            userIds.add(Id(userIdStr), ID_PROVIDER);
        }
    }

    if(json.isMember("event")) {
        event = (json["event"].asString());
        if(eventType.find(event) == eventType.end()){
            response.valid = false;
            response.error = "UNSUPPORTED_EVENT_TYPE";
            response.details = "A conversion notice requires the event field";
        }
    }
    else {
        response.valid = false;
        response.error = "MISSING_EVENT_TYPE";
        response.details = "A conversion notice requires the event field";
    }

    writeUnitTestDeliveryReqOutput(timestamp,
                                    auctionIdStr, 
                                    adSpotIdStr, 
                                    userIdStr,
                                    event);

    if(response.valid) {
        publishCampaignEvent(eventType[event], auctionId, adSpotId, timestamp,
                                 Json::Value(), userIds);
        publisher_.publish(eventType[event], timestamp.print(3), auctionIdStr,
                                adSpotIdStr, userIds.toString());
    }
    return response;
}

void
StandardAdServerConnector::
writeUnitTestWinReqOutput(const Date & timestamp, const Date & bidTimestamp, const string & auctionId, 
                          const string & adSpotId, const string & accountKeyStr, const USD_CPM & winPrice,
                          const string & userId, const USD_CPM dataCost) {

    LOG(adserverTrace) << "{\"timestamp\":\"" << timestamp.print(3) << "\"," <<
        "\"bidTimestamp\":\"" << bidTimestamp.print(3) << "\"," <<
        "\"auctionId\":\"" << auctionId << "\"," <<
        "\"adSpotId\":\"" << adSpotId << "\"," <<
        "\"accountId\":\"" << accountKeyStr << "\"," <<
        "\"winPrice\":\"" << winPrice.toString() << "\"," <<
        "\"userIds\":" << "\"" << userId << "\"," <<
        "\"dataCost\":\"" << dataCost.toString() << "\"}";
} 

void
StandardAdServerConnector::
writeUnitTestDeliveryReqOutput(const Date & timestamp, const string & auctionIdStr, const string & adSpotIdStr, 
                               const string &  userIdStr, const string &event){

    LOG(adserverTrace) << "{\"timestamp\":\"" << timestamp.print(3) << "\"," <<
        "\"auctionId\":\"" << auctionIdStr << "\"," <<
        "\"adSpotId\":\"" << adSpotIdStr << "\"," <<
        "\"userIds\":" << "\"" << userIdStr << "\"," <<
        "\"event\":\"" << event << "\"}";
}

namespace {

//Logging::Category adserverTrace("Standard Ad-Server connector");

struct AtInit {
    AtInit()
    {
        AdServerConnector::registerFactory("standard", [](std::string const & serviceName , std::shared_ptr<ServiceProxies> const & proxies, Json::Value const & json) {
            return new StandardAdServerConnector(serviceName, proxies, json);
        });
    }
} atInit;

}

