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

    Date timestamp;
    string bidRequestIdStr;
    string impIdStr;
    double winPriceDbl;

    Id bidRequestId;
    Id impId;
    USD_CPM winPrice;

    UserIds userIds;
    string userIdStr;
    string passback;

    /*
     *  Timestamp is an required field.
     *  If null, we return an error response.
     */
    if (json.isMember("timestamp")) {
        timestamp = Date::fromSecondsSinceEpoch(json["timestamp"].asDouble());
    } else {
        response.valid = false;
        response.error = "MISSING_TIMESTAMP";
        response.details = "A win notice requires the timestamp field.";

        return response;
    }

    /*
     *  bidRequestId is an required field.
     *  If null, we return an error response.
     */
    if (json.isMember("bidRequestId")) {
        bidRequestIdStr = json["bidRequestId"].asString();
        bidRequestId = Id(bidRequestIdStr); 
    } else {
        response.valid = false;
        response.error = "MISSING_BIDREQUESTID";
        response.details = "A win notice requires the bidRequestId field.";

        return response;
    }

    /*
     *  impid is an required field.
     *  If null, we return an error response.
     */
    if (json.isMember("impid")) {
        impIdStr = json["impid"].asString();
        impId = Id(impIdStr);
    } else {
        response.valid = false;
        response.error = "MISSING_IMPID";
        response.details = "A win notice requires the impId field.";
    
        return response;
    }

    /*
     *  price is an required field.
     *  If null, we return an error response.
     */
    if (json.isMember("price")) {
        winPriceDbl = json["price"].asDouble();
        winPrice = USD_CPM(winPriceDbl);
    } else {
        response.valid = false;
        response.error = "MISSING_WINPRICE";
        response.details = "A win notice requires the price field.";
    
        return response;
    }
    
    /*
     *  UserIds is an optional field.
     *  If null, we just put an empty array.
     */
    if (json.isMember("userIds")) {
        auto item =  json["userIds"];
        if(!item.empty())
            userIds.add(Id(item[0].asString()), ID_PROVIDER);
    }
    else {
        // UserIds is optional
    }

    /*
     *  Passback is an optional field.
     *  If null, we just put an empty string.
     */
    if (json.isMember("passback")) {
        passback =  json["passback"].asString();
    }
    else {
        // Passback is optional
    }

    LOG(adserverTrace) << "{\"timestamp\":\"" << timestamp.print(3) << "\"," <<
        "\"bidRequestId\":\"" << bidRequestId << "\"," <<
        "\"impId\":\"" << impId << "\"," <<
        "\"winPrice\":\"" << winPrice.toString() << "\" }";

    if(response.valid) {
        publishWin(bidRequestId, impId, winPrice, timestamp, Json::Value(), userIds,
                   AccountKey(), Date());
        publisher_.publish("WIN", timestamp.print(3), bidRequestIdStr,
                           impIdStr, winPrice.toString());
    }

    return response;
}

HttpAdServerResponse
StandardAdServerConnector::
handleDeliveryRq(const HttpHeader & header,
                 const Json::Value & json, const std::string & jsonStr)
{    
    HttpAdServerResponse response;
    string bidRequestIdStr, impIdStr, userIdStr, event;
    Id bidRequestId, impId, userId;
    UserIds userIds;
    Date timestamp;
    
    /*
     *  Timestamp is an required field.
     *  If null, we return an error response.
     */
    if (json.isMember("timestamp")) {
        timestamp = Date::fromSecondsSinceEpoch(json["timestamp"].asDouble());
    } else {
        response.valid = false;
        response.error = "MISSING_TIMESTAMP";
        response.details = "A win notice requires the timestamp field.";

        return response;
    }


    /*
     *  type is an required field.
     *  If null, we return an error response.
     */
    if (json.isMember("type")) {

        event = (json["type"].asString());
        
        if(eventType.find(event) == eventType.end()) {
            response.valid = false;
            response.error = "UNSUPPORTED_TYPE";
            response.details = "A campaign event requires the type field.";
    
            return response;
        }

    } else {

        response.valid = false;
        response.error = "MISSING_TYPE";
        response.details = "A campaign event requires the type field.";
    
        return response;
    }

    /*
     *  impid is an required field.
     *  If null, we return an error response.
     */
    if (json.isMember("impid")) {

    } else {
        response.valid = false;
        response.error = "MISSING_IMPID";
        response.details = "A campaign event requires the impId field.";
    
        return response;
    }

    /*
     *  bidRequestId is an required field.
     *  If null, we return an error response.
     */
    if (json.isMember("bidRequestId")) {

    } else {
        response.valid = false;
        response.error = "MISSING_BIDREQUESTID";
        response.details = "A campaign event requires the bidRequestId field.";
    
        return response;
    }

    /*
     *  UserIds is an optional field.
     *  If null, we just put an empty array.
     */
    if (json.isMember("userIds")) {
        auto item =  json["userIds"];

        if(!item.empty())
            userIds.add(Id(item[0].asString()), ID_PROVIDER);
    }
    else {
        // UserIds is optional
    }

    bidRequestIdStr = json["bidRequestId"].asString();
    impIdStr = json["impid"].asString();
    bidRequestId = Id(bidRequestIdStr);
    impId = Id(impIdStr);
    
    LOG(adserverTrace) << "{\"timestamp\":\"" << timestamp.print(3) << "\"," <<
        "\"bidRequestId\":\"" << bidRequestIdStr << "\"," <<
        "\"impId\":\"" << impIdStr << "\"," <<
        "\"event\":\"" << event << 
        "\"userIds\":" << userIds.toString() << "\"}";

    if(response.valid) {
        publishCampaignEvent(eventType[event], bidRequestId, impId, timestamp,
                                 Json::Value(), userIds);
        publisher_.publish(eventType[event], timestamp.print(3), bidRequestIdStr,
                                impIdStr, userIds.toString());
    }
    return response;
}

namespace {

struct AtInit {
    AtInit()
    {
        AdServerConnector::registerFactory("standard", [](std::string const & serviceName , std::shared_ptr<ServiceProxies> const & proxies, Json::Value const & json) {
            return new StandardAdServerConnector(serviceName, proxies, json);
        });
    }
} atInit;

}

