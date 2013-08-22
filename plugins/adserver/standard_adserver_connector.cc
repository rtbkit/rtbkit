/* standard_adserver_connector.cc
   Wolfgang Sourdeau, March 2013
   Copyright (c) 2013 Datacratic.  All rights reserved. */


#include "rtbkit/common/account_key.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/json_holder.h"

#include "standard_adserver_connector.h"


using namespace std;

using namespace boost::program_options;

using namespace RTBKIT;


/* STANDARDADSERVERARGUMENTS */
boost::program_options::options_description
StandardAdServerArguments::
makeProgramOptions()
{
    boost::program_options::options_description stdOptions
        = ServiceProxyArguments::makeProgramOptions();

    boost::program_options::options_description
        options("Standard Ad Server Connector");
    options.add_options()
        ("win-port,w", value(&winPort), "listening port for wins")
        ("events-port,e", value(&eventsPort), "listening port for events")
        ("external-win-port,x", value(&externalWinPort),
         "listening port for external wins");
    stdOptions.add(options);

    return stdOptions;
}

void
StandardAdServerArguments::
validate()
{
    ExcCheck(winPort > 0, "winPort is not set");
    ExcCheck(eventsPort > 0, "eventsPort is not set");
    ExcCheck(externalWinPort > 0, "externalWinPort is not set");
}

/* STANDARDADSERVERCONNECTOR */

StandardAdServerConnector::
StandardAdServerConnector(std::shared_ptr<ServiceProxies> & proxy,
                           const string & serviceName)
    : HttpAdServerConnector(serviceName, proxy),
      publisher_(proxy->zmqContext)
{
}

StandardAdServerConnector::
StandardAdServerConnector(std::shared_ptr<ServiceProxies> const & proxies,
                          Json::Value const & json) :
    HttpAdServerConnector(json.get("name", "standard-adserver").asString(), proxies),
    publisher_(getServices()->zmqContext) {
    int winPort = json.get("winPort", 18143).asInt();
    int eventsPort = json.get("eventsPort", 18144).asInt();
    int externalWinPort = json.get("externalWinPort", 18145).asInt();
    init(winPort, eventsPort, externalWinPort);
}

void
StandardAdServerConnector::
init(StandardAdServerArguments & ssConfig)
{
    ssConfig.validate();
    init(ssConfig.winPort, ssConfig.eventsPort, ssConfig.externalWinPort);
}

void
StandardAdServerConnector::
init(int winsPort, int eventsPort, int externalPort)
{
    shared_ptr<ServiceProxies> services = getServices();

    auto onWinRq = [=] (const HttpHeader & header,
                        const Json::Value & json,
                        const std::string & jsonStr) {
        this->handleWinRq(header, json, jsonStr);
    };
    registerEndpoint(winsPort, onWinRq);

    auto onDeliveryRq = [=] (const HttpHeader & header,
                        const Json::Value & json,
                        const std::string & jsonStr) {
        this->handleDeliveryRq(header, json, jsonStr);
    };
    registerEndpoint(eventsPort, onDeliveryRq);

    auto onExternalWinRq = [=] (const HttpHeader & header,
                        const Json::Value & json,
                        const std::string & jsonStr) {
        this->handleExternalWinRq(header, json, jsonStr);
    };
    registerEndpoint(externalPort, onExternalWinRq);

    HttpAdServerConnector::init(services->config);
    publisher_.init(services->config, serviceName_ + "/logger");
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

void
StandardAdServerConnector::
handleWinRq(const HttpHeader & header,
            const Json::Value & json, const std::string & jsonStr)
{
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

    const Json::Value & meta = json["winMeta"];

    publishWin(auctionId, adSpotId, winPrice, timestamp, meta, userIds,
               accountKey, bidTimestamp);
    publisher_.publish("WIN", timestamp.print(3), auctionIdStr,
                       adSpotIdStr, accountKeyStr,
                       winPrice.toString(), dataCost.toString(), meta);
}

void
StandardAdServerConnector::
handleDeliveryRq(const HttpHeader & header,
                 const Json::Value & json, const std::string & jsonStr)
{
    Date timestamp = Date::fromSecondsSinceEpoch(json["timestamp"].asDouble());
    Date bidTimestamp;
    if (json.isMember("bidTimestamp")) {
        bidTimestamp
            = Date::fromSecondsSinceEpoch(json["bidTimestamp"].asDouble());
    }
    int matchType(0); /* 1: campaign, 2: user, 0: none */
    string auctionIdStr, adSpotIdStr, userIdStr;
    Id auctionId, adSpotId, userId;
    UserIds userIds;
    
    if (json.isMember("auctionId")) {
        auctionIdStr = json["auctionId"].asString();
        adSpotIdStr = json["adSpotId"].asString();
        auctionId = Id(auctionIdStr);
        adSpotId = Id(adSpotIdStr);
        matchType = 1;
    }
    if (json.isMember("userId")) {
        userIdStr = json["userId"].asString();
        userId = Id(userIdStr);
        if (!matchType)
            matchType = 2;
    }

    string event(json["event"].asString());
    if (event == "click") {
        if (matchType != 1) {
            throw ML::Exception("click events must have auction/spot ids");
        }
        publishCampaignEvent("CLICK", auctionId, adSpotId, timestamp,
                             Json::Value(), userIds);
        publisher_.publish("CLICK", timestamp.print(3), auctionIdStr,
                           adSpotIdStr, userIds.toString());
    }
    else if (event == "conversion") {
        Json::Value meta;
        meta["payout"] = json["payout"];
        USD_CPM payout(json["payout"].asDouble());

        if (matchType == 1) {
            publishCampaignEvent("CONVERSION", auctionId, adSpotId,
                                 timestamp, meta, userIds);
            publisher_.publish("CONVERSION", timestamp.print(3), "campaign", 
                               auctionIdStr, adSpotIdStr, payout.toString());
        }
        else if (matchType == 2) {
            publishUserEvent("CONVERSION", userId,
                             timestamp, meta, userIds);
            publisher_.publish("CONVERSION", timestamp.print(3), "user",
                               auctionId.toString(), payout.toString());
        }
        else {
            publisher_.publish("CONVERSION", timestamp.print(3), "unmatched",
                               auctionId.toString(), payout.toString());
        }
    }
    else {
        throw ML::Exception("invalid event type: '" + event + "'");
    }
}

void
StandardAdServerConnector::
handleExternalWinRq(const HttpHeader & header,
                    const Json::Value & json, const std::string & jsonStr)
{
    Date now = Date::now();
    string auctionIdStr(json["auctionId"].asString());
    Id auctionId(auctionIdStr);

    double price(json["winPrice"].asDouble());
    double dataCostDbl(0.0);
    if (json.isMember("dataCost")) {
        dataCostDbl = json["dataCost"].asDouble();
    }
    USD_CPM dataCost(dataCostDbl);
    Json::Value bidRequest = json["bidRequest"];

    publisher_.publish("EXTERNALWIN", now.print(3), auctionIdStr,
                       std::to_string(price), dataCost.toString(),
                       boost::trim_copy(bidRequest.toString()));
}

namespace {

struct AtInit {
    AtInit()
    {
        AdServerConnector::registerFactory("standard", [](std::shared_ptr<ServiceProxies> const & proxies,
                                                          Json::Value const & json) {
            return new StandardAdServerConnector(proxies, json);
        });
    }
} atInit;

}

