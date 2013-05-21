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


static Id emptySpotId("1");


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
        ("external-win-port,c", value(&externalWinPort),
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
    ExcCheck(!installation.empty(), "'installation' is required");
    ExcCheck(!nodeName.empty(), "'node-name' is required");
}

/* STANDARDADSERVERCONNECTOR */

StandardAdServerConnector::
StandardAdServerConnector(std::shared_ptr<ServiceProxies> & proxy,
                           const string & serviceName)
    : HttpAdServerConnector(serviceName, proxy),
      publisher_(proxy->zmqContext)
{
}

void
StandardAdServerConnector::
init(StandardAdServerArguments & ssConfig)
{
    ssConfig.validate();

    shared_ptr<ServiceProxies> services = getServices();

    auto onWinRq = bind(&StandardAdServerConnector::handleWinRq, this,
                        placeholders::_1, placeholders::_2,
                        placeholders::_3);
    registerEndpoint(ssConfig.winPort, onWinRq);

    auto onDeliveryRq = bind(&StandardAdServerConnector::handleDeliveryRq,
                             this,
                             placeholders::_1, placeholders::_2,
                             placeholders::_3);
    registerEndpoint(ssConfig.eventsPort, onDeliveryRq);

    auto onExternalWinRq
        = bind(&StandardAdServerConnector::handleExternalWinRq,
               this,
               placeholders::_1, placeholders::_2, placeholders::_3);
    registerEndpoint(ssConfig.externalWinPort, onExternalWinRq);

    HttpAdServerConnector::init(services->config);
    publisher_.init(services->config, serviceName_ + "/logger");
}

void
StandardAdServerConnector::
start()
{
    bindTcp();

    publisher_.bindTcp(getServices()->ports->getRange("adServer/logger"));
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
    Id auctionId(json["auctionId"].asString());
    double price(json["price"].asDouble());
    USD_CPM usdCpmPrice(price);
    double dataCost(json["dataCost"].asDouble());
    USD_CPM usdCpmDC(dataCost);
    AccountKey accountKey(json["account-key"].asString());

    // onWin(auctionId, price, dataCost, accountKey);
    JsonHolder meta("{'dataCost':" + to_string(dataCost) + "}");
    UserIds userIds;
    Date emptyDate;
    Date now = Date::now();

    publishWin(auctionId, emptySpotId,
               usdCpmPrice, now, meta, userIds, accountKey,
               emptyDate);
    publisher_.publish("WIN", now.print(3),
                       auctionId.toString(),
                       usdCpmPrice.toString(), usdCpmDC.toString());
}

void
StandardAdServerConnector::
handleDeliveryRq(const HttpHeader & header,
                 const Json::Value & json, const std::string & jsonStr)
{
    Date now = Date::now();
    string eventType(json["eventType"].asString());

    if (eventType == "click") {
        Id auctionId(json["auctionId"].asString());
        UserIds userIds;

        publishCampaignEvent("CLICK",
                             auctionId, emptySpotId,
                             now, Json::Value(Json::nullValue),
                             userIds);

        publisher_.publish("CLICK", now.print(3), auctionId.toString());
    }
    else if (eventType == "conversion") {
        Id auctionId(json["auctionId"].asString());
        double payout(json["payout"].asDouble());
        USD_CPM usdCpmPayout(payout);
        publisher_.publish("CONVERSION", now.print(3),
                           auctionId.toString(), usdCpmPayout.toString(),
                           jsonStr);
    }
}

void
StandardAdServerConnector::
handleExternalWinRq(const HttpHeader & header,
                    const Json::Value & json, const std::string & jsonStr)
{
    Id auctionId(json["auctionId"].asString());
    double price(json["price"].asDouble());
    USD_CPM usdCpmPrice(price);
    double dataCost(json["dataCost"].asDouble());
    USD_CPM usdCpmDC(dataCost);
    const Json::Value & br = json["bidRequest"];
    const Json::Value & ext = br["ext"];
    const Json::Value & cids = ext["cids"];
    uint32_t cid(cids[0].asInt());

    // onExternalWin(auctionId, price, dataCost, cid, boost::trim_copy(br.toString()));
    UserIds userIds;
    Date now = Date::now();

    publisher_.publish("EXTERNALWIN", now.print(3),
                       auctionId.toString(),
                       usdCpmPrice.toString(), usdCpmDC.toString(),
                       to_string(cid),
                       boost::trim_copy(br.toString()));
}
