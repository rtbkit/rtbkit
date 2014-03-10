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

    isUnitTest = false;

    boost::program_options::options_description
        options("Standard Ad Server Connector");
    options.add_options()
        ("win-port,w", value(&winPort), "listening port for wins")
        ("events-port,e", value(&eventsPort), "listening port for events")
        ("isUnitTest,i", value(&isUnitTest), "unit test for Ad-Server connector");
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
}

StandardAdServerConnector::
StandardAdServerConnector(std::shared_ptr<ServiceProxies> const & proxies,
                          Json::Value const & json) :
    HttpAdServerConnector(json.get("name", "standard-adserver").asString(), proxies),
    publisher_(getServices()->zmqContext) {
    int winPort = json.get("winPort", 18143).asInt();
    int eventsPort = json.get("eventsPort", 18144).asInt();
    isUnitTest = json.get("isUnitTest", false).asBool();
    init(winPort, eventsPort, isUnitTest);
}

void
StandardAdServerConnector::
init(StandardAdServerArguments & ssConfig)
{
    ssConfig.validate();
    init(ssConfig.winPort, ssConfig.eventsPort, ssConfig.isUnitTest);
}

void
StandardAdServerConnector::
init(int winsPort, int eventsPort, bool isTest)
{

    isUnitTest = isTest;

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
    string userIdStr;


    if (json.isMember("userIds")) {
        auto item =  json["userIds"];
        if(item.isMember("prov")){
             userIdStr = item["prov"].asString();
             userIds.add(Id(userIdStr), ID_PROVIDER);
        }
    }
    else {
        throw ML::Exception("UserIds is mandatory in a WIN Impression");
    }
    
    const Json::Value & meta = json["winMeta"];

    if(isUnitTest) {
         writeUnitTestWinReqOutput(timestamp,
                                    bidTimestamp,
                                    auctionIdStr, 
                                    adSpotIdStr, 
                                    accountKeyStr, 
                                    winPrice,
                                    userIdStr,
                                    dataCost);
    }
    else {
        publishWin(auctionId, adSpotId, winPrice, timestamp, meta, userIds,
                   accountKey, bidTimestamp);
        publisher_.publish("WIN", timestamp.print(3), auctionIdStr,
                           adSpotIdStr, accountKeyStr,
                           winPrice.toString(), dataCost.toString(), meta);
    }
}

void
StandardAdServerConnector::
handleDeliveryRq(const HttpHeader & header,
                 const Json::Value & json, const std::string & jsonStr)
{
    Date timestamp = Date::fromSecondsSinceEpoch(json["timestamp"].asDouble());
    
    string auctionIdStr, adSpotIdStr, event;
    Id auctionId, adSpotId, userId;
    
    if (json.isMember("auctionId")) {
        auctionIdStr = json["auctionId"].asString();
        adSpotIdStr = json["adSpotId"].asString();
        auctionId = Id(auctionIdStr);
        adSpotId = Id(adSpotIdStr);
    
	event = (json["event"].asString());

        if (isUnitTest) {
            writeUnitTestDeliveryReqOutput(timestamp,
                                           auctionIdStr,
                                           adSpotIdStr,
                                           event);
        }
        else {
            if (event == "click") {
                publishCampaignEvent("CLICK", auctionId, adSpotId, timestamp,
                                     Json::Value());
                publisher_.publish("CLICK", timestamp.print(3), auctionIdStr,
                                   adSpotIdStr);
            }
            else if (event == "conversion") {
                publishCampaignEvent("CONVERSION", auctionId, adSpotId,
                                     timestamp, Json::Value());
                publisher_.publish("CONVERSION", timestamp.print(3),
                                   auctionIdStr, adSpotIdStr);
            }
            else {
                publisher_.publish("CONVERSION", timestamp.print(3), "unmatched",
                                   auctionId.toString());
            }
        }
    }
    else {
        throw ML::Exception("invalid event type: '" + event + "'");
    }
}

void
StandardAdServerConnector::
writeUnitTestWinReqOutput(const Date & timestamp, const Date & bidTimestamp, const string & auctionId, 
                          const string & adSpotId, const string & accountKeyStr, const USD_CPM & winPrice,
                          const string & userId, const USD_CPM dataCost) {

    if (!isUnitTest) {
        throw ML::Exception(string("Illegal to call writeUnitTestWinReqOutput if isUnitTest is False"));
    }

    stringstream testOutStr;
    testOutStr << "{\"timestamp\":\"" << timestamp.print(3) << "\"," <<
        "\"bidTimestamp\":\"" << bidTimestamp.print(3) << "\"," <<
        "\"auctionId\":\"" << auctionId << "\"," <<
        "\"adSpotId\":\"" << adSpotId << "\"," <<
        "\"accountId\":\"" << accountKeyStr << "\"," <<
        "\"winPrice\":\"" << winPrice.toString() << "\"," <<
        "\"userIds\":" << "\"" << userId << "\"," <<
        "\"dataCost\":\"" << dataCost.toString() << "\"}";

    cerr << testOutStr.str() << endl;
} 

void
StandardAdServerConnector::
writeUnitTestDeliveryReqOutput(const Date & timestamp, const string & auctionIdStr, const string & adSpotIdStr, 
                               const string &event){

    if (!isUnitTest) {
        throw ML::Exception(string("Illegal to call writeUnitTestDeliveryRqOutput if isUnitTest is False\n"));
    }

    stringstream testOutStr;
    testOutStr << "{\"timestamp\":\"" << timestamp.print(3) << "\"," <<
        "\"auctionId\":\"" << auctionIdStr << "\"," <<
        "\"adSpotId\":\"" << adSpotIdStr << "\"," <<
        "\"event\":\"" << event << "\"}";

    cerr << testOutStr.str() << endl;
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

