/* adserver_connector.cc
   Jeremy Barnes, 19 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Base class to connect to an ad server.  We also have an http ad server
   connector that builds on top of this.
*/


#include "adserver_connector.h"

#include "rtbkit/common/auction_events.h"
#include <dlfcn.h>

using namespace std;

namespace RTBKIT {


/*****************************************************************************/
/* POST AUCTION PROXY                                                        */
/*****************************************************************************/

AdServerConnector::
AdServerConnector(const string & serviceName,
                  const shared_ptr<Datacratic::ServiceProxies> & proxy)
    : ServiceBase(serviceName, proxy),
      toPostAuctionService_(proxy)
{
}

AdServerConnector::
~AdServerConnector()
{
}

void
AdServerConnector::
init(shared_ptr<ConfigurationService> config)
{
    shared_ptr<ServiceProxies> services = getServices();

    registerServiceProvider(serviceName_, { "adServer" });
    services->config->removePath(serviceName());

    toPostAuctionService_.init();
}

void
AdServerConnector::
start()
{
    startTime_ = Date::now();
    recordHit("up");
}

void
AdServerConnector::
shutdown()
{
}

void
AdServerConnector::
recordUptime()
    const
{
    recordLevel(Date::now().secondsSince(startTime_), "uptime");
}

void
AdServerConnector::
publishWin(const Id & bidRequestId,
           const Id & impId,
           Amount winPrice,
           Date timestamp,
           const JsonHolder & winMeta,
           const UserIds & ids,
           const AccountKey & account,
           Date bidTimestamp)
{
    recordHit("receivedEvent");
    recordHit("event.WIN");

    PostAuctionEvent event;
    event.type = PAE_WIN;
    event.auctionId = bidRequestId;
    event.adSpotId = impId;
    event.winPrice = winPrice;
    event.timestamp = timestamp;
    event.metadata = winMeta;
    event.uids = ids;
    event.account = account;
    event.bidTimestamp = bidTimestamp;

    toPostAuctionService_.sendEvent(event);
}

void
AdServerConnector::
publishLoss(const Id & bidRequestId,
            const Id & impId,
            Date timestamp,
            const JsonHolder & lossMeta,
            const AccountKey & account,
            Date bidTimestamp)
{
    recordHit("receivedEvent");
    recordHit("event.LOSS");

    PostAuctionEvent event;
    event.type = PAE_LOSS;
    event.auctionId = bidRequestId;
    event.adSpotId = impId;
    event.timestamp = timestamp;
    event.metadata = lossMeta;
    event.account = account;
    event.bidTimestamp = bidTimestamp;

    toPostAuctionService_.sendEvent(event);
}

void
AdServerConnector::
publishCampaignEvent(const string & label,
                     const Id & bidRequestId,
                     const Id & impId,
                     Date timestamp,
                     const JsonHolder & impressionMeta,
                     const UserIds & ids)
{ 
    recordHit("receivedEvent");
    recordHit("event." + label);

    PostAuctionEvent event;
    event.type = PAE_CAMPAIGN_EVENT;
    event.label = label;
    event.auctionId = bidRequestId;
    event.adSpotId = impId;
    event.timestamp = timestamp;
    event.uids = ids;
    event.metadata = impressionMeta;

    toPostAuctionService_.sendEvent(event);
}

void
AdServerConnector::
publishUserEvent(const string & label,
                 const Id & userId,
                 Date timestamp,
                 const JsonHolder & impressionMeta,
                 const UserIds & ids)
{ 
    recordHit("receivedEvent");
    recordHit("event." + label);
}

namespace {
    typedef std::lock_guard<ML::Spinlock> Guard;
    static ML::Spinlock lock;
    static std::unordered_map<std::string, AdServerConnector::Factory> factories;
}


AdServerConnector::Factory getFactory(std::string const & name) {
    // see if it's already existing
    {
        Guard guard(lock);
        auto i = factories.find(name);
        if (i != factories.end()) return i->second;
    }

    // else, try to load the exchange library
    std::string path = "lib" + name + "_adserver.so";
    void * handle = dlopen(path.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << dlerror() << std::endl;
        throw ML::Exception("couldn't load adserver library " + path);
    }

    // if it went well, it should be registered now
    Guard guard(lock);
    auto i = factories.find(name);
    if (i != factories.end()) return i->second;

    throw ML::Exception("couldn't find adserver name " + name);
}


void AdServerConnector::registerFactory(std::string const & name, Factory callback) {
    Guard guard(lock);
    if (!factories.insert(std::make_pair(name, callback)).second)
        throw ML::Exception("already had an adserver factory registered");
}


std::unique_ptr<AdServerConnector> AdServerConnector::create(
        std::string const & serviceName, 
        std::shared_ptr<ServiceProxies> const & proxies, 
        Json::Value const & json) {
    
    auto name = json.get("type", "unknown").asString();
    auto factory = getFactory(name);
    
    auto nameService  = serviceName;
    if(nameService.empty()) {
        nameService = json.get("name", "adserver").asString();
    }

    return std::unique_ptr<AdServerConnector>(factory(nameService, proxies, json));
}

} // namespace RTBKIT
