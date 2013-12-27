/* exchange_connector.cc
   Jeremy Barnes, 13 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Exchange connector class.
*/

#include "exchange_connector.h"
#include <dlfcn.h>

using namespace std;

namespace RTBKIT {

ExchangeConnector::CampaignCompatibility::
CampaignCompatibility(const AgentConfig & config)
    : creatives(config.creatives.size())
{
}


/*****************************************************************************/
/* EXCHANGE CONNECTOR                                                        */
/*****************************************************************************/

ExchangeConnector::
ExchangeConnector(const std::string & name,
                  ServiceBase & parent)
    : ServiceBase(name, parent)
{
    onNewAuction  = [=] (std::shared_ptr<Auction> a) {
        cerr << "WARNING: an auction was lost into the void.  exchange=" << name <<
            ", auction=" << a->id << endl; };
    onAuctionDone = [=] (std::shared_ptr<Auction> a) {};

    numRequests = 0;
    numAuctions = 0;
    acceptAuctionProbability = 1.0;
}

ExchangeConnector::
ExchangeConnector(const std::string & name,
                  std::shared_ptr<ServiceProxies> proxies)
    : ServiceBase(name, proxies)
{
    onNewAuction  = [=] (std::shared_ptr<Auction> a) {
        cerr << "WARNING: an auction was lost into the void.  exchange=" << name <<
        ", auction=" << a->id << endl; };
    onAuctionDone = [=] (std::shared_ptr<Auction> a) {};

    numRequests = 0;
    numAuctions = 0;
    acceptAuctionProbability = 1.0;
}

ExchangeConnector::
~ExchangeConnector()
{
}

void
ExchangeConnector::
start()
{
}

void
ExchangeConnector::
shutdown()
{
}

WinCostModel
ExchangeConnector::
getWinCostModel(Auction const & auction,
                AgentConfig const & agent)
{
    return WinCostModel();
}

std::string
ExchangeConnector::
getBidSourceConfiguration() const
{
    return "{\"type\":\"unknown\"}";
}

ExchangeConnector::ExchangeCompatibility
ExchangeConnector::
getCampaignCompatibility(const AgentConfig & config,
                         bool includeReasons) const
{
    ExchangeCompatibility result;
    result.setCompatible();
    return result;
}

ExchangeConnector::ExchangeCompatibility
ExchangeConnector::
getCreativeCompatibility(const Creative & creative,
                         bool includeReasons) const
{
    ExchangeCompatibility result;
    result.setCompatible();
    return result;
}

bool
ExchangeConnector::
bidRequestPreFilter(const BidRequest & request,
                    const AgentConfig & config,
                    const void * info) const
{
    return true;
}

bool
ExchangeConnector::
bidRequestPostFilter(const BidRequest & request,
                     const AgentConfig & config,
                     const void * info) const
{
    return true;
}

bool
ExchangeConnector::
bidRequestCreativeFilter(const BidRequest & request,
                         const AgentConfig & config,
                         const void * info) const
{
    return true;
}

namespace {
typedef std::lock_guard<ML::Spinlock> Guard;

static ML::Spinlock lock;
static std::unordered_map<std::string, ExchangeConnector::Factory> factories;
} // file scope

ExchangeConnector::Factory
getFactory(std::string const & name) {
    // see if it's already existing
    {
        Guard guard(lock);
        auto i = factories.find(name);
        if (i != factories.end()) return i->second;
    }

    // else, try to load the exchange library
    std::string path = "lib" + name + "_exchange.so";
    void * handle = dlopen(path.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << dlerror() << std::endl;
        throw ML::Exception("couldn't find exchange connector library " + path);
    }

    // if it went well, it should be registered now
    Guard guard(lock);
    auto i = factories.find(name);
    if (i != factories.end()) return i->second;

    throw ML::Exception("couldn't find exchange connector named " + name);
}

void
ExchangeConnector::
registerFactory(const std::string & exchange, Factory factory)
{
    Guard guard(lock);
    if (!factories.insert(make_pair(exchange, factory)).second)
        throw ML::Exception("already had a bid request factory registered");
}

std::unique_ptr<ExchangeConnector>
ExchangeConnector::
create(const std::string & exchange, ServiceBase & owner, const std::string & name)
{
    auto factory = getFactory(exchange);
    return std::unique_ptr<ExchangeConnector>(factory(&owner, name));
}

} // namespace RTBKIT
