/* exchange_connector.cc
   Jeremy Barnes, 13 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Exchange connector class.
*/

#include "exchange_connector.h"
#include "rtbkit/core/router/router.h"

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
    onNewAuction  = [=] (std::shared_ptr<Auction> a) {cerr << "WARNING: an auction was lost into the void" << endl; };
    onAuctionDone = [=] (std::shared_ptr<Auction> a) {};
    
    numServingRequest = 0;
    numAuctions = 0;
    acceptAuctionProbability = 1.0;
}

ExchangeConnector::
ExchangeConnector(const std::string & name,
                  std::shared_ptr<ServiceProxies> proxies)
    : ServiceBase(name, proxies)
{
    onNewAuction  = [=] (std::shared_ptr<Auction> a) {cerr << "WARNING: an auction was lost into the void"; };
    onAuctionDone = [=] (std::shared_ptr<Auction> a) {};

    numServingRequest = 0;
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

std::shared_ptr<BidSource>
ExchangeConnector::
getBidSource() const
{
    return 0;
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
typedef std::unordered_map<std::string, ExchangeConnector::Factory> Factories;
static Factories factories;
typedef boost::lock_guard<ML::Spinlock> Guard;
static ML::Spinlock lock;
} // file scope

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

    Factory factory;
    {
        Guard guard(lock);
        auto it = factories.find(exchange);
        if (it == factories.end())
            throw ML::Exception("couldn't find exchange factory for exchange "
                                + exchange);
        factory = it->second;
    }

    return std::unique_ptr<ExchangeConnector>(factory(&owner, name));
}

} // namespace RTBKIT
