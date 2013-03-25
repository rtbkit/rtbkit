/* exchange_connector.cc
   Jeremy Barnes, 13 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Exchange connector class.
*/

#include <boost/thread/thread.hpp>
#include "exchange_connector.h"
#include "rtbkit/core/router/router.h"


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
}

ExchangeConnector::
ExchangeConnector(const std::string & name,
                  std::shared_ptr<ServiceProxies> proxies)
    : ServiceBase(name, proxies)
{
}

ExchangeConnector::
~ExchangeConnector()
{
}

void
ExchangeConnector::
setRouter(Router * router)
{
    this->router = router;
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
create(const std::string & exchange, std::shared_ptr<Router> router, const std::string & name)
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

    return std::unique_ptr<ExchangeConnector>(factory(router.get(), name));
}

void
ExchangeConnector::
startExchange(std::shared_ptr<Router> router,
              const std::string & exchangeType,
              const Json::Value & exchangeConfig)
{
    auto exchange = ExchangeConnector::
        create(exchangeType, router, exchangeType);
    exchange->configure(exchangeConfig);
    exchange->start();

    router->addExchange(std::move(exchange));
}

void
ExchangeConnector::
startExchange(std::shared_ptr<Router> router,
              const Json::Value & exchangeConfig)
{
    std::string exchangeType = exchangeConfig["exchangeType"].asString();
    startExchange(router, exchangeType, exchangeConfig);
}

} // namespace RTBKIT
