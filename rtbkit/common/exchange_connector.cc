/* exchange_connector.cc
   Jeremy Barnes, 13 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Exchange connector class.
*/

#include "exchange_connector.h"

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
, hasCurrencyConfigured_(false)
, currency_("USD")
, currencyCode_(CurrencyCode::CC_USD)
{
    onNewAuction  = [=] (std::shared_ptr<Auction> a) {
        cerr << "WARNING: an auction was lost into the void.  exchange=" << name <<
            ", auction=" << a->id << endl; };
    onAuctionDone = [=] (std::shared_ptr<Auction> a) {};
    onAuctionError = [=] (const std::string & channel,
                          std::shared_ptr<Auction> auction,
                          const string & message) {};

    numRequests = 0;
    numAuctions = 0;
    acceptAuctionProbability = 1.0;
}

ExchangeConnector::
ExchangeConnector(const std::string & name,
                  std::shared_ptr<ServiceProxies> proxies)
: ServiceBase(name, proxies)
, hasCurrencyConfigured_(false)
, currency_("USD")
, currencyCode_(CurrencyCode::CC_USD)
{
    onNewAuction  = [=] (std::shared_ptr<Auction> a) {
        cerr << "WARNING: an auction was lost into the void.  exchange=" << name <<
        ", auction=" << a->id << endl; };
    onAuctionDone = [=] (std::shared_ptr<Auction> a) {};
    onAuctionError = [=] (const std::string & channel,
                          std::shared_ptr<Auction> auction,
                          const string & message) {};

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
configure(const Json::Value & parameters)
{
    const auto & currency = parameters["currency"];
    if (currency != Json::Value::null) {
        currency_ = currency.asString();
        try
        {
            // try to parse a standard currency
            currencyCode_ =
                parseCurrencyCode(currency_);
        }
        catch (const ML::Exception &)
        {
            // Not a standard currency, let's try a RTBKIT's currency
            currencyCode_ = Amount::parseCurrency(currency_);
            // get standard currency str !
            currency_ = toString(currencyCode_);
        }

        hasCurrencyConfigured_ = true;
    }
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

std::unique_ptr<ExchangeConnector>
ExchangeConnector::
create(const std::string & exchange, ServiceBase & owner, const std::string & name)
{
    auto factory = PluginInterface<ExchangeConnector>::getPlugin(exchange);
    return std::unique_ptr<ExchangeConnector>(factory(&owner, name));
}

} // namespace RTBKIT
