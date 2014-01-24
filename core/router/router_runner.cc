/* router_runner.cc
   Jeremy Barnes, 13 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Tool to run the router.
*/

#include "router_runner.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>

#include "rtbkit/core/router/router.h"
#include "rtbkit/core/banker/slave_banker.h"
#include "jml/arch/timers.h"
#include "jml/utils/file_functions.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

static inline Json::Value loadJsonFromFile(const std::string & filename)
{
    ML::File_Read_Buffer buf(filename);
    return Json::parse(std::string(buf.start(), buf.end()));
}


/*****************************************************************************/
/* ROUTER RUNNER                                                             */
/*****************************************************************************/


RouterRunner::
RouterRunner() :
    exchangeConfigurationFile("examples/router-config.json"),
    lossSeconds(15.0),
    noPostAuctionLoop(false),
    logAuctions(false),
    logBids(false)
{
}

namespace RTBKIT {

void validate(boost::any & v, const std::vector<std::string> & values, Amount *, int)
{
    namespace po = boost::program_options;
    po::validators::check_first_occurrence(v);
    const std::string & amountString = po::validators::get_single_string(values);

    auto idx = amountString.find(':');
    if (idx == std::string::npos) {
        throw po::validation_error(po::validation_error::invalid_option_value);
    }

    try
    {
        std::string currency = amountString.substr(0, idx);
        std::string value = amountString.substr(idx + 1);
        double dvalue = std::stod(value);
        CurrencyCode code = RTBKIT::parseCurrencyCode(currency);
        v = boost::any(Amount(code, dvalue));
    }
    catch (const std::exception & e)
    {
        throw po::validation_error(po::validation_error::invalid_option_value,
                                   e.what());
    }
}

} // namespace RTBKIT

void
RouterRunner::
doOptions(int argc, char ** argv,
          const boost::program_options::options_description & opts)
{
    using namespace boost::program_options;

    options_description router_options("Router options");
    router_options.add_options()
        ("loss-seconds,l", value<float>(&lossSeconds),
         "number of seconds after which a loss is assumed")
        ("no-post-auction-loop", bool_switch(&noPostAuctionLoop),
         "don't connect to the post auction loop")
        ("log-uri", value<vector<string> >(&logUris),
         "URI to publish logs to")
        ("exchange-configuration,x", value<string>(&exchangeConfigurationFile),
         "configuration file with exchange data")
        ("log-auctions", value<bool>(&logAuctions)->zero_tokens(),
         "log auction requests")
        ("log-bids", value<bool>(&logBids)->zero_tokens(),
         "log bid responses")
        ("max-bid-price", value(&maxBidPrices),
         "maximum bid price accepted by router. It should be CURRENCY:amount");

    options_description all_opt = opts;
    all_opt
        .add(serviceArgs.makeProgramOptions())
        .add(router_options);
    all_opt.add_options()
        ("help,h", "print this message");
    
    variables_map vm;
    store(command_line_parser(argc, argv)
          .options(all_opt)
          //.positional(p)
          .run(),
          vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << all_opt << endl;
        exit(1);
    }
}

void
RouterRunner::
init()
{
    auto proxies = serviceArgs.makeServiceProxies();
    auto serviceName = serviceArgs.serviceName("router");

    exchangeConfig = loadJsonFromFile(exchangeConfigurationFile);

    auto connectPostAuctionLoop = !noPostAuctionLoop;
    router = std::make_shared<Router>(proxies,
                                      serviceName,
                                      lossSeconds,
                                      connectPostAuctionLoop,
                                      logAuctions,
                                      logBids,
                                      maxBidPrices);
    router->init();

    banker = std::make_shared<SlaveBanker>(proxies->zmqContext,
                                           proxies->config,
                                           router->serviceName() + ".slaveBanker");

    router->setBanker(banker);
    router->bindTcp();
}

void
RouterRunner::
start()
{
    banker->start();
    router->start();

    // Start all exchanges
    for (auto & exchange: exchangeConfig)
        router->startExchange(exchange);
}

void
RouterRunner::
shutdown()
{
    router->shutdown();
    banker->shutdown();
}

int main(int argc, char ** argv)
{
    RouterRunner runner;

    runner.doOptions(argc, argv);
    runner.init();
    runner.start();

    runner.router->forAllExchanges([](std::shared_ptr<ExchangeConnector> const & item) {
        item->enableUntil(Date::positiveInfinity());
    });

    for (;;) {
        ML::sleep(10.0);
    }
}
