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

#include "rtbkit/common/bidder_interface.h"
#include "rtbkit/core/router/router.h"
#include "rtbkit/core/banker/slave_banker.h"
#include "rtbkit/core/banker/local_banker.h"
#include "rtbkit/core/banker/split_banker.h"
#include "rtbkit/core/banker/null_banker.h"
#include "soa/service/process_stats.h"
#include "jml/arch/timers.h"
#include "jml/utils/file_functions.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

Logging::Category RouterRunner::print("RouterRunner");
Logging::Category RouterRunner::error("RouterRUnner Error", RouterRunner::print);
Logging::Category RouterRunner::trace("RouterRunner Trace", RouterRunner::print);

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
    exchangeConfigurationFile("rtbkit/examples/router-config.json"),
    bidderConfigurationFile("rtbkit/examples/bidder-config.json"),
    lossSeconds(15.0),
    noPostAuctionLoop(false),
    noBidProb(false),
    logAuctions(false),
    logBids(false),
    maxBidPrice(40),
    localBankerDebug(false),
    slowModeTimeout(MonitorClient::DefaultCheckTimeout),
    slowModeTolerance(MonitorClient::DefaultTolerance),
    slowModeMoneyLimit(""),
    analyticsOn(false),
    analyticsConnections(1),
    augmentationWindowms(5),
    dableSlowMode(false),
    enableJsonFiltersFile("")
{
}

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
        ("slowModeTimeout", value<int>(&slowModeTimeout),
         "number of seconds after which the system consider to be in SlowMode")
        ("slowModeTolerance", value<int>(&slowModeTolerance),
         "number of seconds allowed to bid normally since last successful monitor check")
        ("no-post-auction-loop", bool_switch(&noPostAuctionLoop),
         "don't connect to the post auction loop")
        ("no-bidprob", bool_switch(&noBidProb),
         "don't use bid probability to sample the traffic")
        ("log-uri", value<vector<string> >(&logUris),
         "URI to publish logs to")
        ("exchange-configuration,x", value<string>(&exchangeConfigurationFile),
         "configuration file with exchange data")
        ("bidder,b", value<string>(&bidderConfigurationFile),
         "configuration file with bidder interface data")
        ("log-auctions", value<bool>(&logAuctions)->zero_tokens(),
         "log auction requests")
        ("log-bids", value<bool>(&logBids)->zero_tokens(),
         "log bid responses")
        ("max-bid-price", value(&maxBidPrice),
         "maximum bid price accepted by router")
        ("slow-mode-money-limit,s", value<string>(&slowModeMoneyLimit)->default_value("100000USD/1M"),
         "Amout of money authorized per second when router enters slow mode (default is 100000USD/1M).")
        ("analytics,a", bool_switch(&analyticsOn),
         "Send data to analytics logger.")
        ("analytics-connections", value<int>(&analyticsConnections),
         "Number of connections for the analytics publisher.")
        ("local-banker", value<string>(&localBankerUri),
         "address of where the local banker can be found.")
        ("local-banker-debug", bool_switch(&localBankerDebug),
         "enable local banker debug for more precise tracking by account")
        ("banker-choice", value<string>(&bankerChoice),
         "split or local banker can be chosen.")
         ("augmenter-timeout",value<int>(&augmentationWindowms),
         "configure the augmenter  timeout (in milliseconds)")
        ("no slow mode", value<bool>(&dableSlowMode)->zero_tokens(),
         "disable the slow mode.")
        ("filters-configuration", value<string>(&enableJsonFiltersFile),
          "configuration file with enabled filters data");

    options_description all_opt = opts;
    all_opt
        .add(serviceArgs.makeProgramOptions())
        .add(router_options)
        .add(bankerArgs.makeProgramOptions());
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
    bidderConfig = loadJsonFromFile(bidderConfigurationFile);

    if (!enableJsonFiltersFile.empty())
        filtersConfig = loadJsonFromFile(enableJsonFiltersFile);

    const auto amountSlowModeMoneyLimit = Amount::parse(slowModeMoneyLimit);
    const auto maxBidPriceAmount = USD_CPM(maxBidPrice);

    if (maxBidPriceAmount > amountSlowModeMoneyLimit) {
        THROW(error) << "max-bid-price and slow-mode-money-limit "
            << "configuration is invalid" << endl
            << "usage:  max-bid-price must be lower or equal to the "
            << "slow-mode-money-limit." << endl
            << "max-bid-price= " << maxBidPriceAmount << endl
            << "slow-mode-money-limit= " << amountSlowModeMoneyLimit <<endl;
    }

    Seconds augmentationWindow = std::chrono::milliseconds(augmentationWindowms);

    auto connectPostAuctionLoop = !noPostAuctionLoop;
    auto enableBidProbability = !noBidProb;
    router = std::make_shared<Router>(proxies, serviceName, lossSeconds,
                                      connectPostAuctionLoop,
                                      enableBidProbability,
                                      logAuctions, logBids,
                                      USD_CPM(maxBidPrice),
                                      slowModeTimeout,
                                      amountSlowModeMoneyLimit,
                                      augmentationWindow,
                                      exchangeConfig.get("numAugmentationLoops", 1).asInt());
    router->slowModeTolerance = slowModeTolerance;
    router->initBidderInterface(bidderConfig);
    if (dableSlowMode) {
       router->unsafeDisableSlowMode();
    }
    if (analyticsOn) {
        const auto & analyticsUri = proxies->params["analytics-uri"].asString();
        if (!analyticsUri.empty()) {
            router->initAnalytics(analyticsUri, analyticsConnections);
        }
        else
            LOG(print) << "analytics-uri is not in the config" << endl;
    }
    router->init();

    if (localBankerUri != "") {
        localBanker = make_shared<LocalBanker>(proxies, ROUTER, router->serviceName());
        localBanker->init(localBankerUri);
        localBanker->setDebug(localBankerDebug);
        localBanker->setSpendRate(bankerArgs.spendRate());
    }
    if (localBanker && bankerChoice == "split") {
        unordered_set<string> campaignSet;
        if (proxies->params.isMember("goBankerCampaigns")) {
            Json::Value campaigns = proxies->params["goBankerCampaigns"];
            if (campaigns.isArray()) {
                for (auto cmp : campaigns) {
                    campaignSet.insert(cmp.asString());
                }
            }
        }
        slaveBanker = bankerArgs.makeBanker(proxies, router->serviceName() + ".slaveBanker");
        banker = make_shared<SplitBanker>(slaveBanker, localBanker, campaignSet);
    } else if (localBanker && bankerChoice == "local") {
        banker = localBanker;
    } else if (bankerChoice == "null") {
        banker = make_shared<NullBanker>(true, router->serviceName());
    } else {
        slaveBanker = bankerArgs.makeBanker(proxies, router->serviceName() + ".slaveBanker");
        banker = slaveBanker;
    }

    router->setBanker(banker);
    router->initExchanges(exchangeConfig);
    router->initFilters(filtersConfig);
    router->bindTcp();
}

void
RouterRunner::
start()
{
    if (slaveBanker) slaveBanker->start();
    if (localBanker) localBanker->start();
    router->start();
}

void
RouterRunner::
shutdown()
{
    router->shutdown();
    if (slaveBanker) slaveBanker->shutdown();
    if (localBanker) localBanker->shutdown();
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

    ProcessStats lastStats;
    auto onStat = [&] (std::string key, double val) {
        runner.router->recordStableLevel(val, key);
    };

    for (;;) {
        ML::sleep(1.0);

        ProcessStats curStats;
        ProcessStats::logToCallback(onStat, lastStats, curStats, "process");
        lastStats = curStats;
    }
}
