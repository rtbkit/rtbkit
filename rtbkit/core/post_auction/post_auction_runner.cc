/* post_auction_runner.cc
   JS Bejeau , 13 February 2014

   Copyright (c) 2013 Datacratic Inc.  All rights reserved.
*/

#include <string>
#include "post_auction_runner.h"
#include "post_auction_service.h"
#include "rtbkit/core/banker/slave_banker.h"
#include "rtbkit/core/banker/local_banker.h"
#include "rtbkit/core/banker/split_banker.h"
#include "rtbkit/core/banker/null_banker.h"
#include "soa/service/service_utils.h"
#include "soa/service/process_stats.h"
#include "soa/utils/print_utils.h"
#include "jml/utils/file_functions.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace boost::program_options;
using namespace Datacratic;
using namespace RTBKIT;

Logging::Category PostAuctionRunner::print("PostAuctionRunner");
Logging::Category PostAuctionRunner::error("PostAuctionRUnner Error", PostAuctionRunner::print);
Logging::Category PostAuctionRunner::trace("PostAuctionRunner Trace", PostAuctionRunner::print);

static Json::Value loadJsonFromFile(const std::string & filename)
{
    ML::File_Read_Buffer buf(filename);
    return Json::parse(std::string(buf.start(), buf.end()));
}

/************************************************************************/
/* POST AUCTION LOOP RUNNER                                             */
/************************************************************************/
PostAuctionRunner::
PostAuctionRunner() :
    shard(0),
    auctionTimeout(EventMatcher::DefaultAuctionTimeout),
    winTimeout(EventMatcher::DefaultWinTimeout),
    bidderConfigurationFile("rtbkit/examples/bidder-config.json"),
    winLossPipeTimeout(PostAuctionService::DefaultWinLossPipeTimeout),
    campaignEventPipeTimeout(PostAuctionService::DefaultCampaignEventPipeTimeout),
    analyticsOn(false),
    analyticsConnections(1),
    localBankerDebug(false)
{
}

void
PostAuctionRunner::
doOptions(int argc, char ** argv,
        const boost::program_options::options_description & opts)
{
    using namespace boost::program_options;

    options_description postAuctionLoop_options("Post Auction Loop options");
    postAuctionLoop_options.add_options()
        ("bidder,b", value<string>(&bidderConfigurationFile),
         "configuration file with bidder interface data")
        ("shard,s", value<size_t>(&shard),
         "Shard index starting at 0 for this post auction loop")
        ("win-seconds", value<float>(&winTimeout),
         "Timeout for storing win auction")
        ("auction-seconds", value<float>(&auctionTimeout),
         "Timeout to get late win auction")
        ("winlossPipe-seconds", value<int>(&winLossPipeTimeout),
         "Timeout before sending error on WinLoss pipe")
        ("campaignEventPipe-seconds", value<int>(&campaignEventPipeTimeout),
         "Timeout before sending error on CampaignEvent pipe")
        ("analytics,a", bool_switch(&analyticsOn),
         "Send data to analytics logger.")
        ("analytics-connections", value<int>(&analyticsConnections),
         "Number of connections for the analytics publisher.")
        ("forward-auctions", value<std::string>(&forwardAuctionsUri),
         "When provided the PAL will forward all auctions to the given URI.")
        ("local-banker", value<string>(&localBankerUri),
         "address of where the local banker can be found.")
        ("local-banker-debug", bool_switch(&localBankerDebug),
         "enable local banker debug for more precise tracking by account")
        ("banker-choice", value<string>(&bankerChoice),
         "split or local banker can be chosen.");

    options_description all_opt = opts;
    all_opt
        .add(serviceArgs.makeProgramOptions())
        .add(postAuctionLoop_options)
        .add(bankerArgs.makeProgramOptions());

    all_opt.add_options()
        ("help,h","print this message");

    variables_map vm;
    store(command_line_parser(argc, argv)
            .options(all_opt)
            .run(),
            vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << all_opt << endl;
        exit(1);
    }
}

void
PostAuctionRunner::
init()
{
    auto proxies = serviceArgs.makeServiceProxies();
    auto serviceName = serviceArgs.serviceName("PostAuctionLoop");

    auto bidderConfig = loadJsonFromFile(bidderConfigurationFile);

    postAuctionLoop = std::make_shared<PostAuctionService>(proxies, serviceName);
    postAuctionLoop->initBidderInterface(bidderConfig);
    postAuctionLoop->init(shard);

    postAuctionLoop->setWinTimeout(winTimeout);
    postAuctionLoop->setAuctionTimeout(auctionTimeout);
    postAuctionLoop->setWinLossPipeTimeout(winLossPipeTimeout);
    postAuctionLoop->setCampaignEventPipeTimeout(campaignEventPipeTimeout);

    LOG(print) << "win timeout is " << winTimeout << std::endl;
    LOG(print) << "auction timeout is " << auctionTimeout << std::endl;
    LOG(print) << "winLoss pipe timeout is " << winLossPipeTimeout << std::endl;
    LOG(print) << "campaignEvent pipe timeout is " << campaignEventPipeTimeout << std::endl;

    if (localBankerUri != "") {
        localBanker = make_shared<LocalBanker>(proxies, POST_AUCTION, postAuctionLoop->serviceName());
        localBanker->init(localBankerUri);
        localBanker->setDebug(localBankerDebug);
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

        slaveBanker = bankerArgs.makeBanker(proxies, postAuctionLoop->serviceName() + ".slaveBanker");
        banker = make_shared<SplitBanker>(slaveBanker, localBanker, campaignSet);
        postAuctionLoop->addSource("local-banker", *localBanker);
        postAuctionLoop->addSource("slave-banker", *slaveBanker);

    } else if (localBanker && bankerChoice == "local") {
        banker = localBanker;
        postAuctionLoop->addSource("local-banker", *localBanker);

    } else if (bankerChoice == "null") {
        banker = make_shared<NullBanker>(true, postAuctionLoop->serviceName());

    } else {
        slaveBanker = bankerArgs.makeBanker(proxies, postAuctionLoop->serviceName() + ".slaveBanker");
        banker = slaveBanker;
        postAuctionLoop->addSource("slave-banker", *slaveBanker);
    }
    postAuctionLoop->setBanker(banker);

    if (analyticsOn) {
        const auto & analyticsUri = proxies->params["analytics-uri"].asString();
        if (!analyticsUri.empty()) {
            postAuctionLoop->initAnalytics(analyticsUri, analyticsConnections);
        }
        else
            LOG(print) << "analytics-uri is not in the config" << endl;
    }

    postAuctionLoop->bindTcp();

    if (!forwardAuctionsUri.empty())
        postAuctionLoop->forwardAuctions(forwardAuctionsUri);
}

void
PostAuctionRunner::
start()
{
    postAuctionLoop->start();
}

void
PostAuctionRunner::
shutdown()
{
    postAuctionLoop->shutdown();
    if (slaveBanker) slaveBanker->shutdown();
    if (localBanker) localBanker->shutdown();
}



PostAuctionService::Stats
report( const PostAuctionService& service,
        double delta,
        const PostAuctionService::Stats& last = PostAuctionService::Stats())
{
    auto current = service.stats;

    auto diff = current;
    diff -= last;

    double bidsThroughput = diff.auctions / delta;
    double eventsThroughput = diff.events / delta;
    double winsThroughput = diff.matchedWins / delta;
    double lossThroughput = diff.matchedLosses / delta;

    std::stringstream ss;
    ss << std::endl
        << printValue(bidsThroughput) << " bids/sec\n"
        << printValue(eventsThroughput) << " events/sec\n"
        << printValue(winsThroughput) << " wins/sec\n"
        << printValue(lossThroughput) << " loss/sec\n"
        << printValue(current.unmatchedEvents) << " unmatched\n"
        << printValue(current.errors) << " errors\n";
    LOG(PostAuctionService::print) << ss.str();

    return current;
}

void setMemoryLimit()
{
    rlimit64 currentLimit;
    int status = getrlimit64(RLIMIT_AS, &currentLimit);
    if(status != 0)
        throw ML::Exception("Failed to get the current system limits");

    currentLimit.rlim_cur = 1UL << 36;// 64G
    if(setrlimit64(RLIMIT_AS, &currentLimit) != 0)
        throw ML::Exception("Failed to set the current system limits to 512G");
}

int main(int argc, char ** argv)
{
    setMemoryLimit();

    PostAuctionRunner runner;

    runner.doOptions(argc, argv);
    runner.init();
    runner.start();

    auto stats = report(*runner.postAuctionLoop, 0.1);
    ProcessStats lastStats;

    auto onStat = [&] (std::string key, double val) {
        runner.postAuctionLoop->recordStableLevel(val, key);
    };

    for (size_t i = 0;; ++i) {
        ML::sleep(1.0);

        ProcessStats curStats;
        ProcessStats::logToCallback(onStat, lastStats, curStats, "process");
        lastStats = curStats;

        if (i % 10 == 0)
            stats = report(*runner.postAuctionLoop, 10.0, stats);
    }

}
