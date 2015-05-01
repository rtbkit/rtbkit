/** post_auction_sharding_bench.cc                                 -*- C++ -*-
    Rémi Attab, 30 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Post auction service bench for sharding.

*/

#include "rtbkit/core/post_auction/post_auction_service.h"
#include "rtbkit/core/post_auction/simple_event_matcher.h"
#include "rtbkit/core/banker/null_banker.h"
#include "rtbkit/common/post_auction_proxy.h"
#include "soa/service/zookeeper_configuration_service.h"
#include "soa/service/testing/zookeeper_temporary_server.h"
#include "soa/utils/print_utils.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <thread>
#include <atomic>

using namespace std;
using namespace ML;
using namespace Datacratic;


typedef std::shared_ptr<PostAuctionService> ServicePtr;

/******************************************************************************/
/* CONFIG                                                                     */
/******************************************************************************/

struct Config
{
    Config() :
        shards(1), feeders(1), pauseMs(1), durationSec(10), lossTimeout(15)
    {}

    size_t shards;
    size_t feeders;
    size_t pauseMs;
    size_t durationSec;
    size_t lossTimeout;

};

Config getConfig(int argc, char** argv)
{
    using namespace boost::program_options;

    Config config;

    options_description postAuctionLoop_options("Bench options");
    options_description opt;
    opt.add_options()
        ("shards,s", value<size_t>(&config.shards))
        ("feeders,f", value<size_t>(&config.feeders))
        ("pauseMs,p", value<size_t>(&config.pauseMs))
        ("durationSec,d", value<size_t>(&config.durationSec))
        ("lossTimeout,l", value<size_t>(&config.lossTimeout))
        ("help,h","print this message");

    variables_map vm;
    store(command_line_parser(argc, argv).options(opt).run(), vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << opt << endl;
        exit(1);
    }

    return config;
}


/******************************************************************************/
/* FEEDER                                                                     */
/******************************************************************************/

struct Feeder
{
    Feeder(std::shared_ptr<ServiceProxies> proxies, Config config) :
        done(false), feed(proxies), config(std::move(config))
    {
        feed.init();
    }

    void start()
    {
        runThread = std::thread([=] { run(); });
    }

    void join()
    {
        done = true;
        runThread.join();
    }


private:

    void run()
    {
        while(!done) {
            auto auction = makeAuction();
            feed.sendAuction(auction);

            if (auction->auctionId.hash() % 7 == 0)
                feed.sendEvent(makeWin(auction));

            std::this_thread::sleep_for(std::chrono::milliseconds(config.pauseMs));
        }
    }

    std::string makeBidRequest(Id auctionId) const
    {
        BidRequest bidRequest;

        FormatSet formats;
        formats.push_back(Format(160,600));
        AdSpot spot;
        spot.id = Id(1);
        spot.formats = formats;
        bidRequest.imp.push_back(spot);

        formats[0] = Format(300,250);
        spot.id = Id(2);
        bidRequest.imp.push_back(spot);

        bidRequest.location.countryCode = "CA";
        bidRequest.location.regionCode = "QC";
        bidRequest.location.cityName = "Montreal";
        bidRequest.auctionId = auctionId;
        bidRequest.exchange = "mock";
        bidRequest.language = "en";
        bidRequest.url = Url("http://datacratic.com");
        bidRequest.timestamp = Date::now();

        return bidRequest.toJsonStr();
    }

    Auction::Response makeBid() const
    {
        Auction::Response bid(USD_CPM(2), 1, AccountKey("a.b.c"));
        bid.bidData = Bids::fromJson("{\"bids\":[{\"spotIndex\":0}]}");
        return bid;
    }

    std::shared_ptr<SubmittedAuctionEvent>
    makeAuction() const
    {
        auto event = std::make_shared<SubmittedAuctionEvent>();
        static size_t counter = 0;

        event->auctionId = Id((size_t(this) << 32) + counter++);
        event->adSpotId = Id(1);
        event->lossTimeout = Date::now().plusSeconds(config.lossTimeout);
        event->bidRequestStr = makeBidRequest(event->auctionId);
        event->bidRequestStrFormat = "datacratic";
        event->bidResponse = {};
        event->bidResponse = makeBid();

        return event;
    }


    std::shared_ptr<PostAuctionEvent>
    makeWin(std::shared_ptr<SubmittedAuctionEvent> auction) const
    {
        auto event = std::make_shared<PostAuctionEvent>();

        event->type = PAE_WIN;
        event->auctionId = auction->auctionId;
        event->adSpotId = auction->adSpotId;
        event->winPrice = USD_CPM(1);
        event->timestamp = Date::now();
        event->uids = {};
        event->account = auction->bidResponse.account;
        event->bidTimestamp = Date::now();

        return event;
    }

    std::atomic<bool> done;
    std::thread runThread;

    PostAuctionProxy feed;
    Config config;
};


/******************************************************************************/
/* INIT                                                                       */
/******************************************************************************/

// Note: we want a seperate zmq context to avoid unrealistic zmq fast-paths.
std::shared_ptr<ServiceProxies>
makeProxies(const Config& config)
{
    static auto configService = std::make_shared<InternalConfigurationService>();

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->config = configService;
    proxies->params["postAuctionShards"] = config.shards;

    return proxies;
}

// Will leak feeder objects at the end of the test (who cares)
void initFeeders(std::vector<Feeder*>& feeders, const Config& config)
{
    auto proxies = makeProxies(config);
    for (size_t i = 0; i < config.feeders; ++i) {
        feeders.emplace_back(new Feeder(proxies, config));
        feeders.back()->start();
    }
}

ServicePtr
initService(const Config& config, size_t shard)
{
    std::string name = "bob-" + std::to_string(shard);
    auto service = std::make_shared<PostAuctionService>(makeProxies(config), name);
    service->init(shard);
    service->setBanker(std::make_shared<NullBanker>());
    service->bindTcp();
    service->start();

    return service;
}


/******************************************************************************/
/* REPORT                                                                     */
/******************************************************************************/

PostAuctionService::Stats
getStats(const std::vector<ServicePtr>& services)
{
    PostAuctionService::Stats sum;
    for (const auto& service : services)
        sum += service->stats;
    return sum;
}

float
sampleLoad(const std::vector<ServicePtr>& services)
{
    float load = 0.0;
    for (const auto& service : services)
        load = std::max(load, service->sampleLoad());
    return load;
}

PostAuctionService::Stats
report( const std::vector<ServicePtr>& services,
        double delta,
        const PostAuctionService::Stats& last = PostAuctionService::Stats())
{
    float load = sampleLoad(services);
    auto current = getStats(services);

    auto diff = current;
    diff -= last;

    double bidsThroughput = diff.auctions / delta;
    double eventsThroughput = diff.events / delta;
    double winsThroughput = diff.matchedWins / delta;
    double lossThroughput = diff.matchedLosses / delta;
    double totalThroughput =
        bidsThroughput + eventsThroughput + winsThroughput + lossThroughput;

    std::stringstream ss;
    ss << "\r"
        << "load=" << printValue(load)
        << ", total/sec=" << printValue(totalThroughput)
        << ", bids/sec=" << printValue(bidsThroughput)
        << ", events/sec=" << printValue(eventsThroughput)
        << ", wins/sec= " << printValue(winsThroughput)
        << ", loss/sec=" << printValue(lossThroughput);
    std::cerr << ss.str();

    return current;
}


/******************************************************************************/
/* RUN                                                                        */
/******************************************************************************/

void run(const std::vector<ServicePtr>& services, const Config& config)
{
    auto now = Date::now();
    auto stop = now.plusSeconds(config.durationSec);

    auto stats = report(services, 0.1);

    while ((now = Date::now()) < stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        stats = report(services, 0.1, stats);
    }
}


/******************************************************************************/
/* MAIN                                                                       */
/******************************************************************************/

int main(int argc, char* argv[])
{
    ZmqLogs::print.deactivate();
    PostAuctionService::print.deactivate();
    SimpleEventMatcher::print.deactivate();
    MessageLoopLogs::print.deactivate();

    auto config = getConfig(argc, argv);

    std::vector<ServicePtr> services;
    for (size_t shard = 0; shard < config.shards; ++shard)
        services.emplace_back(initService(config, shard));

    std::vector<Feeder*> feeders;
    initFeeders(feeders, config);

    auto start = getStats(services);
    run(services, config);

    std::cerr << "\n\n"
        << printValue(config.durationSec) << " Duration\n"
        << printValue(config.shards) << " Shards\n"
        << printValue(config.feeders * (1000.0 / config.pauseMs) ) << " Request/sec\n"
        << std::endl;

    auto end = report(services, config.durationSec, start);

    std::cerr << "\n"
        << printValue(end.unmatchedEvents) << "Unmatched\n"
        << printValue(end.errors) << "Errors\n"
        << std::endl;

    // No worth trying to figure out the various shutdown issues.
    _exit(0);
}
