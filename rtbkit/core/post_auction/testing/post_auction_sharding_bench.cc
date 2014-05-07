/** post_auction_sharding_bench.cc                                 -*- C++ -*-
    Rémi Attab, 30 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Post auction service bench for sharding.

*/

#include "rtbkit/core/post_auction/post_auction_service.h"
#include "rtbkit/core/banker/null_banker.h"
#include "rtbkit/common/messages.h"
#include "soa/service/zookeeper_configuration_service.h"
#include "soa/service/testing/zookeeper_temporary_server.h"
#include "soa/utils/print_utils.h"

#include <thread>
#include <atomic>

using namespace std;
using namespace ML;
using namespace Datacratic;


/******************************************************************************/
/* CONFIG                                                                     */
/******************************************************************************/

enum {
    NumShards = 1,
    LossTimeout = 15,

    NumFeeders = 1,
    PauseMs = 1,
    DurationSec = 10
};


const char* requestTemplate = "{\"id\":\"%s\"}";

/******************************************************************************/
/* FEEDER                                                                     */
/******************************************************************************/

struct Feeder
{
    Feeder(std::shared_ptr<zmq::context_t> context) :
        done(false), feed(context)
    {}

    void init(std::shared_ptr<ConfigurationService> config)
    {
        feed.init(config, ZMQ_XREQ);
        feed.connectToServiceClass("rtbPostAuctionService", "events");
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
            sendAuction(auction);

            if (auction.auctionId.hash() % 7 == 0)
                sendWin(makeWin(auction));

            std::this_thread::sleep_for(std::chrono::milliseconds(PauseMs));
        }
    }


    SubmittedAuctionEvent makeAuction() const
    {
        SubmittedAuctionEvent event;
        static size_t counter = 0;
        
        event.auctionId = Id((size_t(this) << 32) + counter++);
        event.adSpotId = Id(0);
        event.lossTimeout = Date::now().plusSeconds(LossTimeout);
        event.bidRequestStr = ML::format(requestTemplate, event.auctionId.toString());
        event.bidRequestStrFormat = "datacratic";
        event.bidResponse = {};
        event.bidResponse.account = AccountKey("a.b.c");
        event.bidResponse.price = USD_CPM(2);
        event.bidResponse.bidData = "{\"bids\":[{\"spotIndex\":0}]}";

        return event;
    }

    void sendAuction(const SubmittedAuctionEvent& event)
    {
        Message<SubmittedAuctionEvent> message(std::move(event));
        feed.sendMessage("AUCTION", message.toString());
    }


    PostAuctionEvent makeWin(const SubmittedAuctionEvent& auction) const
    {
        PostAuctionEvent event;

        event.type = PAE_WIN;
        event.auctionId = auction.auctionId;
        event.adSpotId = auction.adSpotId;
        event.winPrice = USD_CPM(1);
        event.timestamp = Date::now();
        event.uids = {};
        event.account = auction.bidResponse.account;
        event.bidTimestamp = Date::now();

        return event;
    }

    void sendWin(const PostAuctionEvent& event)
    {
        feed.sendMessage("WIN", ML::DB::serializeToString(event));
    }

    std::atomic<bool> done;
    std::thread runThread;

    ZmqNamedProxy feed;

};


/******************************************************************************/
/* INIT                                                                       */
/******************************************************************************/

// Will leak feeder objects at the end of the test (who cares)
void initFeeders(
        std::vector<Feeder*>& feeders, 
        std::shared_ptr<ConfigurationService> config)
{
    auto zmq = std::make_shared<zmq::context_t>(1);

    for (size_t i = 0; i < NumFeeders; ++i) {
        feeders.emplace_back(new Feeder(zmq));
        feeders.back()->init(config);
        feeders.back()->start();
    }
}

std::shared_ptr<PostAuctionService>
initService(std::shared_ptr<ServiceProxies> proxies)
{
    auto service = std::make_shared<PostAuctionService>(proxies, "bob");
    service->init(NumShards);
    service->setBanker(std::make_shared<NullBanker>());
    service->bindTcp();
    service->start();

    return service;
}


/******************************************************************************/
/* REPORT                                                                     */
/******************************************************************************/

PostAuctionService::Stats
report( const PostAuctionService& service, 
        double delta,
        const PostAuctionService::Stats& last = PostAuctionService::Stats())
{
    auto current = service.stats;

    auto diff = current;
    diff -= current;

    double bidsThroughput = diff.auctions / delta;
    double eventsThroughput = diff.events / delta;
    double winsThroughput = diff.matchedWins / delta;
    double lossThroughput = diff.matchedLosses / delta;

    std::stringstream ss;
    ss << "\r"
        << "bids/sec=" << printValue(bidsThroughput)
        << ", events/sec=" << printValue(eventsThroughput)
        << ", wins/sec= " << printValue(winsThroughput)
        << ", loss/sec=" << printValue(lossThroughput)
        << ", unmatched=" << printValue(current.unmatchedEvents)
        << ", errors=" << printValue(current.errors);
    std::cerr << ss.str();

    return current;
}


/******************************************************************************/
/* RUN                                                                        */
/******************************************************************************/

void run(const PostAuctionService& service)
{
    auto now = Date::now();
    auto stop = now.plusSeconds(DurationSec);
    
    auto stats = report(service, 0.1);
    
    while ((now = Date::now()) < stop) {
        stats = report(service, 0.1, stats);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


/******************************************************************************/
/* MAIN                                                                       */
/******************************************************************************/

int main(int argc, char* argv[])
{
    ZmqLogs::print.deactivate();

    auto proxies = std::make_shared<ServiceProxies>();
    auto service = initService(proxies);

    std::vector<Feeder*> feeders;
    initFeeders(feeders, proxies->config);

    run(*service);

    service->shutdown();
    for (auto& feeder : feeders) feeder->join();

    std::cerr << "\n\n"
        << printValue(DurationSec) << " Duration\n"
        << printValue(NumShards) << " Shards\n"
        << printValue(NumFeeders * (1000.0 / PauseMs) ) << " Request/sec\n"
        << std::endl;
    
    report(*service, DurationSec);
}
