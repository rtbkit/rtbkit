/** mock_exchange.cc                                 -*- C++ -*-
    Rémi Attab, 18 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Implementation of the mock exchange.

*/

#include "mock_exchange.h"

#include "rtbkit/core/post_auction/post_auction_loop.h"
#include "soa/service/http_header.h"

#include <array>

using namespace std;
using namespace ML;

namespace RTBKIT {

/******************************************************************************/
/* MOCK EXCHANGE                                                              */
/******************************************************************************/

MockExchange::
MockExchange(Datacratic::ServiceProxyArguments & args, const std::string& name) :
    ServiceBase(name, args.makeServiceProxies()),
    running(0) {
}


MockExchange::
MockExchange(const shared_ptr<ServiceProxies> proxies, const string& name) :
    ServiceBase(name, proxies),
    running(0) {
}


MockExchange::
~MockExchange()
{
    threads.join_all();
}


void
MockExchange::
start(size_t threadCount,
      size_t numBidRequests,
      std::vector<NetworkAddress> const & bids,
      std::vector<NetworkAddress> const & wins)
{
    try {
        running = threadCount;

        auto startWorker = [=](NetworkAddress bid, NetworkAddress win) {
            Worker worker(this, std::move(bid), std::move(win));
            if(numBidRequests) {
                worker.run(numBidRequests);
            }
            else {
                worker.run();
            }

            ML::atomic_dec(running);
        };

        for(size_t i = 0; i != threadCount; ++i) {
            int a = i % bids.size();
            int b = i % wins.size();

            std::cerr << "worker " << i
                      << " connects to bid=" << bids[a].host << ":" << bids[a].port
                      << " connects to win=" << wins[b].host << ":" << wins[b].port
                      << std::endl;

            threads.create_thread(std::bind(startWorker, bids[a], wins[b]));
        }
    }
    catch (const exception& ex) {
        cerr << "got exception on request: " << ex.what() << endl;
    }
}


MockExchange::Worker::
Worker(MockExchange * exchange, NetworkAddress bid, NetworkAddress win) :
    exchange(exchange),
    bids(std::move(bid)),
    wins(std::move(win)),
    rng(random())
{
}


void
MockExchange::Worker::
run() {
    for(;;) {
        bid();
    }
}


void
MockExchange::Worker::
run(size_t requests) {
    for(size_t i = 0; i != requests; ++i) {
        bid();
    }
}


void
MockExchange::Worker::bid() {
    BidRequest bidRequest = bids.makeBidRequest();
    exchange->recordHit("requests");

    for (;;) {
        bids.sendBidRequest(bidRequest);
        exchange->recordHit("sent");

        auto response = bids.recvBid();
        if (!response.first) continue;
        exchange->recordHit("bids");

        vector<ExchangeSource::Bid> bids = response.second;

        for (auto & bid : bids) {
            if(bid.maxPrice == 0) continue;

            auto ret = isWin(bidRequest, bid);
            if (!ret.first) continue;

            wins.sendWin(bidRequest, bid, ret.second);
            exchange->recordHit("wins");

            wins.sendImpression(bidRequest, bid);
            exchange->recordHit("impressions");

            if (!isClick(bidRequest, bid)) continue;
            wins.sendClick(bidRequest, bid);
            exchange->recordHit("clicks");
        }

        break;
    }
}


pair<bool, Amount>
MockExchange::Worker::isWin(const BidRequest&, const ExchangeSource::Bid& bid)
{
    if (rng.random01() >= 0.1)
        return make_pair(false, Amount());

    return make_pair(true, MicroUSD_CPM(bid.maxPrice * 0.6 + bid.maxPrice * rng.random01() * 0.4));
}


bool
MockExchange::Worker::isClick(const BidRequest&, const ExchangeSource::Bid&)
{
    return rng.random01() <= 0.1;
}


} // namepsace RTBKIT
