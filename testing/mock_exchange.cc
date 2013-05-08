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
start(size_t threadCount, size_t numBidRequests, std::vector<int> const & bidPorts, std::vector<int> const & winPorts)
{
    try {
        running = threadCount;

        auto startWorker = [=](size_t i, int bidPort, int winPort) {
            Worker worker(this, i, bidPort, winPort);
            if(numBidRequests) {
                worker.run(numBidRequests);
            }
            else {
                worker.run();
            }

            ML::atomic_dec(running);
        };

        int bp = 0;
        int wp = 0;

        for(size_t i = 0; i != threadCount; ++i) {
            int bidPort = bidPorts[bp++ % bidPorts.size()];
            int winPort = winPorts[wp++ % winPorts.size()];
            threads.create_thread(std::bind(startWorker, i, bidPort, winPort));
        }
    }
    catch (const exception& ex) {
        cerr << "got exception on request: " << ex.what() << endl;
    }
}


MockExchange::Worker::
Worker(MockExchange * exchange, size_t id, int bidPort, int winPort) : exchange(exchange), bids(bidPort, id), wins(winPort), rng(random()) {
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
            auto ret = isWin(bidRequest, bid);
            if (!ret.first) continue;
            wins.sendWin(bidRequest, bid, ret.second);
            exchange->recordHit("wins");
        }

        break;
    }
}


pair<bool, Amount>
MockExchange::Worker::isWin(const BidRequest&, const ExchangeSource::Bid& bid)
{
    if (rng.random01() >= 0.1)
        return make_pair(false, Amount());

    return make_pair(true, MicroUSD_CPM(bid.maxPrice * rng.random01()));
}


} // namepsace RTBKIT
