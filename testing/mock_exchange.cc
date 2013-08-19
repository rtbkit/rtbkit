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
~MockExchange() {
    threads.join_all();
}


void
MockExchange::
start(Json::Value const & configuration) {
    auto workers = configuration["workers"];

    for(auto i = workers.begin(), end = workers.end(); i != end; ++i) {
        auto json = *i;
        auto count = json.get("threads", 1).asInt();

        for(auto j = 0; j != count; ++j) {
            std::cerr << "starting worker " << running << std::endl;
            ML::atomic_inc(running);

            auto bid = json["bids"];
            auto win = json["wins"];
            threads.create_thread([=]() {
                Worker worker(this, bid, win);
                worker.run();

                ML::atomic_dec(running);
            });
        }
    }
}


void
MockExchange::
add(BidSource * bid, WinSource * win) {
    std::cerr << "starting worker " << running << std::endl;
    ML::atomic_inc(running);

    threads.create_thread([=]() {
        Worker worker(this, bid, win);
        worker.run();

        ML::atomic_dec(running);
    });
}


MockExchange::Worker::
Worker(MockExchange * exchange, BidSource * bid, WinSource * win) :
    exchange(exchange),
    bids(bid),
    wins(win),
    rng(random()) {
}


MockExchange::Worker::
Worker(MockExchange * exchange, Json::Value bid, Json::Value win) :
    exchange(exchange),
    bids(BidSource::createBidSource(std::move(bid))),
    wins(WinSource::createWinSource(std::move(win))),
    rng(random()) {
}


void
MockExchange::Worker::
run() {
    while(bid());
}

bool
MockExchange::Worker::bid() {
    for (;;) {
        auto br = bids->sendBidRequest();
        exchange->recordHit("requests");

        auto response = bids->receiveBid();
        exchange->recordHit("responses");

        if (!response.first) continue;
        vector<ExchangeSource::Bid> items = response.second;

        for (auto & bid : items) {
            if(bid.maxPrice == 0) continue;
            exchange->recordHit("responses");

            if (!wins) break;

            auto ret = isWin(br, bid);
            if (!ret.first) continue;

            wins->sendWin(br, bid, ret.second);
            exchange->recordHit("wins");

            wins->sendImpression(br, bid);
            exchange->recordHit("impressions");

            if (!isClick(br, bid)) continue;
            wins->sendClick(br, bid);
            exchange->recordHit("clicks");
        }

        break;
    }

    return !bids->isDone();
}


pair<bool, Amount>
MockExchange::Worker::isWin(const BidRequest&, const ExchangeSource::Bid& bid) {
    if (rng.random01() >= 0.1)
        return make_pair(false, Amount());

    return make_pair(true, MicroUSD_CPM(bid.maxPrice * 0.6 + bid.maxPrice * rng.random01() * 0.4));
}


bool
MockExchange::Worker::isClick(const BidRequest&, const ExchangeSource::Bid&) {
    return rng.random01() <= 0.1;
}


} // namepsace RTBKIT
