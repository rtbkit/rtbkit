/** mock_exchange.cc                                 -*- C++ -*-
    Rémi Attab, 18 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Implementation of the mock exchange.

*/

#include "mock_exchange.h"

#include "soa/service/http_header.h"
#include "jml/utils/smart_ptr_utils.h"

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

            threads.create_thread([=]() {
                Worker worker(this, json["bids"], json["wins"], json["events"]);
                worker.run();

                ML::atomic_dec(running);
            });
        }
    }
}


void
MockExchange::
add(BidSource * bid, WinSource * win, EventSource * event) {
    std::cerr << "starting worker " << running << std::endl;
    ML::atomic_inc(running);

    threads.create_thread([=]() {
        Worker worker(this, bid, win, event);
        worker.run();

        ML::atomic_dec(running);
    });
}


MockExchange::Worker::
Worker(MockExchange * exchange, BidSource *bid, WinSource *win, EventSource *event) :
    exchange(exchange),
    bids(bid),
    wins(win),
    events(event),
    rng(random()) {
}


MockExchange::Worker::
Worker(MockExchange * exchange, Json::Value bid, Json::Value win, Json::Value event) :
    exchange(exchange),
    bids(BidSource::createBidSource(std::move(bid))),
    wins(WinSource::createWinSource(std::move(win))),
    events(EventSource::createEventSource(std::move(event))),
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

        vector<ExchangeSource::Bid> items = response.second;

        if (!response.first || items.empty()) {
            exchange->recordHit("noBids");
            break;
        }

        for (auto & bid : items) {
            if(bid.maxPrice == 0) continue;
            exchange->recordHit("bids");

            if (!wins) break;

            auto ret = isWin(br, bid);
            if (!ret.first) continue;
            ML::sleep(0.5);

			bid.bidTimestamp = Date::now();
            wins->sendWin(br, bid, ret.second);
            exchange->recordHit("wins");

            if (!isClick(br, bid)) continue;
            events->sendClick(br, bid);
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
