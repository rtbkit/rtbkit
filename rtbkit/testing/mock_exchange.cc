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

    winsDelay = win.get("delay", 0).asInt();
    eventsDelay = event.get("delay", 0).asInt();

    lastSentWins = lastSentEvents = Date::now();
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

            const Date now = Date::now();
            bid.bidTimestamp = now;
            if (winsDelay == 0) {
                wins->sendWin(br, bid, ret.second);
                exchange->recordHit("wins");

                if (!isClick(br, bid)) continue;

                if (eventsDelay == 0) {
                    events->sendClick(br, bid);
                    exchange->recordHit("clicks");
                }
                else {
                    eventsQueue.push_back(Event { Event::Click, br, bid });
                    if (now.secondsSince(lastSentEvents) >= eventsDelay) {
                        processEventsQueue();
                    }
                }
            }
            else {
                winsQueue.push_back(Win { br, bid, ret.second });
                if (now.secondsSince(lastSentWins) >= winsDelay) {
                    processWinsQueue();
                }
            }
        }

        break;
    }

    return !bids->isDone();
}

void
MockExchange::Worker::processWinsQueue() {
    const Date now = Date::now();
    while (!winsQueue.empty()) {
        const Win& win = winsQueue.front();
        wins->sendWin(win.br, win.bid, win.winPrice);
        exchange->recordHit("wins");

        if (isClick(win.br, win.bid)) {
            if (eventsDelay == 0) {
                events->sendClick(win.br, win.bid);
            }
            else {
                eventsQueue.push_back(Event { Event::Click, win.br, win.bid });
                if (now.secondsSince(lastSentEvents) >= eventsDelay) {
                    processEventsQueue();
                }
            }
        }

        winsQueue.pop_front();
    }
    lastSentWins = Date::now();
}

void
MockExchange::Worker::processEventsQueue() {
    while (!eventsQueue.empty()) {
        const Event& event = eventsQueue.front();
        events->sendClick(event.br, event.bid);
        exchange->recordHit("clicks");

        eventsQueue.pop_front();
    }

    lastSentEvents = Date::now();
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
