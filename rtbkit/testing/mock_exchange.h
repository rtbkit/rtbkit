/** mock_exchange.h                                 -*- C++ -*-
    Rémi Attab, 18 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Mock exchanges used for testing various types of bid request behaviours.

    \todo This a work in progress that needs to be generic-ified.

*/

#ifndef __rtbkit__mock_exchange_h__
#define __rtbkit__mock_exchange_h__

#include "rtbkit/common/testing/exchange_source.h"
#include "rtbkit/common/bids.h"
#include "soa/service/service_utils.h"
#include "soa/service/service_base.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/types/id.h"
#include "soa/types/date.h"
#include "jml/utils/rng.h"

#include <netdb.h>
#include <vector>

namespace RTBKIT {

/******************************************************************************/
/* MOCK EXCHANGE                                                              */
/******************************************************************************/

struct MockExchange : public Datacratic::ServiceBase
{
    MockExchange(Datacratic::ServiceProxyArguments & args,
                 const std::string& name = "mock-exchange");
    MockExchange(const std::shared_ptr<Datacratic::ServiceProxies> services,
                 const std::string& name = "mock-exchange");

    ~MockExchange();

    void start(Json::Value const & configuration);

    bool isDone() const {
        return !running;
    }

    void add(BidSource * bids, WinSource * wins, EventSource * events);

private:
    int running;

    struct Worker {
        Worker(MockExchange * exchange, BidSource *bid, WinSource *win, EventSource *event);
        Worker(MockExchange * exchange, Json::Value bid, Json::Value win, Json::Value event);

        struct Win {
            Date timestamp;
            BidRequest br;
            ExchangeSource::Bid bid;
            Amount winPrice;
        };

        struct Event {
            Date timestamp;

            enum Type { Impression, Click } type;
            BidRequest br;
            ExchangeSource::Bid bid;
        };

        void run();
        bool bid();

        std::pair<bool, Amount>
        isWin(const BidRequest&, const ExchangeSource::Bid& bid);
        bool isClick(const BidRequest&, const ExchangeSource::Bid& bid);

        void processWinsQueue();
        void processEventsQueue();

        MockExchange * exchange;
        std::unique_ptr<BidSource> bids;
        std::unique_ptr<WinSource> wins;
        std::unique_ptr<EventSource> events;
        ML::RNG rng;

        int winsDelay;
        int eventsDelay;

        std::deque<Win> winsQueue;
        std::deque<Event> eventsQueue;

    };

    std::vector<Worker> workers;
    boost::thread_group threads;
};


} // namespace RTBKIT

#endif // __rtbkit__mock_exchange_h__
