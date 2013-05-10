/** mock_exchange.h                                 -*- C++ -*-
    Rémi Attab, 18 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Mock exchanges used for testing various types of bid request behaviours.

    \todo This a work in progress that needs to be generic-ified.

*/

#ifndef __rtbkit__mock_exchange_h__
#define __rtbkit__mock_exchange_h__

#include "rtbkit/common/testing/exchange_source.h"
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

    void start(size_t threadCount, size_t numBidRequests, std::vector<int> const & bidPorts, std::vector<int> const & winPorts);

    bool isDone() const {
        return !running;
    }

private:
    int running;

    struct Worker {
        Worker(MockExchange * exchange, size_t id, int bidPort, int winPort);

        void run();
        void run(size_t requests);
        void bid();

        std::pair<bool, Amount> isWin(const BidRequest&, const ExchangeSource::Bid& bid);

        MockExchange * exchange;
        BidSource bids;
        WinSource wins;
        ML::RNG rng;
    };

    std::vector<Worker> workers;
    boost::thread_group threads;
};


} // namespace RTBKIT

#endif // __rtbkit__mock_exchange_h__
