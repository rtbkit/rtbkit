/** mock_exchange.h                                 -*- C++ -*-
    Rémi Attab, 18 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Mock exchanges used for testing various types of bid request behaviours.

    \todo This a work in progress that needs to be generic-ified.

*/

#ifndef __rtbkit__mock_exchange_h__
#define __rtbkit__mock_exchange_h__

#include "common/account_key.h"
#include "common/currency.h"
#include "common/bid_request.h"
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
    MockExchange(
            const std::shared_ptr<Datacratic::ServiceProxies> services
                = std::make_shared<Datacratic::ServiceProxies>(),
            const std::string& name = "mock-exchange");

    ~MockExchange();

    void init(size_t exchangeId, const std::vector<int>& ports);
    void start(size_t numBidRequests);

protected:

    struct Bid
    {
        Datacratic::Id adSpotId;
        int maxPrice;
        std::string tagId;

        AccountKey account;
        Datacratic::Date bidTimestamp;
    };

    virtual BidRequest makeBidRequest(size_t i);

    virtual std::pair<bool, Amount>
    isWin(const BidRequest&, const Bid& bid);

private:

    void connect();

    void sendBidRequest(const BidRequest& request);

    std::pair<bool, std::vector<Bid> >
    parseResponse(const std::string& rawResponse);

    std::pair<bool, std::vector<Bid> > recvBid();


    /** In reality, the exchange wouldn't send anything directly to the
        PAL. That'd be the job of the connector.
     */
    void sendWin(
            const BidRequest& bidRequest,
            const Bid& bid,
            const Amount& winPrice);


    ML::RNG rng;

    ZmqNamedProxy toPostAuctionService;
    addrinfo* toRouterAddr;
    int toRouterFd;

    size_t exchangeId;
};


} // namespace RTBKIT

#endif // __rtbkit__mock_exchange_h__
