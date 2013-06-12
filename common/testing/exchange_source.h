/** exchange_source.h                                 -*- C++ -*-
    Eric Robert, 6 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Simple stream abstraction to simulate an exchange

*/

#include "jml/utils/rng.h"
#include "common/account_key.h"
#include "common/currency.h"
#include "common/bid_request.h"

#include <netdb.h>

namespace RTBKIT {

struct PostAuctionEvent;

struct ExchangeSource {
    ExchangeSource(int port);
    ~ExchangeSource();

    void connect();

    struct Bid
    {
        Datacratic::Id adSpotId;
        int maxPrice;
        AccountKey account;
        Datacratic::Date bidTimestamp;
    };

    addrinfo * addr;
    int fd;

    ML::RNG rng;
};

struct BidSource : public ExchangeSource {
    BidSource(int port = 0, int id = 0) :
        ExchangeSource(port), id(id * port), key(0) {
    }

    void write(std::string const & text);
    std::string read();

    void sendBidRequest(const BidRequest& request);
    std::pair<bool, std::vector<Bid>> parseResponse(const std::string& rawResponse);
    std::pair<bool, std::vector<Bid>> recvBid();

    BidRequest makeBidRequest();

    virtual void generateRandomBidRequest() {
    }

    long long id;
    long long key;
};

struct WinSource : public ExchangeSource {
    WinSource(int port) :
        ExchangeSource(port) {
    }

    void sendWin(
            const BidRequest& bidRequest,
            const Bid& bid,
            const Amount& winPrice);
    void sendImpression(const BidRequest& bidRequest, const Bid& bid);
    void sendClick(const BidRequest& bidRequest, const Bid& bid);

private:
    void sendEvent(const PostAuctionEvent& ev);
};

} // namespace RTBKIT
