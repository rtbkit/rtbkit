/** exchange_source.h                                 -*- C++ -*-
    Eric Robert, 6 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Simple stream abstraction to simulate an exchange

*/

#include "jml/utils/rng.h"
#include "jml/arch/info.h"
#include "common/account_key.h"
#include "common/currency.h"
#include "common/bid_request.h"

#include <netdb.h>

namespace RTBKIT {

struct PostAuctionEvent;

struct NetworkAddress {
    NetworkAddress(int port, std::string host = "localhost") :
        host(std::move(host)),
        port(port) {
    }

    NetworkAddress(std::string url) {
        auto k = url.find_first_of(':');
        if(k == std::string::npos) throw ML::Exception("url parsing failed for '" + url + "' and should be (host:port)");
        host = url.substr(0, k);
        port = std::stoi(url.substr(k + 1));
    }

    std::string host;
    int port;
};

struct ExchangeSource {
    ExchangeSource(NetworkAddress address);
    ~ExchangeSource();

    void setup();
    void connect();

    struct Bid
    {
        Datacratic::Id adSpotId;
        int maxPrice;
        AccountKey account;
        Datacratic::Date bidTimestamp;
    };

    NetworkAddress address;
    addrinfo * addr;
    int fd;
    ML::RNG rng;
};

struct BidSource : public ExchangeSource {
    BidSource(NetworkAddress address) : ExchangeSource(std::move(address)) {
        key = rng.random();
    }

    void write(std::string const & text);
    std::string read();

    void sendBidRequest(const BidRequest& request);
    std::pair<bool, std::vector<Bid>> parseResponse(const std::string& rawResponse);
    std::pair<bool, std::vector<Bid>> recvBid();

    BidRequest makeBidRequest();

    virtual void generateRandomBidRequest() {
    }

    long long key;
};

struct WinSource : public ExchangeSource {
    WinSource(NetworkAddress address) : ExchangeSource(std::move(address)) {
    }

    void sendWin(const BidRequest& bidRequest,
                 const Bid& bid,
                 const Amount& winPrice);
    void sendImpression(const BidRequest& bidRequest, const Bid& bid);
    void sendClick(const BidRequest& bidRequest, const Bid& bid);

private:
    void sendEvent(const PostAuctionEvent& ev);
};

} // namespace RTBKIT
