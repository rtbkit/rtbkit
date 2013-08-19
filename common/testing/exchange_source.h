/** exchange_source.h                                 -*- C++ -*-
    Eric Robert, 6 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Simple stream abstraction to simulate an exchange

*/

#pragma once

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
        if(k == std::string::npos)
            throw ML::Exception("url parsing failed for '" + url + "' and should be (host:port)");
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
    BidSource(NetworkAddress address);
    BidSource(NetworkAddress address, int lifetime);
    BidSource(Json::Value const & json);

    bool isDone() const;

    void write(std::string const & text);
    std::string read();

    BidRequest sendBidRequest();
    std::pair<bool, std::vector<Bid>> receiveBid();

    virtual std::pair<bool, std::vector<Bid>> parseResponse(const std::string& rawResponse) {
        return std::make_pair(false, std::vector<Bid>());
    }

    virtual BidRequest generateRandomBidRequest() {
        return BidRequest();
    }

    typedef std::function<BidSource * (Json::Value const &)> Factory;
    static void registerBidSourceFactory(std::string const & name, Factory callback);
    static std::unique_ptr<BidSource> createBidSource(Json::Value const & json);

    bool bidForever;
    long bidCount;
    long bidLifetime;

    unsigned long long key;
};

struct WinSource : public ExchangeSource {
    WinSource(NetworkAddress address);
    WinSource(Json::Value const & json);

    void write(const std::string & data);

    virtual void sendWin(const BidRequest& br,
                         const Bid& bid,
                         const Amount& winPrice);
    virtual void sendImpression(const BidRequest& br, const Bid& bid);
    virtual void sendClick(const BidRequest& br, const Bid& bid);

    typedef std::function<WinSource * (Json::Value const &)> Factory;
    static void registerWinSourceFactory(std::string const & name, Factory callback);
    static std::unique_ptr<WinSource> createWinSource(Json::Value const & json);
};

} // namespace RTBKIT
