/** exchange_source.h                                 -*- C++ -*-
    Eric Robert, 6 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Simple stream abstraction to simulate an exchange

*/

#pragma once

#include "jml/utils/rng.h"
#include "jml/arch/info.h"
#include "rtbkit/common/account_key.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/bid_request.h"
#include "rtbkit/common/plugin_interface.h"

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

    std::string read();
    void write(std::string const & text);

    struct Bid
    {
        Datacratic::Id adSpotId;
        Amount maxPrice;
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

    BidRequest sendBidRequest();
    std::pair<bool, std::vector<Bid>> receiveBid();

    virtual std::pair<bool, std::vector<Bid>> parseResponse(const std::string& rawResponse) {
        return std::make_pair(false, std::vector<Bid>());
    }

    virtual BidRequest generateRandomBidRequest() {
        return BidRequest();
    }

    typedef std::function<BidSource * (Json::Value const &)> Factory;

    // FIXME: this is being kept just for compatibility reasons.
    // we don't want to break compatibility now, although this interface does not make
    // sense any longer  
    // so any use of it should be considered deprecated
    static void registerBidSourceFactory(std::string const & name, Factory callback)
    {
      PluginInterface<BidSource>::registerPlugin(name, callback);
    }

    /** plugin interface needs to be able to request the root name of the plugin library */
    static const std::string libNameSufix() {return "bid_request";};

  
    static std::unique_ptr<BidSource> createBidSource(Json::Value const & json);

    bool bidForever;
    long bidCount;
    long bidLifetime;

    unsigned long long key;
};

struct WinSource : public ExchangeSource {
    WinSource(NetworkAddress address);
    WinSource(Json::Value const & json);

    virtual void sendWin(const BidRequest& br,
                         const Bid& bid,
                         const Amount& winPrice);

    typedef std::function<WinSource * (Json::Value const &)> Factory;
  
    // FIXME: this is being kept just for backwards compatibility reasons.
    // we don't want to break compatibility now, although this interface does not make
    // sense any longer
    // so any use of it should be considered deprecated
    static void registerWinSourceFactory(std::string const & name, Factory callback)
    {
      PluginInterface<WinSource>::registerPlugin(name, callback);
    }
  
    /** plugin interface needs to be able to request the root name of the plugin library */
    static const std::string libNameSufix() {return "adserver";};  
  
    static std::unique_ptr<WinSource> createWinSource(Json::Value const & json);
};

struct EventSource : public ExchangeSource {
    EventSource(NetworkAddress address);
    EventSource(Json::Value const & json);

    virtual void sendImpression(const BidRequest& br, const Bid& bid);
    virtual void sendClick(const BidRequest& br, const Bid& bid);

    typedef std::function<EventSource * (Json::Value const &)> Factory;
  
    // FIXME: this is being kept just for compatibility reasons.
    // we don't want to break compatibility now, although this interface does not make
    // sense any longer
    // so any use of it should be considered deprecated 
    static void registerEventSourceFactory(std::string const & name, Factory callback)
    {
      PluginInterface<EventSource>::registerPlugin(name, callback);
    }


  
    /** plugin interface needs to be able to request the root name of the plugin library */
    static const std::string libNameSufix() {return "adserver";};

    static std::unique_ptr<EventSource> createEventSource(Json::Value const & json);
};

} // namespace RTBKIT
