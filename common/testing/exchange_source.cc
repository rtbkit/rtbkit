/** exchange_source.cc                                 -*- C++ -*-
    Eric Robert, 6 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Implementation of exchange sources

*/

#include "exchange_source.h"
#include "rtbkit/core/post_auction/post_auction_loop.h"
#include "soa/service/http_header.h"

#include <dlfcn.h>
#include <array>

using namespace RTBKIT;

ExchangeSource::
ExchangeSource(NetworkAddress address_) :
    address(std::move(address_)),
    addr(0),
    fd(-1)
{
    static int seed;
    ML::atomic_inc(seed);
    rng.seed(seed);

    addrinfo hint = { 0, AF_INET, SOCK_STREAM, 0, 0, 0, 0, 0 };

    char const * host = 0;
    if(address.host != "localhost") host = address.host.c_str();
    int res = getaddrinfo(host, std::to_string(address.port).c_str(), &hint, &addr);
    ExcCheckErrno(!res, "getaddrinfo failed");
    if(!addr) throw ML::Exception("cannot find suitable address");

    connect();
    std::cerr << "sending to " << address.host << ":" << address.port << std::endl;
}


ExchangeSource::
~ExchangeSource()
{
    if (addr) freeaddrinfo(addr);
    if (fd >= 0) close(fd);
}


void
ExchangeSource::
connect()
{
    if(fd >= 0) close(fd);
    fd = socket(AF_INET, SOCK_STREAM, 0);
    ExcCheckErrno(fd != -1, "socket failed");

    for(;;) {
        int res = ::connect(fd, addr->ai_addr, addr->ai_addrlen);
        if(res == 0) {
            break;
        }

        ML::sleep(0.1);
    }
}


BidSource::BidSource(NetworkAddress address) :
    ExchangeSource(std::move(address)),
    bidForever(true),
    bidCount(0),
    bidLifetime(0) {
}


BidSource::BidSource(NetworkAddress address, int lifetime) :
    ExchangeSource(std::move(address)),
    bidForever(false),
    bidCount(0),
    bidLifetime(lifetime) {
}


BidSource::
BidSource(Json::Value const & json) :
    ExchangeSource(json["url"].asString()),
    bidForever(true),
    bidCount(0),
    bidLifetime(0) {
    if(json.isMember("lifetime")) {
        bidForever = false;
        bidLifetime = json["lifetime"].asInt();
    }
}


bool
BidSource::
isDone() const {
    return bidForever ? false : bidLifetime <= bidCount;
}


void
BidSource::
write(std::string const & request) {
    const char * current = request.c_str();
    const char * end = current + request.size();

    while (current != end) {
        int res = send(fd, current, end - current, MSG_NOSIGNAL);
        if(res == 0 || res == -1) {
            connect();
            current = request.c_str();
            continue;
        }

        current += res;
    }

    ExcAssertEqual((void *)current, (void *)end);
}


std::string
BidSource::
read()
{
    std::array<char, 16384> buffer;

    int res = recv(fd, buffer.data(), buffer.size(), 0);
    if (res == 0 || (res == -1 && errno == ECONNRESET)) {
        return "";
    }

    ExcCheckErrno(res != -1, "recv");
    return std::string(buffer.data(), res);
}


BidRequest
BidSource::
sendBidRequest() {
    ++bidCount;
    return generateRandomBidRequest();
}


std::pair<bool, std::vector<ExchangeSource::Bid>>
BidSource::
receiveBid() {
    return parseResponse(read());
}


WinSource::
WinSource(NetworkAddress address) :
    ExchangeSource(std::move(address)) {
}


WinSource::
WinSource(Json::Value const & json) :
    ExchangeSource(json["url"].asString()) {
}

void
WinSource::
sendWin(const BidRequest& bidRequest, const Bid& bid, const Amount& winPrice)
{
}


void
WinSource::
sendImpression(const BidRequest& bidRequest, const Bid& bid)
{
}


void
WinSource::
sendClick(const BidRequest& bidRequest, const Bid& bid)
{
}


void
WinSource::
write(const std::string & data)
{
    const char * current = data.c_str();
    const char * end = current + data.size();

    while (current != end) {
        int res = send(fd, current, end - current, MSG_NOSIGNAL);
        if(res == 0 || res == -1) {
            connect();
            current = data.c_str();
            continue;
        }

        current += res;
    }

    ExcAssertEqual((void *)current, (void *)end);
}


namespace {
    typedef std::lock_guard<ML::Spinlock> Guard;
    static ML::Spinlock bidLock;
    static ML::Spinlock winLock;
    static std::unordered_map<std::string, BidSource::Factory> bidFactories;
    static std::unordered_map<std::string, WinSource::Factory> winFactories;
}


BidSource::Factory getBidFactory(std::string const & name) {
    // see if it's already existing
    {
        Guard guard(bidLock);
        auto i = bidFactories.find(name);
        if (i != bidFactories.end()) return i->second;
    }

    // else, try to load the exchange library
    std::string path = "lib" + name + "_bid_request.so";
    void * handle = dlopen(path.c_str(), RTLD_NOW);
    if (!handle) {
        throw ML::Exception("couldn't find bid request/source library " + path);
    }

    // if it went well, it should be registered now
    Guard guard(bidLock);
    auto i = bidFactories.find(name);
    if (i != bidFactories.end()) return i->second;

    throw ML::Exception("couldn't find bid source name " + name);
}


void BidSource::registerBidSourceFactory(std::string const & name, Factory callback) {
    Guard guard(bidLock);
    if (!bidFactories.insert(std::make_pair(name, callback)).second)
        throw ML::Exception("already had a bid source factory registered");
}


std::unique_ptr<BidSource> BidSource::createBidSource(Json::Value const & json) {
    auto name = json.get("type", "unknown").asString();
    auto factory = getBidFactory(name);
    return std::unique_ptr<BidSource>(factory(json));
}


WinSource::Factory getWinFactory(std::string const & name) {
    // see if it's already existing
    {
        Guard guard(winLock);
        auto i = winFactories.find(name);
        if (i != winFactories.end()) return i->second;
    }

    // else, try to load the adserver library
    std::string path = "lib" + name + "_adserver.so";
    void * handle = dlopen(path.c_str(), RTLD_NOW);
    if (!handle) {
        throw ML::Exception("couldn't find adserver library " + path);
    }

    // if it went well, it should be registered now
    Guard guard(winLock);
    auto i = winFactories.find(name);
    if (i != winFactories.end()) return i->second;

    throw ML::Exception("couldn't find win source name " + name);
}


void WinSource::registerWinSourceFactory(std::string const & name, Factory callback) {
    Guard guard(winLock);
    if (!winFactories.insert(std::make_pair(name, callback)).second)
        throw ML::Exception("already had a win source factory registered");
}


std::unique_ptr<WinSource> WinSource::createWinSource(Json::Value const & json) {
    auto name = json.get("type", "unknown").asString();
    if(name == "none") {
        return 0;
    }

    auto factory = getWinFactory(name);
    return std::unique_ptr<WinSource>(factory(json));
}

