/** mock_exchange.cc                                 -*- C++ -*-
    Rémi Attab, 18 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Implementation of the mock exchange.

*/

#include "mock_exchange.h"

#include "rtbkit/core/post_auction/post_auction_loop.h"
#include "soa/service/http_header.h"

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
~MockExchange()
{
    threads.join_all();
}


void
MockExchange::
start(size_t threadCount, size_t numBidRequests, std::vector<int> const & bidPorts, std::vector<int> const & winPorts)
{
    try {
        running = threadCount;

        auto startWorker = [=](size_t i, int bidPort, int winPort) {
            Worker worker(this, i, bidPort, winPort);
            if(numBidRequests) {
                worker.run(numBidRequests);
            }
            else {
                worker.run();
            }

            ML::atomic_dec(running);
        };

        int bp = 0;
        int wp = 0;

        for(size_t i = 0; i != threadCount; ++i) {
            int bidPort = bidPorts[bp++ % bidPorts.size()];
            int winPort = winPorts[wp++ % winPorts.size()];
            threads.create_thread(std::bind(startWorker, i, bidPort, winPort));
        }
    }
    catch (const exception& ex) {
        cerr << "got exception on request: " << ex.what() << endl;
    }
}


MockExchange::Stream::
Stream(int port) : addr(0), fd(-1)
{
    addrinfo hint = { 0, AF_INET, SOCK_STREAM, 0, 0, 0, 0, 0 };

    int res = getaddrinfo(0, to_string(port).c_str(), &hint, &addr);
    ExcCheckErrno(!res, "getaddrinfo failed");

    if(!addr) {
        throw ML::Exception("cannot find suitable address");
    }

    std::cerr << "publishing on port " << port << std::endl;
    connect();
}


MockExchange::Stream::
~Stream()
{
    if (addr) freeaddrinfo(addr);
    if (fd >= 0) close(fd);
}


void
MockExchange::Stream::
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

        //ML::sleep(0.1);
    }
}


void
MockExchange::BidStream::
sendBidRequest(const BidRequest& originalBidRequest)
{
    string strBidRequest = originalBidRequest.toJsonStr();
    string httpRequest = ML::format(
            "POST /bids HTTP/1.1\r\n"
            "Content-Length: %zd\r\n"
            "Content-Type: application/json\r\n"
            "Connection: Keep-Alive\r\n"
            "\r\n"
            "%s",
            strBidRequest.size(),
            strBidRequest.c_str());


    const char * current = httpRequest.c_str();
    const char * end = current + httpRequest.size();

    while (current != end) {
        int res = send(fd, current, end - current, MSG_NOSIGNAL);
        if(res == 0 || res == -1) {
            connect();
            current = httpRequest.c_str();
            continue;
        }

        current += res;
    }

    ExcAssertEqual((void *)current, (void *)end);
}


auto
MockExchange::BidStream::
parseResponse(const string& rawResponse) -> pair<bool, vector<Bid>>
{
    Json::Value payload;

    try {
        HttpHeader header;
        header.parse(rawResponse);
        payload = Json::parse(header.knownData);
    }
    catch (const exception & exc) {
        cerr << "invalid response received: " << exc.what() << endl;
        return make_pair(false, vector<Bid>());
    }

    if (payload.isMember("error")) {
        cerr << "error returned: "
            << payload["error"] << endl
            << payload["details"] << endl;
        return make_pair(false, vector<Bid>());
    }

    ExcAssert(payload.isMember("imp"));

    vector<Bid> bids;

    for (size_t i = 0; i < payload["imp"].size(); ++i) {
        auto& spot = payload["imp"][i];

        Bid bid;

        bid.adSpotId = Id(spot["id"].asString());
        bid.maxPrice = spot["max_price"].asInt();
        bid.account = AccountKey(spot["account"].asString(), '.');
        bids.push_back(bid);
    }

    return make_pair(true, bids);
}


auto
MockExchange::BidStream::
recvBid() -> pair<bool, vector<Bid>>
{
    array<char, 16384> buffer;

    int res = recv(fd, buffer.data(), buffer.size(), 0);
    if (res == 0 || (res == -1 && errno == ECONNRESET)) {
        return make_pair(false, vector<Bid>());
    }

    ExcCheckErrno(res != -1, "recv");

    close(fd);
    fd = -1;

    return parseResponse(string(buffer.data(), res));
}


auto
MockExchange::BidStream::
makeBidRequest() -> BidRequest
{
    BidRequest bidRequest;

    FormatSet formats;
    formats.push_back(Format(160,600));
    AdSpot spot;
    spot.id = Id(1);
    spot.formats = formats;
    bidRequest.imp.push_back(spot);

    spot.formats[0] = Format(300,250);
    spot.id = Id(2);
    bidRequest.imp.push_back(spot);

    bidRequest.location.countryCode = "CA";
    bidRequest.location.regionCode = "QC";
    bidRequest.location.cityName = "Montreal";
    bidRequest.auctionId = Id(id * 10000000 + key);
    bidRequest.exchange = "test";
    bidRequest.language = "en";
    bidRequest.url = Url("http://datacratic.com");
    bidRequest.timestamp = Date::now();
    bidRequest.userIds.add(Id(std::string("foo")), ID_EXCHANGE);
    bidRequest.userIds.add(Id(std::string("bar")), ID_PROVIDER);
    ++key;

    return bidRequest;
}


void
MockExchange::WinStream::
sendWin(const BidRequest& bidRequest, const Bid& bid, const Amount& winPrice)
{
    PostAuctionEvent event;
    event.type = PAE_WIN;
    event.auctionId = bidRequest.auctionId;
    event.adSpotId = bid.adSpotId;
    event.timestamp = Date::now();
    event.winPrice = winPrice;
    event.uids = bidRequest.userIds;
    event.account = bid.account;
    event.bidTimestamp = bid.bidTimestamp;

    string str = event.toJson().toString();
    string httpRequest = ML::format(
            "POST /win HTTP/1.1\r\n"
            "Content-Length: %zd\r\n"
            "Content-Type: application/json\r\n"
            "Connection: Keep-Alive\r\n"
            "\r\n"
            "%s",
            str.size(),
            str.c_str());

    const char * current = httpRequest.c_str();
    const char * end = current + httpRequest.size();

    while (current != end) {
        int res = send(fd, current, end - current, MSG_NOSIGNAL);
        if(res == 0 || res == -1) {
            connect();
            current = httpRequest.c_str();
            continue;
        }

        current += res;
    }

    close(fd);
    fd = -1;

    std::cerr << "win sent payload=" << str << std::endl;

    ExcAssertEqual((void *)current, (void *)end);
}


MockExchange::Worker::
Worker(MockExchange * exchange, size_t id, int bidPort, int winPort) : exchange(exchange), bids(bidPort, id), wins(winPort), rng(random()) {
}


void
MockExchange::Worker::
run() {
    for(;;) {
        bid();
    }
}


void
MockExchange::Worker::
run(size_t requests) {
    for(size_t i = 0; i != requests; ++i) {
        bid();
    }
}


void
MockExchange::Worker::bid() {
    BidRequest bidRequest = bids.makeBidRequest();
    exchange->recordHit("requests");

    for (;;) {
        bids.sendBidRequest(bidRequest);
        exchange->recordHit("sent");

        auto response = bids.recvBid();
        if (!response.first) continue;
        exchange->recordHit("bids");

        vector<Bid> bids = response.second;

        for (const Bid& bid : bids) {
            auto ret = isWin(bidRequest, bid);
            if (!ret.first) continue;
            wins.sendWin(bidRequest, bid, ret.second);
            exchange->recordHit("wins");
        }

        break;
    }
}


pair<bool, Amount>
MockExchange::Worker::isWin(const BidRequest&, const Bid& bid)
{
    if (rng.random01() >= 0.1)
        return make_pair(false, Amount());

    return make_pair(true, MicroUSD_CPM(bid.maxPrice * rng.random01()));
}


} // namepsace RTBKIT
