/** mock_exchange.cc                                 -*- C++ -*-
    Rémi Attab, 18 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Implementation of the mock exchange.

*/

#include "mock_exchange.h"

#include "core/post_auction/post_auction_loop.h"
#include "soa/service/http_header.h"

#include <array>

using namespace std;
using namespace ML;

namespace RTBKIT {


/******************************************************************************/
/* MOCK EXCHANGE                                                              */
/******************************************************************************/

MockExchange::
MockExchange(const shared_ptr<ServiceProxies> proxies, const string& name) :
    ServiceBase(name, proxies),
    rng(random()),
    toPostAuctionService(getZmqContext()),
    toRouterAddr(nullptr),
    toRouterFd(-1)
{}


MockExchange::
~MockExchange()
{
    if (toRouterAddr) freeaddrinfo(toRouterAddr);
    if (toRouterFd >= 0) close(toRouterFd);
}


void
MockExchange::
init(size_t exchangeId, const vector<int>& ports)
{
    this->exchangeId = exchangeId;

    toPostAuctionService.init(getServices()->config, ZMQ_XREQ);
    toPostAuctionService.connectToServiceClass("rtbPostAuctionService", "events");

    string addr = "127.0.0.1";
    addrinfo hints = { 0, AF_INET, SOCK_STREAM, 0, 0, 0, 0, 0 };
    int port = ports[random() % ports.size()];

    // Now we have it as a sockaddr_t. Convert it back to a numeric address
    int res = getaddrinfo(
            addr.c_str(), to_string(port).c_str(), &hints, &toRouterAddr);

    ExcCheckErrno(!res, ML::format("addrToIp(%s)", addr.c_str()));
    ExcCheck(toRouterAddr, "no addresses");

    connect();
}


void
MockExchange::
start(size_t numBidRequests)
{
    try {
        for (size_t i = 0; i < numBidRequests; ++i) {

            BidRequest bidRequest = makeBidRequest(i);
            recordHit("requests");

            while (true) {
                sendBidRequest(bidRequest);
                recordHit("sent");

                auto response = recvBid();
                if (!response.first) continue;
                recordHit("bids");

                vector<Bid> bids = response.second;

                for (const Bid& bid : bids) {

                    auto ret = isWin(bidRequest, bid);
                    if (!ret.first) continue;
                    sendWin(bidRequest, bid, ret.second);
                    recordHit("wins");

                    // \todo simulate the other PAL events.
                }
                break;
            }

        }
    }
    catch (const exception& ex) {
        cerr << "got exception on request: " << ex.what() << endl;
    }
}


BidRequest
MockExchange::
makeBidRequest(size_t i)
{
    BidRequest bidRequest;

    FormatSet formats;
    formats.push_back(Format(160,600));
    AdSpot spot;
    spot.id = Id(1);
    spot.formats = formats;
    bidRequest.spots.push_back(spot);

    formats[0] = Format(300,250);
    spot.id = Id(2);
    bidRequest.spots.push_back(spot);

    bidRequest.location.countryCode = "CA";
    bidRequest.location.regionCode = "QC";
    bidRequest.location.cityName = "Montreal";
    bidRequest.auctionId = Id(exchangeId * 10000000 + i);
    bidRequest.exchange = "test";
    bidRequest.language = "en";
    bidRequest.url = Url("http://rtbkit.com");
    bidRequest.timestamp = Date::now();
    bidRequest.userIds.add(Id(std::string("foo")), ID_EXCHANGE);
    bidRequest.userIds.add(Id(std::string("bar")), ID_PROVIDER);

    return bidRequest;
}


pair<bool, Amount>
MockExchange::
isWin(const BidRequest&, const Bid& bid)
{
    if (rng.random01() >= 0.1)
        return make_pair(false, Amount());

    return make_pair(true, MicroUSD(bid.maxPrice * rng.random01()));
}


void
MockExchange::
connect()
{
    if (toRouterFd != -1)
        close(toRouterFd);

    toRouterFd = socket(AF_INET, SOCK_STREAM, 0);
    ExcCheckErrno(toRouterFd != -1, "couldn't get socket");

    int res = ::connect(
            toRouterFd, toRouterAddr->ai_addr, toRouterAddr->ai_addrlen);
    ExcCheckErrno(!res, "couldn't connect to router");
}


void
MockExchange::
sendBidRequest(const BidRequest& originalBidRequest)
{
    string strBidRequest = originalBidRequest.toJsonStr();
    string httpRequest = ML::format(
            "POST /bidreq HTTP/1.1\r\n"
            "Content-Length: %zd\r\n"
            "Content-Type: application/json\r\n"
            "\r\n"
            "%s",
            strBidRequest.size(),
            strBidRequest.c_str());


    const char * current = httpRequest.c_str();
    const char * end = current + httpRequest.size();

    while (current != end) {
        int res = send(toRouterFd, current, end - current, MSG_NOSIGNAL);
        ExcCheckErrno(res != -1, "send()");
        current += res;
    }

    ExcAssertEqual((void *)current, (void *)end);
}


auto
MockExchange::
parseResponse(const string& rawResponse) -> pair<bool, vector<Bid> >
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

    ExcAssert(payload.isMember("spots"));


    vector<Bid> bids;

    for (size_t i = 0; i < payload["spots"].size(); ++i) {
        auto& spot = payload["spots"][i];

        Bid bid;

        bid.adSpotId = Id(spot["id"].asString());
        bid.maxPrice = spot["max_price"].asInt();
        bid.tagId = spot["tag_id"].toString();

        string passback = spot["passback"].asString();
        vector<string> passbackFields = split(passback, ',');

        bid.account = AccountKey(passbackFields.at(1));
        bid.bidTimestamp =
            Date::parseSecondsSinceEpoch(passbackFields.at(2));

        bids.push_back(bid);
    }

    return make_pair(true, bids);
}


auto
MockExchange::
recvBid() -> pair<bool, vector<Bid> >
{
    array<char, 16384> buffer;

    int res = recv(toRouterFd, buffer.data(), buffer.size(), 0);
    if (res == 0 || (res == -1 && errno == ECONNRESET)) {
        connect();
        return make_pair(false, vector<Bid>());
    }

    ExcCheckErrno(res != -1, "recv");

    return parseResponse(string(buffer.data(), res));
}


void
MockExchange::
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

    string str = ML::DB::serializeToString(event);
    toPostAuctionService.sendMessage("WIN", str);
}


} // namepsace RTBKIT
