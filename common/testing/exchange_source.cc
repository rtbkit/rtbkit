/** exchange_source.cc                                 -*- C++ -*-
    Eric Robert, 6 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Implementation of exchange sources

*/

#include "exchange_source.h"
#include "rtbkit/core/post_auction/post_auction_loop.h"
#include "soa/service/http_header.h"

#include <array>

using namespace std;
using namespace RTBKIT;

ExchangeSource::ExchangeSource(int port) :
    addr(0), fd(-1)
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


ExchangeSource::~ExchangeSource()
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

        //ML::sleep(0.1);
    }
}

void
BidSource::
write(std::string const & request)
{
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
    array<char, 16384> buffer;

    int res = recv(fd, buffer.data(), buffer.size(), 0);
    if (res == 0 || (res == -1 && errno == ECONNRESET)) {
        return "";
    }

    ExcCheckErrno(res != -1, "recv");

    close(fd);
    fd = -1;

    return string(buffer.data(), res);
}

void
BidSource::
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

    write(httpRequest);
}


auto
BidSource::
parseResponse(const string& rawResponse) -> pair<bool, vector<Bid>>
{
    Json::Value payload;

    if(rawResponse.empty()) {
        return make_pair(false, vector<Bid>());
    }

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
BidSource::
recvBid() -> pair<bool, vector<Bid>>
{
    return parseResponse(read());
}


auto
BidSource::
makeBidRequest() -> BidRequest
{
    BidRequest bidRequest;

    FormatSet formats;
    formats.push_back(Format(160,600));
    AdSpot spot;
    spot.id = Id(1);
    spot.formats = formats;
    bidRequest.imp.push_back(spot);

    formats[0] = Format(300,250);
    spot.id = Id(2);
    bidRequest.imp.push_back(spot);

    bidRequest.location.countryCode = "CA";
    bidRequest.location.regionCode = "QC";
    bidRequest.location.cityName = "Montreal";
    bidRequest.auctionId = Id(id * 10000000 + key);
    bidRequest.exchange = "mock";
    bidRequest.language = "en";
    bidRequest.url = Url("http://datacratic.com");
    bidRequest.timestamp = Date::now();
    bidRequest.userIds.add(Id(std::string("foo")), ID_EXCHANGE);
    bidRequest.userIds.add(Id(std::string("bar")), ID_PROVIDER);
    ++key;

    return bidRequest;
}


void
WinSource::
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
