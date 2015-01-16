/** mock_bid_source.cc                                 -*- C++ -*-
    Eric Robert, 14 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

*/

#include "mock_bid_source.h"
#include "soa/service/http_header.h"

using namespace Datacratic;
using namespace RTBKIT;

BidRequest MockBidSource::generateRandomBidRequest() {
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
    bidRequest.auctionId = Id(rng.random());
    bidRequest.exchange = "mock";
    bidRequest.language = "en";
    bidRequest.url = Url("http://datacratic.com");
    bidRequest.timestamp = Date::now();
    bidRequest.userIds.add(Id(rng.random()), ID_EXCHANGE);
    bidRequest.userIds.add(Id(rng.random()), ID_PROVIDER);

    std::string strBidRequest = bidRequest.toJsonStr();
    std::string httpRequest = ML::format(
        "%s %s HTTP/1.1\r\n"
        "Content-Length: %zd\r\n"
        "Content-Type: application/json\r\n"
        "Connection: Keep-Alive\r\n"
        "\r\n"
        "%s",
        verb,
        path,
        strBidRequest.size(),
        strBidRequest.c_str());

    write(httpRequest);
    return bidRequest;
}

auto
MockBidSource::
parseResponse(const std::string& rawResponse) -> std::pair<bool, std::vector<Bid>>
{
    Json::Value payload;

    if(rawResponse.empty()) {
        return std::make_pair(false, std::vector<Bid>());
    }

    try {
        HttpHeader header;
        header.parse(rawResponse);
        if (!header.contentLength || header.resource != "200") {
            return std::make_pair(false, std::vector<Bid>());
        }

        payload = Json::parse(header.knownData);
    }
    catch (const std::exception & exc) {
        std::cerr << "invalid response received: " << exc.what() << std::endl;
        return std::make_pair(false, std::vector<Bid>());
    }

    if (payload.isMember("error")) {
        std::cerr << "error returned: "
                  << payload["error"] << std::endl
                  << payload["details"] << std::endl;
        return std::make_pair(false, std::vector<Bid>());
    }

    ExcAssert(payload.isMember("imp"));

    std::vector<Bid> bids;

    for (size_t i = 0; i < payload["imp"].size(); ++i) {
        auto& spot = payload["imp"][i];

        Bid bid;

        bid.adSpotId = Id(spot["id"].asString());
        bid.maxPrice = MicroUSD_CPM(spot["max_price"].asInt());
        bid.account = AccountKey(spot["account"].asString(), '.');
        bids.push_back(bid);
    }

    return std::make_pair(true, bids);
}

namespace {

struct AtInit {
    AtInit()
    {
        PluginInterface<BidSource>::registerPlugin("mock",
						   [](Json::Value const & json) {
            return new MockBidSource(json);
        });
    }
} atInit;
  

}

