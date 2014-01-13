/** openrtb_bid_source.cc                                 -*- C++ -*-
    Eric Robert, 13 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

*/

#include "openrtb_bid_source.h"
#include "openrtb_bid_request.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/service/http_header.h"

using namespace Datacratic;
using namespace RTBKIT;

BidRequest OpenRTBBidSource::generateRandomBidRequest() {
    key += rng.random();

    OpenRTB::BidRequest req;
    req.id = Id(rng.random());
    req.tmax.val = 50;
    req.at = AuctionType::SECOND_PRICE;
    req.imp.emplace_back();
    auto & imp = req.imp[0];
    imp.id = Id(key);
    imp.banner.reset(new OpenRTB::Banner);
    imp.banner->w.push_back(300);
    imp.banner->h.push_back(250);

    StructuredJsonPrintingContext context;
    DefaultDescription<OpenRTB::BidRequest> desc;
    desc.printJson(&req, context);
    std::string content = context.output.toString();

    int length = content.length();

    std::string message = ML::format(
        "%s %s HTTP/1.1\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "accept: */*\r\n"
        "connection: Keep-Alive\r\n"
        "host: %s\r\n"
        "user-agent: be2/1.0\r\n"
        "x-openrtb-version: 2.1\r\n"
        "\r\n%s",
        verb.c_str(), resource.c_str(), length, host.c_str(), content.c_str());

    write(message);
    std::unique_ptr<BidRequest> br(fromOpenRtb(std::move(req), "openrtb", "openrtb"));
    return BidRequest(*br);
}

auto
OpenRTBBidSource::
parseResponse(const std::string& rawResponse) -> std::pair<bool, std::vector<Bid>> {
    OpenRTB::BidResponse response;

    if(rawResponse.empty()) {
        return std::make_pair(false, std::vector<Bid>());
    }

    try {
        HttpHeader header;
        header.parse(rawResponse);
        if (!header.contentLength || header.resource != "200") {
            //std::cerr << rawResponse << std::endl;
            return std::make_pair(false, std::vector<Bid>());
        }

        ML::Parse_Context context("payload", header.knownData.c_str(), header.knownData.size());
        StreamingJsonParsingContext json(context);
        DefaultDescription<OpenRTB::BidResponse> desc;
        desc.parseJson(&response, json);
    }
    catch (const std::exception & exc) {
        std::cerr << "invalid response received: " << exc.what() << std::endl;
        return std::make_pair(false, std::vector<Bid>());
    }

    std::vector<Bid> bids;

    for(auto i = 0; i != response.seatbid[0].bid.size(); ++i) {
        Bid bid;
        bid.adSpotId = response.seatbid[0].bid[i].impid;
        bid.maxPrice = USD_CPM(response.seatbid[0].bid[i].price.val);
        bids.push_back(bid);
    }

    return std::make_pair(true, bids);
}

namespace {

struct AtInit {
    AtInit()
    {
        BidSource::registerBidSourceFactory("openrtb", [](Json::Value const & json) {
            return new OpenRTBBidSource(json);
        });
    }
} atInit;

}

