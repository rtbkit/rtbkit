/** openrtb_bid_source.h                                 -*- C++ -*-
    Eric Robert, 13 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Mock bid sources for OpenRTB.

*/

#pragma once

#include "rtbkit/common/testing/exchange_source.h"
#include "jml/utils/rng.h"

namespace RTBKIT {

struct OpenRTBBidSource : public BidSource {
    std::string host;
    std::string verb;
    std::string resource;

    OpenRTBBidSource(Json::Value const & json) :
        BidSource(json),
        host(json.get("host", ML::hostname()).asString()),
        verb(json.get("verb", "POST").asString()),
        resource(json.get("resource", "/").asString()) {
    }

    BidRequest generateRandomBidRequest();

    auto parseResponse(const std::string& rawResponse) -> std::pair<bool, std::vector<Bid>>;
};

} // namespace RTBKIT

