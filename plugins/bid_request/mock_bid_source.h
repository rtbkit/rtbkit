/** mock_bid_source.h                                 -*- C++ -*-
    Eric Robert, 14 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Mock bid sources.

*/

#pragma once

#include "rtbkit/common/testing/exchange_source.h"
#include "jml/utils/rng.h"

namespace RTBKIT {

struct MockBidSource : public BidSource {
    std::string host;
    std::string verb;
    std::string path;

    MockBidSource(NetworkAddress address) :
        BidSource(std::move(address)) {
        init();
    }

    MockBidSource(NetworkAddress address, int lifetime) :
        BidSource(std::move(address), lifetime) {
        init();
    }

    MockBidSource(Json::Value const & json) :
        BidSource(json),
        host(json.get("host", "").asString()),
        verb(json.get("verb", "").asString()),
        path(json.get("path", "").asString()) {
        init();
    }

    void init() {
        if(host.empty()) host = ML::hostname();
        if(verb.empty()) verb = "POST";
        if(path.empty()) path = "/bids";
    }

    BidRequest generateRandomBidRequest();

    auto parseResponse(const std::string& rawResponse) -> std::pair<bool, std::vector<Bid>>;
};

} // namespace RTBKIT

