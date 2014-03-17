/** mock_win_source.h                                 -*- C++ -*-
    Eric Robert, 20 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

*/

#pragma once

#include "rtbkit/common/testing/exchange_source.h"

namespace RTBKIT {

struct MockEventSource : public EventSource {
    MockEventSource(NetworkAddress address);
    MockEventSource(Json::Value const & json);

    void sendImpression(const BidRequest& br, const Bid& bid);
    void sendClick(const BidRequest& br, const Bid& bid);

private:
    void sendEvent(const PostAuctionEvent& ev);
};

} // namespace RTBKIT
