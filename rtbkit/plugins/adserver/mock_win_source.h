/** mock_win_source.h                                 -*- C++ -*-
    Eric Robert, 20 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

*/

#pragma once

#include "rtbkit/common/testing/exchange_source.h"

namespace RTBKIT {

struct MockWinSource : public WinSource {
    MockWinSource(NetworkAddress address);
    MockWinSource(Json::Value const & json);

    void sendWin(const BidRequest& br,
                 const Bid& bid,
                 const Amount& winPrice);

private:
    void sendEvent(const PostAuctionEvent& ev);
};

} // namespace RTBKIT
