/** post_auction_proxy.h                                 -*- C++ -*-
    Rémi Attab, 05 Aug 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Proxy class for the post auction service.

*/

#pragma once

#include "rtbkit/common/auction_events.h"
#include "soa/service/service_base.h"
#include "soa/service/zmq_endpoint.h"

namespace RTBKIT {


/******************************************************************************/
/* POST AUCTION PROXY                                                         */
/******************************************************************************/

/** Event submission proxy for the post auction loop. Takes care of directing
    messages to the correct post auction shard.

    Requires that the postAuctionShard configuration parameter be provided in
    the bootstrap.json to determine the number of active post auction shards. If
    not present, assumes that there's only one active post auction shard.
 */
struct PostAuctionProxy
{
    PostAuctionProxy(std::shared_ptr<Datacratic::ServiceProxies> proxies);

    void init();

    // Returns true only the proxy is connected to all shards.
    bool isConnected() const;

    // Sends an auction to the post auction loop.
    void sendAuction(SubmittedAuctionEvent auction);

    // Sends an event to the post auction loop.
    void sendEvent(PostAuctionEvent event);

private:
    size_t shards;
    std::shared_ptr<Datacratic::ServiceProxies> proxies;
    Datacratic::ZmqMultipleNamedClientBusProxy toPostAuction;

};

} // namespace RTBKIT
