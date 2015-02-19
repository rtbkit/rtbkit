/** post_auction_proxy.h                                 -*- C++ -*-
    Rémi Attab, 05 Aug 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Proxy class for the post auction service.

*/

#pragma once

#include "rtbkit/common/auction_events.h"

namespace Datacratic {

struct ServiceProxies;
struct ZmqMultipleNamedClientBusProxy;

}

namespace RTBKIT {

struct EventForwarder;

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
    PostAuctionProxy(Datacratic::ServiceBase& parent);
    PostAuctionProxy(std::shared_ptr<Datacratic::ServiceProxies> proxies);

    void init();

    // Returns true only the proxy is connected to all shards.
    bool isConnected() const;

    // Sends an auction to the post auction loop.
    void sendAuction(std::shared_ptr<SubmittedAuctionEvent> auction);

    // Sends an event to the post auction loop.
    void sendEvent(std::shared_ptr<PostAuctionEvent> event);

private:
    void initZMQ();
    void initHTTP();

    Datacratic::ServiceBase* parent;
    std::shared_ptr<Datacratic::ServiceProxies> proxies;

    size_t shards;
    std::unique_ptr<Datacratic::ZmqMultipleNamedClientBusProxy> zmq;
    std::vector< std::shared_ptr<EventForwarder> > http;
};

} // namespace RTBKIT
