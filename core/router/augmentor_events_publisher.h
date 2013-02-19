/** augmentor_events_publisher.h                                 -*- C++ -*-
    Rémi Attab, 20 Nov 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Publishes misc events to the augmentors.

    Famous Last Words: This should be temporary until we find a more
    standardized way of doing it.

*/

#ifndef __rtb__augmentor_events_publisher_h__
#define __rtb__augmentor_events_publisher_h__

#include "soa/types/id.h"

#include "soa/service/zmq.hpp"
#include <string>


namespace RTBKIT {

struct BidRequest;

/******************************************************************************/
/* AUGMENTOR EVENTS PUBLISHER                                                 */
/******************************************************************************/

/** Publishes events to the augmentors through ZMQ. */
struct AugmentorEventsPublisher
{
    AugmentorEventsPublisher(
            const std::string& socketURI,
            const std::string& identity,
            zmq::context_t & context);

    /** Send a win event to the frequency cap augmentor. */
    void publishFrequencyCapEvent(
            const std::string& providerId,
            const std::string& exchangeId,
            const std::string& campaign,
            uint64_t sessionLength);

    /** Send a win event to the global stats table augmentor */
    void publishGstImpressionEvent(
            const BidRequest& bidRequest, const Datacratic::Id& adSpotId);

    /** Send a click event to the global stats table augmentor */
    void publishGstClickEvent(
            const BidRequest& bidRequest, const Datacratic::Id& adSpotId);

private:
    zmq::context_t & context;
    zmq::socket_t serviceSocket;
};


} // namespace RTBKIT

#endif // __rtb__augmentor_events_publisher_h__
