/** augmentor_events_publisher.cc                                 -*- C++ -*-
    Rémi Attab, 20 Nov 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Augmentor events publisher implementations.

*/

#include "augmentor_events_publisher.h"
#include "soa/service/zmq_utils.h"
#include "rtbkit/common/bid_request.h"

#include <boost/lexical_cast.hpp>
#include <string>

using namespace std;
using namespace ML;


namespace RTBKIT {


/******************************************************************************/
/* AUGMENTOR EVENT PUBLISHER                                                  */
/******************************************************************************/

AugmentorEventsPublisher::
AugmentorEventsPublisher(const string& socketURI, const string& identity,
                         zmq::context_t & context) :
    context(context),
    serviceSocket(context, ZMQ_PUB)
{
    setIdentity(serviceSocket, identity);
    serviceSocket.bind(socketURI.c_str());
}

void
AugmentorEventsPublisher::
publishFrequencyCapEvent(
        const string& providerId,
        const string& exchangeId,
        const string& campaign,
        uint64_t sessionLength)
{
    sendMessage(
            serviceSocket, "frequency-cap",
            providerId, exchangeId, campaign,
            boost::lexical_cast<string>(sessionLength));
}


void
AugmentorEventsPublisher::
publishGstImpressionEvent(const BidRequest& bidRequest, const Datacratic::Id& adSpotId)
{
    sendMessage(
            serviceSocket, "global-stats-table",
            "IMPRESSION",
            bidRequest.toJsonStr(),
            to_string(adSpotId.type),
            adSpotId.toString());
}


void
AugmentorEventsPublisher::
publishGstClickEvent(const BidRequest& bidRequest, const Datacratic::Id& adSpotId)
{
    sendMessage(
            serviceSocket, "global-stats-table",
            "CLICK",
            bidRequest.toJsonStr(),
            to_string(adSpotId.type),
            adSpotId.toString());
}


} // namespace RTBKIT
