/** standard_event_source.cc                                 -*- C++ -*-
    Eric Robert, 20 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

*/

#include "standard_event_source.h"

using namespace RTBKIT;

StandardEventSource::
StandardEventSource(NetworkAddress address) :
    EventSource(std::move(address)) {
}


StandardEventSource::
StandardEventSource(Json::Value const & json) :
    EventSource(json) {
}


void
StandardEventSource::
sendImpression(const BidRequest& bidRequest, const Bid& bid)
{
    Json::Value json;
    json["timestamp"] = Date::now().secondsSinceEpoch();
    json["bidRequestId"] = bidRequest.auctionId.toString();
    json["impid"] = bid.adSpotId.toString();
    json["userIds"] = bidRequest.userIds.toJsonArray();
    json["type"] = "CONVERSION";
    sendEvent(json); 
}


void
StandardEventSource::
sendClick(const BidRequest& bidRequest, const Bid& bid)
{
    Json::Value json;
    json["timestamp"] = Date::now().secondsSinceEpoch();
    json["bidRequestId"] = bidRequest.auctionId.toString();
    json["impid"] = bid.adSpotId.toString();
    json["userIds"] = bidRequest.userIds.toJsonArray();
    json["type"] = "CLICK";
    sendEvent(json); 
}


void
StandardEventSource::
sendEvent(Json::Value const & json)
{
    std::string str = json.toString();
    std::string httpRequest = ML::format(
            "POST /events HTTP/1.1\r\n"
            "Content-Length: %zd\r\n"
            "Content-Type: application/json\r\n"
            "Connection: Keep-Alive\r\n"
            "\r\n"
            "%s",
            str.size(),
            str.c_str());

    write(httpRequest);

    std::string result = read();
    std::string status = "HTTP/1.1 200 OK";

    if(result.compare(0, status.length(), status)) {
        std::cerr << result << std::endl;
    }
}


namespace {

struct AtInit {
    AtInit()
    {
        PluginInterface<EventSource>::registerPlugin("standard",
						     [](Json::Value const & json) {
	    return new StandardEventSource(json);
        });
    }
} atInit;

}

