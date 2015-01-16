/** standard_win_source.cc                                 -*- C++ -*-
    Eric Robert, 20 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

*/

#include "standard_win_source.h"

using namespace RTBKIT;

StandardWinSource::
StandardWinSource(NetworkAddress address) :
    WinSource(std::move(address)) {
}


StandardWinSource::
StandardWinSource(Json::Value const & json) :
    WinSource(json) {
}

void
StandardWinSource::
sendWin(const BidRequest& bidRequest, const Bid& bid, const Amount& winPrice)
{
    Json::Value json;
    json["timestamp"] = Date::now().secondsSinceEpoch();
    json["bidTimestamp"] = bid.bidTimestamp.secondsSinceEpoch();
    json["bidRequestId"] = bidRequest.auctionId.toString();
    json["impid"] = bid.adSpotId.toString();
    json["accountId"] = bid.account.toString();
    json["price"] = (double) USD_CPM(winPrice);
    // The standard protocol assumes userIds will be an array without xchg and prov keys.
    json["userIds"] = bidRequest.userIds.toJsonArray();

    sendEvent(json);
}


void
StandardWinSource::
sendEvent(Json::Value const & json)
{
    std::string str = json.toString();
    std::string httpRequest = ML::format(
        "POST /wins HTTP/1.1\r\n"
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
        PluginInterface<WinSource>::registerPlugin("standard",
						   [](Json::Value const & json) {
            return new StandardWinSource(json);
        });
    }
} atInit;

}

