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
    json["auctionId"] = bidRequest.auctionId.toString();
    json["adSpotId"] = bid.adSpotId.toString();
    json["accountId"] = bid.account.toString();
    json["winPrice"] = (double) USD_CPM(winPrice);
    json["userIds"] = bidRequest.userIds.toJson();

    sendEvent(json);
}


void
StandardWinSource::
sendImpression(const BidRequest& bidRequest, const Bid& bid)
{
}


void
StandardWinSource::
sendClick(const BidRequest& bidRequest, const Bid& bid)
{
}


void
StandardWinSource::
sendEvent(Json::Value const & json)
{
    std::string str = json.toString();
    std::string httpRequest = ML::format(
        "POST /win HTTP/1.1\r\n"
        "Content-Length: %zd\r\n"
        "Content-Type: application/json\r\n"
        "Connection: Keep-Alive\r\n"
        "\r\n"
        "%s",
        str.size(),
        str.c_str());

    write(httpRequest);
}

namespace {

struct AtInit {
    AtInit()
    {
        WinSource::registerWinSourceFactory("standard", [](Json::Value const & json) {
            return new StandardWinSource(json);
        });
    }
} atInit;

}

