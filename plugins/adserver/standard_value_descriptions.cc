/* value_description.cc                                            -*- C++ -*-
   Wolfgang Sourdeau, September 2013
   Copyright (C) 2013 Datacratic.  All rights reserved.

*/

#include "soa/types/basic_value_descriptions.h"

#include "value_description.h"

using namespace std;
using namespace RTBKIT;

namespace Datacratic {

DefaultDescription<StandardWin>::
DefaultDescription()
{
    addField("timestamp", &StandardWin::timestamp, "");
    addField("bidTimestamp", &StandardWin::bidTimestamp, "");
    addField("auctionId", &StandardWin::auctionId, "");
    addField("adSpotId", &StandardWin::adSpotId, "");
    addField("userIds", &StandardWin::userIds, "");
    addField("winPrice", &StandardWin::winPrice, "");
    addField("dataCost", &StandardWin::dataCost, "");
    addField("accountId", &StandardWin::accountId, "");
    addField("winMeta", &StandardWin::meta, "");
}

DefaultDescription<StandardExternalWin>::
DefaultDescription()
{
    addField("timestamp", &StandardExternalWin::timestamp, "");
    addField("bidTimestamp", &StandardExternalWin::bidTimestamp, "");
    addField("auctionId", &StandardExternalWin::auctionId, "");
    addField("adSpotId", &StandardExternalWin::adSpotId, "");
    addField("userIds", &StandardExternalWin::userIds, "");
    addField("winPrice", &StandardExternalWin::winPrice, "");
    addField("dataCost", &StandardExternalWin::dataCost, "");
    addField("bidRequest", &StandardExternalWin::bidRequest, "");
}

DefaultDescription<StandardClick>::
DefaultDescription()
{
    addField("timestamp", &StandardClick::timestamp, "");
    addField("bidTimestamp", &StandardClick::bidTimestamp, "");
    addField("auctionId", &StandardClick::auctionId, "");
    addField("adSpotId", &StandardClick::adSpotId, "");
    addField("userIds", &StandardClick::userIds, "");
    addField("event", &StandardClick::event, "");
}

DefaultDescription<StandardConversion>::
DefaultDescription()
{
    addField("timestamp", &StandardConversion::timestamp, "");
    addField("bidTimestamp", &StandardConversion::bidTimestamp, "");
    addField("auctionId", &StandardConversion::auctionId, "");
    addField("adSpotId", &StandardConversion::adSpotId, "");
    addField("userIds", &StandardConversion::userIds, "");
    addField("payout", &StandardConversion::payout, "");
    addField("event", &StandardConversion::event, "");
}

} // namespace Datacratic
