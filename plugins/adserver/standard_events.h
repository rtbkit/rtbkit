/* standard_events.h                                                 -*-C++-*-
   Wolfgang Sourdeau, September 2013
   Copyright (C) 2013 Datacratic.  All rights reserved.

 */

#pragma once

#include <string>
#include <vector>

#include "soa/types/id.h"


namespace RTBKIT {

struct StandardWin {
    StandardWin()
        : timestamp(-1.0), bidTimestamp(-1.0), winPrice(-1.0), dataCost(-1.0)
    {}

    /* timestamp of the win event */
    double timestamp;

    /* winning auction */
    double bidTimestamp;
    Datacratic::Id auctionId;
    Datacratic::Id adSpotId;
    std::vector<Datacratic::Id> userIds;

    /* prices */
    double winPrice;
    double dataCost;

    Datacratic::Id accountId;
    Json::Value meta;
};

struct StandardExternalWin {
    StandardExternalWin()
        : timestamp(-1.0), winPrice(-1.0), dataCost(-1.0)
    {}

    double timestamp;
    double bidTimestamp;
    Datacratic::Id auctionId;
    Datacratic::Id adSpotId;
    std::vector<Datacratic::Id> userIds;
    double winPrice;
    double dataCost;

    /* original bid request */
    Json::Value bidRequest;
};

struct StandardClick {
    StandardClick()
        : timestamp(-1.0), bidTimestamp(-1.0)
    {}

    double timestamp;

    Datacratic::Id auctionId;
    Datacratic::Id adSpotId;
    std::vector<Datacratic::Id> userIds;

    double bidTimestamp;

    std::string event; // "click"
};

struct StandardConversion {
    StandardConversion()
        : timestamp(-1.0), bidTimestamp(-1.0), payout(-1.0)
    {}

    double timestamp;

    double bidTimestamp;
    Datacratic::Id auctionId;
    Datacratic::Id adSpotId;
    std::vector<Datacratic::Id> userIds;

    double payout;
    
    std::string event; // "conversion"
};


} // namespace RTBKIT
