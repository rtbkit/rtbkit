/* port_ranges.h                                                   -*- C++ -*-
   Eric Robert, 22 February 2013
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Predefined port ranges
*/

#pragma once

#include "soa/service/port_range.h"

namespace RTBKIT {

struct PortRanges
{
    static Datacratic::PortRange logs;
    static Datacratic::PortRange router;
    static Datacratic::PortRange augmentors;
    static Datacratic::PortRange agentConfiguration;
    static Datacratic::PortRange configuration;
    static Datacratic::PortRange postAuctionLoop;
    static Datacratic::PortRange postAuctionLoopAgents;
    static Datacratic::PortRange monitor;
};

} // namespace RTBKIT
