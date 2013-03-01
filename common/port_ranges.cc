/* port_ranges.cc
   Eric Robert, 22 February 2013
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Port ranges
*/

#include "rtbkit/common/port_ranges.h"

namespace RTBKIT {

Datacratic::PortRange PortRanges::logs(16000, 16999);
Datacratic::PortRange PortRanges::router(17000, 17999);
Datacratic::PortRange PortRanges::augmentors(18000, 18999);
Datacratic::PortRange PortRanges::configuration(19000, 19999);
Datacratic::PortRange PortRanges::postAuctionLoop(20000, 20999);
Datacratic::PortRange PortRanges::postAuctionLoopAgents(21000, 21999);

PortRanges::Services PortRanges::zmq = {
    { 22000, 22999 }, // banker
    { 23000, 23999 }, // agentConfiguration
    { 24000, 24999 }, // monitor
};

PortRanges::Services PortRanges::http = {
    { 9985 }, // banker
    { 9986 }, // agentConfiguration
    { 9987 }, // monitor
};

} // namespace RTBKIT
