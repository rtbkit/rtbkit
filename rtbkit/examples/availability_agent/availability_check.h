/** availability_check.h                                 -*- C++ -*-
    Rémi Attab, 26 Jul 2012
    Copyright (c) 2012 Recoset.  All rights reserved.

    Datastore for the availability check stuff.

*/

#ifndef __rtb__availability_check_h__
#define __rtb__availability_check_h__

#include "rtbkit/common/bid_request.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/core/router/filter_pool.h"
#include "soa/gc/gc_lock.h"
#include "soa/jsoncpp/value.h"
#include "soa/service/stats_events.h"

#include <atomic>
#include <vector>

namespace Datacratic {


/******************************************************************************/
/* AVAILABILITY CHECK                                                         */
/******************************************************************************/

/** Uses the recorded bid requests to calculate the effectiveness of the filters
    in the agent config. Note that no QPS adjustments are performed by this
    class.

    SWMR but could easily be expanded to MWMR if pos is atomically incremented
    before inserting the new element.

*/
struct AvailabilityCheck
{
    /** size is the total number of BidRequest to keep in the ring buffer. */
    AvailabilityCheck(size_t size);

    /** Atomically adds a bid request to the ring buffer.

        Thread-safe with concurrent calls to report but not with itself.
    */
    void addRequest(const RTBKIT::BidRequest& br);

    /** Reports the effectiveness of each filters employed by the router. */
    Json::Value checkConfig(const RTBKIT::AgentConfig& config);


    /** Notify any attached logger that an event took place. */
    std::function<void(const std::string&, StatEventType, float)> onEvent;

private:
    size_t size;
    size_t pos;
    std::vector<RTBKIT::BidRequest*> requests; // ring buffer.
    GcLock gcLock;

    RTBKIT::FilterPool filters;
};

} // Recoset

#endif // __rtb__availability_check_h__
