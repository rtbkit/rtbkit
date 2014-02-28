/* blacklist.h                                                     -*- C++ -*-
   Jeremy Barnes, 1 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Blacklist functionality.
*/

#ifndef __rtb_router__blacklist_h__
#define __rtb_router__blacklist_h__

#include <string>
#include <vector>
#include "rtbkit/common/bid_request.h"
#include "rtbkit/core/router/router_types.h"
#include "soa/service/timeout_map.h"


namespace RTBKIT {

struct AgentConfig;


/*****************************************************************************/
/* BLACKLIST INFO                                                            */
/*****************************************************************************/

/** For the given user, contains information on who has blacklisted them
    for how much time.
*/
struct BlacklistInfo {
    struct Entry {
        std::string agent;
        AccountKey account;
        std::string site;
        Date expiry;
    };
    std::vector<Entry> entries;
    Date earliestExpiry;

    /* Does the given agent and bid request match the blacklist? */
    bool matches(const BidRequest & request,
                 const std::string & agent,
                 const AgentConfig & agentConfig) const;
        
    /** Add the given entry to the blacklist.  Returns Date() if the
        entry is not the earliest expiring entry, or the date of the
        new expiry if not.
    */
    Date add(const BidRequest & bidRequest,
             const std::string & agent,
             const AgentConfig & agentConfig);
        
    /* Expire any that need to be expired, and return the next lowest
       expiry date or an empty date if none.
    */
    Date expire(Date now);
};


/*****************************************************************************/
/* BLACKLIST                                                                 */
/*****************************************************************************/

/** Indexed on user ID */
struct Blacklist {
    void doExpiries();

    size_t size() const { return entries.size(); }
    
    bool matches(const BidRequest & request,
                 const std::string & agentName,
                 const AgentConfig & config) const;

    void add(const BidRequest & bidRequest,
             const std::string & agent,
             const AgentConfig & agentConfig);
    
    typedef TimeoutMap<Id, BlacklistInfo> Entries;
    Entries entries;
};

} // namespace RTBKIT

#endif /* __rtb_router__blacklist_h__ */
