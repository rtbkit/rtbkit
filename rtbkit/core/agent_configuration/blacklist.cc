/* blacklist.cc                                                    -*- C++ -*-
   Jeremy Barnes, 1 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Blacklist functionality.
*/

#include "blacklist.h"
#include "agent_config.h"

namespace RTBKIT {

Date
BlacklistInfo::
add(const BidRequest & bidRequest,
    const std::string & agent,
    const AgentConfig & config)
{
    Entry entry;
    entry.agent = agent;
    entry.account = config.account;
    entry.site = bidRequest.url.toString();
    entry.expiry = Date::now().plusSeconds(config.blacklistTime);

    entries.push_back(entry);

    if (entries.size() == 1 || entry.expiry < earliestExpiry) {
        earliestExpiry = entry.expiry;
        return earliestExpiry;
    }

    return Date(); // no need to reset the expiry
}

Date
BlacklistInfo::
expire(Date now)
{
    Date result = Date();
    for (unsigned i = 0;  i < entries.size();  /* no inc */) {
        if (entries[i].expiry <= now) {
            std::swap(entries[i], entries.back());
            entries.pop_back();
            // no increment
        }
        else {
            if (result == Date())
                result = entries[i].expiry;
            else result = std::min(result, entries[i].expiry);
            ++i;
        }
    }
    return result;
}

bool
BlacklistInfo::
matches(const BidRequest & bidRequest,
        const std::string & agent,
        const AgentConfig & config) const
{
    auto matchesEntry = [&] (const Entry & entry) -> bool
        {
            switch (config.blacklistScope) {
                case BL_AGENT:
                    return entry.agent == agent;
                case BL_ACCOUNT:
                    return entry.account == config.account;
            default:
                throw ML::Exception("invalid blacklist scope");
            }
        };

    switch (config.blacklistType) {
    case BL_OFF:
        return false;  // shouldn't happen

    case BL_USER:  // fall through
        for (unsigned i = 0;  i < entries.size();  ++i) {
            const Entry & entry = entries[i];
            if (matchesEntry(entry)) return true;
        }

    case BL_USER_SITE:
        for (unsigned i = 0;  i < entries.size();  ++i) {
            const Entry & entry = entries[i];
            if (!matchesEntry(entry)) continue;
            if (entry.site != "" && entry.site == bidRequest.url.toString())
                return true;
        }
        return false;

    default:
        throw ML::Exception("unknown blacklist type");
    }
}

void
Blacklist::
doExpiries()
{
    Date start = Date::now();

    auto onBlacklistFinished = [&] (const Id & userId,
                                    BlacklistInfo & info)
        {
            return info.expire(start);
        };

    entries.expire(onBlacklistFinished, start);
}

bool
Blacklist::
matches(const BidRequest & bidRequest, const std::string & agentName,
        const AgentConfig & config) const
{  
    bool blocked = false;
    const Id & exchangeId = bidRequest.userIds.exchangeId;
    if (!blocked && exchangeId) {
        // TODO: read lock
        auto bit = entries.find(exchangeId);
        if (bit != entries.end()) {
            const BlacklistInfo & binfo = bit->second;
            if (binfo.matches(bidRequest, agentName, config))
                blocked = true;
        }
    }
    const Id & providerId = bidRequest.userIds.providerId;
    if (!blocked && providerId) {
        // TODO: read lock
        auto bit = entries.find(providerId);
        if (bit != entries.end()) {
            const BlacklistInfo & binfo = bit->second;
            if (binfo.matches(bidRequest, agentName, config))
                blocked = true;
        }
    }

    return blocked;
}

void
Blacklist::
add(const BidRequest & bidRequest, const std::string & agent,
    const AgentConfig & agentConfig)
{
    auto addToBlacklist = [&] (const Id & id)
        {
            if (!id) return;

            auto it = this->entries.find(id);
            if (it != this->entries.end()) {
                Date newExpiry
                    = it->second.add(bidRequest, agent, agentConfig);
                if (newExpiry != Date())
                    this->entries.updateTimeout(it, newExpiry);
            }
            else {
                BlacklistInfo binfo;
                Date timeout = binfo.add(bidRequest, agent, agentConfig);
                this->entries.insert(id, binfo, timeout);
            }
        };
    
    addToBlacklist(bidRequest.userIds.exchangeId);
    addToBlacklist(bidRequest.userIds.providerId);
}

} // namespace RTBKIT
