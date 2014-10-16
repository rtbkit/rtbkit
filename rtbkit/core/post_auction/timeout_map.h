/* timeout_map.h                                 -*- C++ -*-
   Rémi Attab (remi.attab@gmail.com), 08 May 2014
   FreeBSD-style copyright and disclaimer apply

   Map that maintains a timeout mechanism.

   Simpler version of the soa TimeoutMap which doesn't require linear scans to
   expire elements. Should eventually replace the one in soa.

*/

#pragma once

#include "soa/types/date.h"

#include <set>
#include <queue>

namespace RTBKIT {

/******************************************************************************/
/* TIMEOUT MAP                                                                */
/******************************************************************************/

template<typename Key, typename Value>
struct TimeoutMap
{

    size_t size() const
    {
        return map.size();
    }

    bool count(const Key& key) const
    {
        return map.count(key);
    }

    Value& get(const Key& key)
    {
        auto it = map.find(key);
        ExcCheck(it != map.end(), "key not present in the timeout map.");
        return it->second.value;
    }

    const Value& get(const Key& key) const
    {
        auto it = map.find(key);
        ExcCheck(it != map.end(), "key not present in the timeout map.");
        return it->second.value;
    }

    bool emplace(Key key, Value value, Datacratic::Date timeout)
    {
        auto ret = map.insert(std::make_pair(
                        std::move(key), Entry(std::move(value), timeout)));
        if (!ret.second) return false;

        queue.emplace(ret.first->first, timeout);
        return true;
    }

    void update(const Key& key, Datacratic::Date timeout)
    {
        auto it = map.find(key);
        ExcCheck(it != map.end(), "key not present in the timeout map.");

        it->second.timeout = timeout;
        queue.emplace(key, timeout);
    }

    Value pop(const Key& key)
    {
        auto it = map.find(key);
        ExcCheck(it != map.end(), "key not present in the timeout map.");

        Value value = std::move(it->second.value);
        map.erase(it);
        return value;
    }

    bool erase(const Key& key)
    {
        return map.erase(key);
    }

    template<typename Fn>
    size_t expire(const Fn& fn, Datacratic::Date now = Datacratic::Date::now())
    {
        std::vector< std::pair<Key, Entry> > toExpire;
        toExpire.reserve(1 << 4);

        while (!queue.empty() && queue.top().timeout <= now) {
            TimeoutEntry entry = std::move(queue.top());
            queue.pop();

            auto it = map.find(entry.key);
            if (it == map.end()) continue;
            if (it->second.timeout > now) continue;

            toExpire.emplace_back(std::move(*it));
            map.erase(it);
        }

        for (auto& entry : toExpire)
            fn(std::move(entry.first), std::move(entry.second.value));

        return toExpire.size();
    }

private:

    struct Entry
    {
        Value value;
        Datacratic::Date timeout;

        Entry(Value value, Datacratic::Date timeout) :
            value(std::move(value)), timeout(timeout)
        {}
    };

    struct TimeoutEntry
    {
        Key key;
        Datacratic::Date timeout;

        TimeoutEntry(Key key, Datacratic::Date timeout) :
            key(std::move(key)), timeout(timeout)
        {}

        bool operator<(const TimeoutEntry& other) const
        {
            return timeout > other.timeout;
        }
    };

    std::unordered_map<Key, Entry> map;
    std::priority_queue<TimeoutEntry> queue;
};

} // namespace RTBKIT
