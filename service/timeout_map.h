/* timeout_map.h                                                   -*- C++ -*-
   Jeremy Barnes, 2 February 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Map from key -> value with inbuilt timeouts.

   Eventually will allow persistance.
*/

#ifndef __router__timeout_map_h__
#define __router__timeout_map_h__

#include <map>
#include "soa/types/date.h"
#include <boost/function.hpp>
#include "jml/arch/exception.h"
#include <math.h>

namespace Datacratic {

template<typename Key, class Value>
struct TimeoutMap {

    TimeoutMap(double defaultTimeout = -INFINITY)
        : defaultTimeout(defaultTimeout), earliest(Date::positiveInfinity())
    {
    }

    double defaultTimeout;

    boost::function<void (const std::string & reason)> throwException;

    void doThrowException(const std::string & reason) const
    {
        if (throwException) throwException(reason);
        else throw ML::Exception(reason);
        std::cerr << "TimeoutMap exception thrower returned" << std::endl;
        abort();
    }

    struct Node;

    /** Returns true if the key is in the map. */
    bool count(const Key & key)
    {
        return nodes.count(key);
    }

    /** Access the entry for the given node.  If it already exists then
        return the existing entry; otherwise insert it with the default
        timeout.
    */
    Node & operator [] (const Key & key)
    {
        auto res = nodes.insert(std::make_pair(key, Node()));
        auto it = res.first;
        if (res.second) {
            if (!std::isnormal(defaultTimeout) || defaultTimeout < 0.0)
                doThrowException("no default timeout specified and insert "
                                 "not used");
            Date timeout = Date::now().plusSeconds(defaultTimeout);
            it->second.timeoutIt = timeouts.insert(std::make_pair(timeout, it));
            if (timeout < earliest) earliest = timeout;
        }
        
        return it->second;
    }
    
    /** Return the given key or insert a default value if it doesn't exist.
        Updates the timeout to the given value.
    */
    Node & access(const Key & key, Date timeout)
    {
        auto res = nodes.insert(std::make_pair(key, Node(Value(), timeout)));
        auto it = res.first;
        if (res.second) {
            // inserted... insert the timeout
            it->second.timeoutIt = timeouts.insert(std::make_pair(timeout, it));
            if (timeout < earliest) earliest = timeout;
        }
        else {
            // already existed... update the timeout
            updateTimeout(it, timeout);
        }
        return it->second;
    }

    /** Insert the given key, value pair with the given timeout.  Throws an
        exception if the key already exists.
    */
    Node & insert(const Key & key, const Value & value,
                  Date timeout)
    {
        auto res = nodes.insert(std::make_pair(key, Node(value, timeout)));
        if (!res.second) {
            std::cerr << "key = " << key << std::endl;
            std::cerr << "contents (" << nodes.size() << ") = " << std::endl;
            int n = 0;
            for (auto it = nodes.begin(), end = nodes.end();  it != end && n < 20;  ++it, ++n)
                std::cerr << it->first << " @ " << it->second.timeoutIt->first << " "
                          << (it->first == key ? "*****" : "") << std::endl;
            doThrowException("TimeoutMap: "
                             "attempt to re-insert existing key");
        }
        auto it = res.first;
        it->second.timeoutIt = timeouts.insert(std::make_pair(timeout, it));
        if (timeout < earliest) earliest = timeout;
        return it->second;
    }

    /** Insert the given key, value pair with the given timeout.  Throws an
        exception if the key already exists.
    */
    Node & insert(const Key & key, Value && value, Date timeout)
    {
        auto res = nodes.insert(std::make_pair(key, Node(value, timeout)));
        if (!res.second) {
            std::cerr << "key = " << key << std::endl;
            std::cerr << "contents (" << nodes.size() << ") = " << std::endl;
            int n = 0;
            for (auto it = nodes.begin(), end = nodes.end();  it != end && n < 20;  ++it, ++n)
                std::cerr << it->first << " @ " << it->second.timeoutIt->first << " "
                          << (it->first == key ? "*****" : "") << std::endl;
            doThrowException("TimeoutMap: "
                             "attempt to re-insert existing key");
        }
        auto it = res.first;
        it->second.timeoutIt = timeouts.insert(std::make_pair(timeout, it));
        if (timeout < earliest) earliest = timeout;
        return it->second;
    }

    /** Update the given key which must already exist. */
    Node & update(const Key & key, Value && value)
    {
        auto it = nodes.find(key);
        if (it == nodes.end())
            doThrowException("TimeoutMap: "
                             "attempt to update nonexistant key");
        Value & v = it->second;
        v = value;
        return it->second;
    }
    
    /** Update the given key which must already exist. */
    Node & update(const Key & key, const Value & value)
    {
        auto it = nodes.find(key);
        if (it == nodes.end())
            doThrowException("TimeoutMap: "
                             "attempt to update nonexistant key");
        Value & v = it->second;
        v = value;
        return it->second;
    }

    void updateTimeout(const Key & key, Date timeout)
    {
        auto it = nodes.find(key);
        if (it == nodes.end())
            doThrowException("TimeoutMap: "
                             "attempt to update nonexistant key");
        updateTimeout(it, timeout);
    }

#if 0
    /** Look up the given key and return its node.  Throws an exception if
        it isn't already there.
    */
    const Node & operator [] (const Key & key) const
    {
        auto res = nodes.insert(std::make_pair(key, Node()));
        if (!res.second)
            doThrowException("TimeoutMap: "
                             "attempt to re-insert existing key");
        auto it = res.first;
        return it->second;
    }
#endif

    /** Call the callback on any which have expired, removing them from
        the map.
    */
    template<typename Callback>
    void expire(const Callback & callback, Date now = Date::now())
    {
        // Look for loss timeout expiries
        for (auto it = timeouts.begin(), end = timeouts.end();
             it != end && it->first <= now;  /* no inc */) {
            auto it2 = it;
            auto expired = it->second;
            Date newExpiry = callback(expired->first, expired->second);
            ++it;
            if (newExpiry != Date()) {
                auto tit = expired->second.timeoutIt;
                timeouts.erase(tit);
                expired->second.timeoutIt
                    = timeouts.insert(make_pair(newExpiry, expired));
                if (newExpiry < earliest) earliest = newExpiry;
            } else {
                erase(expired);
            }
        }
    }

    /** Remove any which have expired. */
    void expire(Date now = Date::now())
    {
        // Look for loss timeout expiries
        for (auto it = timeouts.begin(), end = timeouts.end();
             it != end && it->first <= now;  /* no inc */) {
            auto expired = it->second;
            ++it;
            erase(expired);
        }
    }
    
    typedef std::map<Key, Node> Nodes;
    Nodes nodes;

    /** Ordered set of timeouts in submitted for auction loss messages. */
    typedef std::multimap<Date, typename Nodes::iterator> Timeouts;
    Timeouts timeouts;

    // Date of the earliest timeout
    Date earliest;

    struct Node : public Value {
        Node() {}
        Node(const Value & val, Date timeout)
            : Value(val), timeout(timeout)
        {
        }

        Node(Value && val, Date timeout)
            : Value(val), timeout(timeout)
        {
        }

        Date timeout;
        typename Timeouts::iterator timeoutIt;
    };

    typedef typename Nodes::const_iterator const_iterator;
    typedef typename Nodes::iterator iterator;

    bool count(const Key & key) const
    {
        return nodes.count(key);
    }

    Value get(const Key & key) const
    {
        auto it = find(key);
        if (it == end()) return Value();
        return it->second;
    }

    iterator find(const Key & key)
    {
        return nodes.find(key);
    }

    const_iterator find(const Key & key) const
    {
        return nodes.find(key);
    }

    iterator begin()
    {
        return nodes.begin();
    }

    iterator end()
    {
        return nodes.end();
    }

    const_iterator begin() const
    {
        return nodes.begin();
    }

    const_iterator end() const
    {
        return nodes.end();
    }

    /** Remove the entry for the given key.  Returns true if it was erased
        or false otherwise.
    */
    bool erase(const Key & key)
    {
        auto it = nodes.find(key);
        if (it == nodes.end()) return false;
        erase(it);
        return true;
    }

    void erase(const typename Nodes::iterator & it)
    {
        if (it == nodes.end())
            doThrowException("erasing with invalid iterator");
        auto tit = it->second.timeoutIt;
        nodes.erase(it);
        timeouts.erase(tit);
        if (timeouts.empty())
            earliest = Date::positiveInfinity();
        else earliest = timeouts.begin()->first;
    }

    void updateTimeout(const iterator & it, Date timeout)
    {
        if (it == nodes.end())
            throw ML::Exception("attempt to update wrong timeout");

        auto & tit = it->second.timeoutIt;
        timeouts.erase(tit);
        tit = timeouts.insert(std::make_pair(timeout, it));
        earliest = timeouts.begin()->first;
    }

    size_t size() const
    {
        return nodes.size();
    }

    bool empty() const
    {
        return nodes.empty();
    }

    void clear()
    {
        timeouts.clear();
        nodes.clear();
        earliest = Date::positiveInfinity();
    }
};


} // namespace Datacratic


#endif /* __router__timeout_map_h__ */
