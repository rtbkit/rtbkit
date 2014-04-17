/** port_range_service.h                                 -*- C++ -*-
    Rémi Attab, 12 Mar 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Port range allocation service.
*/

#pragma once

#include "jml/utils/exc_check.h"

#include <iostream>
#include <mutex>
#include <map>

namespace Json { struct Value; }

namespace Datacratic {

/******************************************************************************/
/* PORT RANGE                                                                 */
/******************************************************************************/

/** Encapsulates a range of ports that a service can listen on. */

struct PortRange
{
    PortRange() : first(15000), last(16000) {}

    PortRange(int port) :
        first(port), last(port + 1)
    {
        if (port == -1) {
            first = 15000;
            last = 16000;
        }
    }

    PortRange(const std::string & rangeSpec)
    {
        errno = 0;
        char * endPtr = 0;
        first = strtol(rangeSpec.c_str(), &endPtr, 10);
        if (errno)
            throw ML::Exception(errno, "strtol() parsing first port in '" + rangeSpec + "'");
        if (first <= 0 || first >= 65536)
            throw ML::Exception("first port number %d out of range parsing '%s'",
                                first, rangeSpec.c_str());
        if (*endPtr == 0) {
            last = first + 1;
            return;
        }
        else if (*endPtr != '-')
            throw ML::Exception("Range spec is <int> or <int>-<int> parsing "
                                + rangeSpec);
        
        last = strtol(endPtr + 1, &endPtr, 10);
        if (errno)
            throw ML::Exception(errno, "strtol() parsing last port in'"
                                + rangeSpec + "'");
        if (last <= 0 || last >= 65536)
            throw ML::Exception("last port number %d out of range parsing '%s'",
                                last, rangeSpec.c_str());
        if (*endPtr != 0)
            throw ML::Exception("extra junk after last port parsing '%s'",
                                rangeSpec.c_str());
    }

    PortRange(int first, int last) :
        first(first), last(last)
    {
        ExcCheckGreater(last, first, "invalid port range");
    }

    template<typename F>
    int bindPort(F operation) const
    {
        for(int i = first; i < last; ++i)
            if(operation(i)) return i;

        return -1;
    }

    Json::Value toJson() const;
    static PortRange fromJson(const Json::Value & val);

    int first;
    int last;
};


/******************************************************************************/
/* PORT RANGE SERVICE                                                         */
/******************************************************************************/

/** Abstract base class for a service from which you can query a port range
    by name and receive a PortRange object.

    The goal of this class is to split up the port space into chunks, each of
    which is assigned to a particular kind of service.  When dealing with
    dynamic service creation, this allows us to avoid the situation where
    a service goes down and a different kind of service starts listening on
    its port, confusing anything that was connected and didn't know 
    immediately about the change.
*/

struct PortRangeService
{
    virtual ~PortRangeService() {}

    /** Return the range of ports for the given service name, and return the
        range.  If name is not found, then depending upon the implementation
        either a new range of ports will be defined for that service, or an
        exception will be thrown.
    */
    virtual PortRange getRange(const std::string& name) = 0;

    /** Dump the port ranges to the given stream. */
    virtual void dump(std::ostream& stream = std::cerr) const {}
};


/******************************************************************************/
/* DEFAULT PORT RANGE SERVICE                                                 */
/******************************************************************************/

/** Default implementation of the port range service.  This one will assign
    a new block of ports each time a new service name is created.
*/

struct DefaultPortRangeService : public PortRangeService
{
    /** Ininialize the default port range service.  The rangeSize
        parameter tells how big a chunk of ports to assign each
        time it is asked for one.
    */
    DefaultPortRangeService(unsigned rangeSize = 100);

    virtual PortRange getRange(const std::string& name);

    virtual void dump(std::ostream& stream = std::cerr) const;

private:
    const unsigned rangeSize;
    unsigned currentPort;

    mutable std::mutex lock;
    std::map<std::string, PortRange> rangeMap;

};


/******************************************************************************/
/* JSON PORT RANGE SERVICE                                                    */
/******************************************************************************/

/** Implementation of the port range service that gets its mapping by reading
    a JSON file.
*/

struct JsonPortRangeService : public PortRangeService
{
    JsonPortRangeService(const Json::Value& config);

    virtual PortRange getRange(const std::string& name);

    virtual void dump(std::ostream& stream = std::cerr) const;

private:
    std::map<std::string, PortRange> rangeMap;
};


} // namespace Datacratic
