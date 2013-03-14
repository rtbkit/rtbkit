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

    int first;
    int last;
};


/******************************************************************************/
/* PORT RANGE SERVICE                                                         */
/******************************************************************************/

struct PortRangeService
{
    virtual ~PortRangeService() {}

    virtual PortRange getRange(const std::string& name) = 0;

    virtual void dump(std::ostream& stream = std::cerr) const {}
};


/******************************************************************************/
/* DEFAULT PORT RANGE SERVICE                                                 */
/******************************************************************************/

struct DefaultPortRangeService : public PortRangeService
{
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

struct JsonPortRangeService : public PortRangeService
{
    JsonPortRangeService(const Json::Value& config);

    virtual PortRange getRange(const std::string& name);

    virtual void dump(std::ostream& stream = std::cerr) const;

private:
    std::map<std::string, PortRange> rangeMap;
};


} // namespace Datacratic
