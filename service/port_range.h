/* port_range.h                                                     -*- C++ -*-
   Eric Robert, 22 February 2013
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Port range
*/

#pragma once

#include "jml/arch/exception.h"

namespace Datacratic {

struct PortRange
{
    PortRange() : first(15000), last(15999) {
    }

    PortRange(int port) : first(port), last(port) {
        if(port == -1) {
            first = 15000;
            last = 15999;
        }
    }

    PortRange(int first, int last) : first(first), last(last) {
        if(last < first)
            throw ML::Exception("invalid port range");
    }

    template<typename F>
    int bindPort(F operation) const {
        for(int i = first; i <= last; ++i) {
            if(operation(i)) return i;
        }

        return -1;
    }

private:
    int first;
    int last;
};

} // namespace Datacratic
