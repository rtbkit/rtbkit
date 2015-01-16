/** rest_request_params_types.h                                    -*- C++ -*-
    Wolfgang Sourdeau, 8 January 2015
    Copyright (c) 2015 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include <string>

#include "soa/types/date.h"
#include "soa/service/rest_request_params.h"


namespace Datacratic {

template<>
struct RestCodec<Datacratic::Date> {
    static Date decode(const std::string & str)
    {
        return Date::parseIso8601DateTime(str);
    }

    static std::string encode(const Date & val)
    {
        return val.printIso8601();
    }
};

} // namespace Datacratic
