/** monitor_indicator.h                                 -*- C++ -*-
    Rémi Attab, 24 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Indicator message used to communicate between the provider and the service.

*/

#pragma once

#include "soa/jsoncpp/value.h"

#include <string>

namespace RTBKIT {


/******************************************************************************/
/* MONITOR INDICATOR                                                          */
/******************************************************************************/

struct MonitorIndicator
{
    std::string serviceName;
    bool status;
    std::string message;

    Json::Value toJson() const
    {
        Json::Value value;

        value["serviceName"] = serviceName;
        value["status"] = status;
        value["message"] = message;

        return value;
    }

    static MonitorIndicator fromJson(const Json::Value& json)
    {
        MonitorIndicator ind;

        ind.serviceName = json["serviceName"].asString();
        ind.status = json["status"].asBool();
        ind.message = json["message"].asString();

        return ind;
    }
};

} // namespace RTBKIT
