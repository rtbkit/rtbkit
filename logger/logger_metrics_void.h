/* logger_metrics_void.h                                           -*- C++ -*-
   François-Michel L'Heureux, 21 May 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#pragma once

#include "logger_metrics_interface.h"


namespace Datacratic {

/****************************************************************************/
/* LOGGER METRICS VOID                                                      */
/****************************************************************************/

struct LoggerMetricsVoid : public ILoggerMetrics {
    friend class ILoggerMetrics;

protected:
    LoggerMetricsVoid(Json::Value config,
                      const std::string & coll,
                      const std::string & appName)
        : ILoggerMetrics(coll)
    {
    }
    LoggerMetricsVoid(const std::string & coll)
        : ILoggerMetrics(coll)
    {
    }

    void logInCategory(const std::string &, const Json::Value &)
    {
    }
    void logInCategory(const std::string & category,
                       const std::vector<std::string> & path,
                       const NumOrStr & val)
    {
    }

    std::string getProcessId()
        const
    {
        return "";
    }
};

} // namespace Datacratic
