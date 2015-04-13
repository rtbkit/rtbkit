/* logger_metrics_term.h                                          -*- C++ -*-
   François-Michel L'Heureux, 3 June 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#pragma once

#include "logger_metrics_interface.h"


namespace Datacratic {

/****************************************************************************/
/* LOGGER METRICS TERM                                                      */
/****************************************************************************/

struct LoggerMetricsTerm : public ILoggerMetrics {
    friend class ILoggerMetrics;

protected:
    LoggerMetricsTerm(Json::Value config,
                      const std::string & coll,
                      const std::string & appName);
    void logInCategory(const std::string &, const Json::Value &);
    void logInCategory(const std::string & category,
                       const std::vector<std::string> & path,
                       const NumOrStr & val);
    std::string getProcessId() const;

private:
    std::string pid;
};

} // namespace Datacratic
