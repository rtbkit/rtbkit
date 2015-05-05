/* logger_metrics_mongo.h                                          -*- C++ -*-
   François-Michel L'Heureux, 21 May 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#pragma once

#include "mongo/client/dbclient.h"
#include "logger_metrics_interface.h"


namespace Datacratic {

/****************************************************************************/
/* LOGGER METRICS MONGO                                                     */
/****************************************************************************/

struct LoggerMetricsMongo : public ILoggerMetrics {
    friend class ILoggerMetrics;

protected:
    mongo::OID objectId;
    std::string db;
    std::shared_ptr<mongo::DBClientBase> conn;

    LoggerMetricsMongo(Json::Value config, const std::string & coll,
                       const std::string & appName);
    void logInCategory(const std::string &, const Json::Value &);
    void logInCategory(const std::string & category,
                       const std::vector<std::string> & path,
                       const NumOrStr & val);
    std::string getProcessId() const;

private:
    bool logToTerm;
};

}//namespace Datacratic
