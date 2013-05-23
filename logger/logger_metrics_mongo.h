#pragma once

#include "logger_metrics_interface.h"
#include "mongo/client/dbclient.h"

namespace Datacratic{
class LoggerMetricsMongo : public ILoggerMetrics{
    public:
        LoggerMetricsMongo(Json::Value config,
                           const std::string& coll,
                           const std::string& appName);

    protected:
        const std::string coll;
        mongo::OID objectId;
        std::string db;
        mongo::DBClientConnection conn;
        void logInCategory(const std::string&, Json::Value&);
        void logInCategory(const std::string& category,
                           const std::vector<std::string>& path,
                           const NumOrStr& val);
};
}//namespace Datacratic
