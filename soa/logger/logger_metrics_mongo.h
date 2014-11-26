#pragma once

#include "logger_metrics_interface.h"
#include "mongo/client/dbclient.h"

namespace Datacratic{
class LoggerMetricsMongo : public ILoggerMetrics{
    friend class ILoggerMetrics;

    protected:
        mongo::OID objectId;
        std::string db;
        std::shared_ptr<mongo::DBClientBase> conn;

        LoggerMetricsMongo(Json::Value config,
                           const std::string& coll,
                           const std::string& appName);
        void logInCategory(const std::string&,
                           const Json::Value&);
        void logInCategory(const std::string& category,
                           const std::vector<std::string>& path,
                           const NumOrStr& val);
        const std::string getProcessId() const;
    
    private:
        bool logToTerm;
};
}//namespace Datacratic
