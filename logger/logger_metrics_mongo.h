#pragma once

#include "logger_metrics_interface.h"
#include "mongo/client/dbclient.h"

namespace Datacratic{
class LoggerMetricsMongo : public ILoggerMetrics{
    public:
        LoggerMetricsMongo(Json::Value config,
                           const std::string& coll,
                           const std::string& appName);
        void logMetrics(Json::Value&);
        void logProcess(Json::Value&);
        void logMeta(Json::Value&);

    private:
        const std::string coll;
        bool failSafe;
        mongo::OID objectId;
        std::string db;
        mongo::DBClientConnection conn;
        void doIt(std::function<void()>& fct);
        static mongo::BSONObj _fromJson(const Json::Value&);
        void logInCategory(const std::string&, Json::Value&);
};
}//namespace Datacratic
