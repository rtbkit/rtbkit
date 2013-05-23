#pragma once

#include "logger_metrics_interface.h"

namespace Datacratic{
class LoggerMetricsVoid : public ILoggerMetrics{
    public:
        LoggerMetricsVoid(Json::Value config,
                           const std::string& coll,
                           const std::string& appName){};

    protected:
        void logInCategory(const std::string&, Json::Value&){};
        void logInCategory(const std::string& category,
                           const std::vector<std::string>& path,
                           const NumOrStr& val){};
};
}//namespace Datacratic
