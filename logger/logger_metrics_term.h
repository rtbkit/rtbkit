#pragma once

#include "logger_metrics_interface.h"

namespace Datacratic{
class LoggerMetricsTerm : public ILoggerMetrics{
    friend class ILoggerMetrics;

    protected:
        LoggerMetricsTerm(Json::Value config,
                           const std::string& coll,
                           const std::string& appName);
        void logInCategory(const std::string&,
                           const Json::Value&);
        void logInCategory(const std::string& category,
                           const std::vector<std::string>& path,
                           const NumOrStr& val);
        const std::string getProcessId() const;

    private:
        std::string pid;
};
}//namespace Datacratic
