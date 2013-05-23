#pragma once

#include <iostream>
#include <ostream>
#include <cstring>
#include <boost/shared_ptr.hpp>
#include "jml/arch/exception.h"
#include "soa/jsoncpp/json.h"
#include <mutex>
#include "boost/variant.hpp"

namespace Datacratic{

/**
 * KvpLogger are key-value-pair loggers
 * Mix of patterns:
 * - Provides a factory to instanciate a logger
 * - Provides an interface for metrics loggers
 * - Provides adaptor functions to avoid defining redundant functions in
 *   implementations
 */
class ILoggerMetrics{
    protected:
        typedef boost::variant<int, float, double> Numeric;
        typedef boost::variant<int, float, double, std::string> NumOrStr;

        std::string collection;
        static std::string parentObjectId;
        const static std::string METRICS;
        const static std::string PROCESS;
        const static std::string META;

        virtual void logInCategory(const std::string& category,
                                   const std::vector<std::string>& path,
                                   const NumOrStr& val) = 0;
        virtual void logInCategory(const std::string& category,
                                   Json::Value& j) = 0;

    public:

        static std::shared_ptr<ILoggerMetrics> setup(
            const std::string& configKey,
            const std::string& coll,
            const std::string& appName);
        /**
         * Factory like getter for kvp
         */
        static std::shared_ptr<ILoggerMetrics> getSingleton();

        void logMetrics(Json::Value&);
        void logProcess(Json::Value& j){
            logInCategory(PROCESS, j);
        }
        void logMeta(Json::Value& j){
            logInCategory(META, j);
        }

        template <class jsonifiable>
        void logMetrics(const jsonifiable& j){
                Json::Value root = j.toJson();
                logMetrics(root);
        };
        template <class jsonifiable>
        void logProcess(const jsonifiable& j){
                Json::Value root = j.toJson();
                logProcess(root);
        };
        template <class jsonifiable>
        void logMeta(const jsonifiable& j){
                Json::Value root = j.toJson();
                logMeta(root);
        };

        void logMetrics(const std::vector<std::string>& path, const Numeric& val){
            logInCategory(METRICS, path, val);
        }
        void logProcess(const std::vector<std::string>& path, const NumOrStr& val){
            logInCategory(PROCESS, path, val);
        }
        void logMeta(const std::vector<std::string>& path, const NumOrStr& val){
            logInCategory(META, path, val);
        }

        virtual ~ILoggerMetrics(){};

};
}//namespace Datacratic

