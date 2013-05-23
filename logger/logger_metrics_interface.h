#pragma once

#include <iostream>
#include <ostream>
#include <cstring>
#include <boost/shared_ptr.hpp>
#include "jml/arch/exception.h"
#include "soa/jsoncpp/json.h"
#include <mutex>
#include "boost/variant.hpp"
#include <functional>

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
    private:
        static bool failSafe;

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

        void failSafeHelper(std::function<void()>);

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
            std::function<void()> fct = [&](){
                logInCategory(PROCESS, j);
            };
            failSafeHelper(fct);
        }
        void logMeta(Json::Value& j){
            std::function<void()> fct = [&](){
                logInCategory(META, j);
            };
            failSafeHelper(fct);
        }

        template <class jsonifiable>
        void logMetrics(const jsonifiable& j){
            std::function<void()> fct = [&](){
                Json::Value root = j.toJson();
                logMetrics(root);
            };
            failSafeHelper(fct);
        };
        template <class jsonifiable>
        void logProcess(const jsonifiable& j){
            std::function<void()> fct = [&](){
                Json::Value root = j.toJson();
                logProcess(root);
            };
            failSafeHelper(fct);
        };
        template <class jsonifiable>
        void logMeta(const jsonifiable& j){
            std::function<void()> fct = [&](){
                Json::Value root = j.toJson();
                logMeta(root);
            };
            failSafeHelper(fct);
        };

        void logMetrics(const std::vector<std::string>& path, const Numeric& val){
            std::function<void()> fct = [&](){
                logInCategory(METRICS, path, val);
            };
            failSafeHelper(fct);
        }
        void logProcess(const std::vector<std::string>& path, const NumOrStr& val){
            std::function<void()> fct = [&](){
                logInCategory(PROCESS, path, val);
            };
            failSafeHelper(fct);
        }
        void logMeta(const std::vector<std::string>& path, const NumOrStr& val){
            std::function<void()> fct = [&](){
                logInCategory(META, path, val);
            };
            failSafeHelper(fct);
        }

        virtual ~ILoggerMetrics(){};

};
}//namespace Datacratic

