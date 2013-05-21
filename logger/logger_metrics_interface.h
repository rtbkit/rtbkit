#pragma once

#include <iostream>
#include <ostream>
#include <cstring>
#include <boost/shared_ptr.hpp>
#include "jml/arch/exception.h"
#include "soa/jsoncpp/json.h"
#include <mutex>

namespace Datacratic{

/**
 * KvpLogger are key-value-pair loggers
 */
class ILoggerMetrics{
    public:
        static void setup(const std::string& configKey,
                   const std::string& coll,
                   const std::string& appName);
        /**
         * Factory like getter for kvp
         */
        static std::shared_ptr<ILoggerMetrics> getSingleton();

        virtual void logMetrics(Json::Value&) = 0;
        virtual void logProcess(Json::Value&) = 0;
        virtual void logMeta(Json::Value&) = 0;

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
        virtual ~ILoggerMetrics(){};

    protected:
        std::string collection;
        static std::string parentObjectId;
};
}//namespace Datacratic

