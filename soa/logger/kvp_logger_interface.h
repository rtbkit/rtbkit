#pragma once

#include <iostream>
#include <boost/shared_ptr.hpp>
#include "jml/arch/exception.h"
#include <map>
#include "soa/jsoncpp/json.h"

namespace Datacratic{

/**
 * KvpLogger are key-value-pair loggers
 */
class IKvpLogger{
    public:
        struct KvpLoggerParams{
            std::string hostAndPort;//format host:port
            std::string db;
            std::string user;
            std::string pwd;
            bool failSafe;//if true, all errors are catched but printed to cerr
        };

        /**
         * Factory like getter for kvp
         */
        static std::shared_ptr<IKvpLogger>
            kvpLoggerFactory(const std::string& type, const KvpLoggerParams&);
        static std::shared_ptr<IKvpLogger>
            kvpLoggerFactory(const std::string& configKey);

        virtual void log(const std::map<std::string, std::string>&, const std::string&) = 0;
        virtual void log(Json::Value&, const std::string&) = 0;
        template <class jsonifiable>
        void log(const jsonifiable& j, const std::string& s){
                Json::Value root = j.toJson();
                log(root, s);
        };
        virtual ~IKvpLogger(){};
};


}//namespace Datacratic
