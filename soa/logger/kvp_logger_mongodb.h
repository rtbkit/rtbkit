#pragma once

#include "soa/logger/kvp_logger_interface.h"
#include <iostream>
#include "mongo/client/dbclient.h"
#include <boost/property_tree/json_parser.hpp>

namespace Datacratic{

/**
 * This kvp_logger logs to mongodb
 */
class KvpLoggerMongoDb : public IKvpLogger{
    public:
        KvpLoggerMongoDb(const KvpLoggerParams&);
        KvpLoggerMongoDb(const boost::property_tree::ptree&);
        void log(const std::map<std::string, std::string>&, const std::string&);
        void log(Json::Value&, const std::string&);

    private:
        mongo::DBClientConnection conn;
        const std::string db;
        const bool failSafe;
        void doIt(std::function<void()>& fct);

        //std::function<void()> makeInitFct(KvpLoggerParams& );
        std::function<void()> makeInitFct(const std::string& hostAndPort,
                                          const std::string& db,
                                          const std::string& user,
                                          const std::string& pwd);
};


}//namespace Datacratic
