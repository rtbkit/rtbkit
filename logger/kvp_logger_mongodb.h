#pragma once

#include "soa/logger/kvp_logger_interface.h"
#include <iostream>
#include "mongo/client/dbclient.h"

namespace Datacratic{

/**
 * This kvp_logger logs to mongodb
 */
class KvpLoggerMongoDb : public IKvpLogger{
    public:
        KvpLoggerMongoDb(const KvpLoggerParams&);
        void log(const std::map<std::string, std::string>&, const std::string&);
        void log(Json::Value&, const std::string&);

    private:
        mongo::DBClientConnection conn;
        const IKvpLogger::KvpLoggerParams params;
        void doIt(std::function<void()>& fct);
};


}//namespace Datacratic
