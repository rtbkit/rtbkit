#pragma once

#include "soa/logger/kvp_logger_interface.h"
#include <iostream>

namespace Datacratic{

/**
 * This doesnt do anything (Null object pattern)
 */
class KvpLoggerVoid : public IKvpLogger{
    public:
        KvpLoggerVoid(){}
        void log(const std::map<std::string, std::string>&, const std::string&){}
        void log(Json::Value&, const std::string&){}
};


}//namespace Datacratic
