/* logger_client.h   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/
#pragma once

#include "soa/service/logger.h"

namespace Datacratic {

struct LoggerClient{


    LoggerClient(const std::string & loggerType, 
                 const std::string & loggerUrl,
                 OnClosing onClosing = nullptr,
                 const OnMessageReceived & onMessageReceived = nullptr);

    void subscribe(const std::string & topic, 
                   const std::string & channel);   

    void consumeMessage(const std::string & messageId);
private:
    std::shared_ptr<Logger> logger;
};

} // namespace Datacratic