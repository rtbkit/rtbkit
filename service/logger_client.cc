/* logger_client.cc   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/

#include "soa/service/logger_client.h"
#include "soa/service/nsq_logger.h"


static const std::string NSQ_LOGGING = "nsqLogging";

namespace Datacratic {

LoggerClient::LoggerClient(const std::string & loggerType, 
                           const std::string & loggerUrl,
                           OnClosing onClosing,
                           const OnMessageReceived & onMessageReceived) 
{
    if (loggerType == NSQ_LOGGING)
        logger.reset(new NsqLogger(loggerUrl, onClosing, onMessageReceived));
}


void LoggerClient::subscribe(const std::string & topic, 
               const std::string & channel) 
{
    logger->subscribe(topic,channel);
}

void LoggerClient::consumeMessage(const std::string & messageId)
{
    logger->consumeMessage(messageId);
}

void LoggerClient::publishMessage(const std::string & topic,
                    const std::string & message)
{
    logger->publishMessage(topic,message);    
}



} // namespace Datacratic