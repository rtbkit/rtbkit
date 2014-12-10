/* log_publisher.cc   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/

#include "soa/service/log_publisher.h"
#include "soa/service/nsq_logger.h"


static const std::string NSQ_LOGGING = "nsqLogging";

using namespace Datacratic;

LogPublisher::
LogPublisher(const std::string & loggerType, 
             const std::string & loggerUrl) 
{
    if (loggerType == NSQ_LOGGING)
        logger.reset(new NsqLogger(loggerUrl));
}


void 
LogPublisher::
publishMessage(const std::string & topic,
               const std::string & message)
{
    logger->publishMessage(topic,message);    
}

