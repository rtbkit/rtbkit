/* event_publisher.cc   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/

#include "soa/service/event_publisher.h"
#include "soa/service/nsq_event_handler.h"


static const std::string NSQ_LOGGING = "nsqLogging";

using namespace Datacratic;

EventPublisher::
EventPublisher(const std::string & loggerType, 
               const std::string & loggerUrl) 
{
    if (loggerType == NSQ_LOGGING)
        logger.reset(new NsqEventHandler(loggerUrl));
}


void 
EventPublisher::
publishMessage(const std::string & topic,
               const std::string & message)
{
    logger->publishMessage(topic,message);    
}

