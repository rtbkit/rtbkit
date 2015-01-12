/* event_subscriber.cc   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/

#include "soa/service/event_subscriber.h"
#include "soa/service/nsq_event_handler.h"

static const std::string NSQ_LOGGING = "nsqLogging";

using namespace Datacratic;


EventSubscriber::
EventSubscriber(const std::string & loggerType, 
                const std::string & loggerUrl,
                const OnMessageReceived & onMessageReceived)
{
    if (loggerType == NSQ_LOGGING)
        logger.reset(new NsqEventHandler(loggerUrl, onMessageReceived));
}

void 
EventSubscriber::
subscribe(const std::string & topic, const std::string & channel) 
{
    logger->subscribe(topic,channel);
}

void
EventSubscriber::
consumeMessage(const std::string & messageId)
{
    logger->consumeMessage(messageId);
}
