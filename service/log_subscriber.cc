/* log_subscriber.cc   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/

#include "soa/service/log_subscriber.h"
#include "soa/service/nsq_logger.h"

static const std::string NSQ_LOGGING = "nsqLogging";

using namespace Datacratic;

LogSubscriber::
LogSubscriber(const std::string & loggerType, 
        	    const std::string & loggerUrl,
              const OnMessageReceived & onMessageReceived)
{
    if (loggerType == NSQ_LOGGING)
        logger.reset(new NsqLogger(loggerUrl, onMessageReceived));
}


void 
LogSubscriber::
subscribe(const std::string & topic, 
          const std::string & channel) 
{
    logger->subscribe(topic,channel);
}

void
LogSubscriber::
consumeMessage(const std::string & messageId)
{
    logger->consumeMessage(messageId);
}

