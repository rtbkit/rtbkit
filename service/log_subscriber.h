/* logger_subscriber.h   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/

#pragma once

#include "soa/types/date.h"
#include "soa/service/ilogger.h"
#include <string>
#include <functional>

namespace Datacratic {


typedef std::function<void (Date, uint16_t,
                            const std::string &,
                            const std::string &)> OnMessageReceived;
typedef std::function<void(bool,
                           const std::vector<std::string> & msgs)> OnClosing;

/****************************************************************************/
/* LOG SUBSCRIBER                                                             */
/****************************************************************************/

/* This struct is responsible for subscribing to a topic and channel. */
struct LogSubscriber{
  
    LogSubscriber(const std::string & loggerType, 
                  const std::string & loggerUrl,
                  OnClosing onClosing = nullptr,
                  const OnMessageReceived & onMessageReceived = nullptr);

    ~LogSubscriber(){}

    void init(const std::string & loggerUrl);
    void subscribe(const std::string & topic, 
    			   const std::string & channel);

    void consumeMessage(const std::string & messageId);

private:
    std::shared_ptr<ILogger> logger;
};

} // namespace Datacratic