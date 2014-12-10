/* logger_publisher.h   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/

#pragma once

#include "soa/types/date.h"
#include "soa/service/ilogger.h"   
#include <string>
#include <functional>

namespace Datacratic {


/****************************************************************************/
/* LOG PUBLISHER                                                             */
/****************************************************************************/

/* This struct is responsible for publishing to a topic . */
struct LogPublisher{
  
    LogPublisher(const std::string & loggerType, 
                 const std::string & loggerUrl);

    ~LogPublisher() {}

    void init(const std::string & loggerUrl);
    void publishMessage(const std::string & topic,
    							      const std::string & message);

private:
    std::shared_ptr<ILogger> logger;

};

} // namespace Datacratic