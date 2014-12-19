/* event_publisher.h                                                   -*-C++-*-
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

*/

#pragma once

#include "soa/types/date.h"
#include "soa/service/event_handler.h"
#include <string>
#include <functional>

namespace Datacratic {


/****************************************************************************/
/* EVENT PUBLISHER                                                             */
/****************************************************************************/

/* This struct is responsible for publishing to a topic . */
struct EventPublisher{
    EventPublisher(const std::string & loggerType,
                   const std::string & loggerUrl);

    ~EventPublisher() {}

    void init(const std::string & loggerUrl);
    void publishMessage(const std::string & topic,
                        const std::string & message);

private:
    std::shared_ptr<EventHandler> logger;

};

} // namespace Datacratic
