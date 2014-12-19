/* event_subscriber.h                                                  -*-C++-*-
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

*/

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "soa/service/event_handler.h"

namespace Datacratic {

/****************************************************************************/
/* EVENT SUBSCRIBER                                                         */
/****************************************************************************/

/* This struct is responsible for subscribing to a topic and channel. */
struct EventSubscriber {
    EventSubscriber(const std::string & loggerType,
                    const std::string & loggerUrl,
                    const OnMessageReceived & onMessageReceived = nullptr);

    ~EventSubscriber(){}

    void init(const std::string & loggerUrl);
    void subscribe(const std::string & topic, const std::string & channel);

    void consumeMessage(const std::string & messageId);

private:
    std::shared_ptr<EventHandler> logger;
};

} // namespace Datacratic
