/* event_handler.h
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

*/

#pragma once

#include <string>
#include <functional>

#include "soa/types/date.h"

namespace Datacratic {

typedef std::function<void (Date, uint16_t,
                            const std::string &,
                            const std::string &)> OnMessageReceived;


/****************************************************************************/
/* EVENT HANDLER                                                            */
/****************************************************************************/

/* Interface for an event handler. */
struct EventHandler {
    virtual ~EventHandler() {}

    virtual void init(const std::string & loggerUrl) = 0;
    virtual void subscribe(const std::string & topic,
                           const std::string & channel) = 0;

    virtual void consumeMessage(const std::string & messageId) = 0;

    virtual void publishMessage(const std::string & topic,
                                const std::string & message) = 0;
};

} // namespace Datacratic
