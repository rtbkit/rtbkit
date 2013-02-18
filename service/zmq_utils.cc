/* zmq_utils.cc
   Jeremy Barnes, 14 January 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#include "zmq_utils.h"

namespace Datacratic {

std::string printZmqEvent(int event)
{
#define printZmqEventImpl(ev) case ev: return #ev

    switch (event) {
        printZmqEventImpl(ZMQ_EVENT_LISTENING);
        printZmqEventImpl(ZMQ_EVENT_BIND_FAILED);
        printZmqEventImpl(ZMQ_EVENT_ACCEPTED);
        printZmqEventImpl(ZMQ_EVENT_ACCEPT_FAILED);
        printZmqEventImpl(ZMQ_EVENT_CONNECTED);
        printZmqEventImpl(ZMQ_EVENT_CONNECT_DELAYED);
        printZmqEventImpl(ZMQ_EVENT_CONNECT_RETRIED);
        printZmqEventImpl(ZMQ_EVENT_CLOSE_FAILED);
        printZmqEventImpl(ZMQ_EVENT_CLOSED);
        printZmqEventImpl(ZMQ_EVENT_DISCONNECTED);
    default:
        return ML::format("ZMQ_EVENT_UNKNOWN(%d)", event);
    }

#undef printZmqEventImpl
}

bool zmqEventIsError(int event)
{
    switch (event) {
    case ZMQ_EVENT_BIND_FAILED:
    case ZMQ_EVENT_ACCEPT_FAILED:
    case ZMQ_EVENT_CONNECT_DELAYED:
    case ZMQ_EVENT_CLOSE_FAILED:
        return true;
    default:
        return false;
    }
}

} // namespace Datacratic
