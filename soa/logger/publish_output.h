/* publish_output.h                                                -*- C++ -*-
   Jeremy Barnes, 29 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Output that publishes on a zeromq socket.
*/

#ifndef __logger__publish_output_h__
#define __logger__publish_output_h__

#include "logger.h"

namespace Datacratic {

/*****************************************************************************/
/* PUBLISH OUTPUT                                                            */
/*****************************************************************************/

/** Class that publishes messages to a zeromq socket.  Other things can
    subscribe to it to get the results.
*/

struct PublishOutput : public LogOutput {

    /** Create a logger with its own zeromq context. */
    PublishOutput();

    PublishOutput(zmq::context_t & context);

    PublishOutput(std::shared_ptr<zmq::context_t> context);

    virtual ~PublishOutput();

    /** Bind it to a port.  Other processes can connect to this to subscribe
        to the logging feed.
    */
    void bind(const std::string & uri);

    virtual void logMessage(const std::string & channel,
                            const std::string & message);

    virtual void close();

    /// Zeromq context that we use
    std::shared_ptr<zmq::context_t> context;

    /// Socket that we publish to
    zmq::socket_t sock;
};




} // namespace Datacratic


#endif /* __logger__publish_output_h__ */
