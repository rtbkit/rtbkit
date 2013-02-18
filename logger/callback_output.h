/* callback_output.h                                                -*- C++ -*-
   Jeremy Barnes, 29 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Output that callbacks on a zeromq socket.
*/

#ifndef __logger__callback_output_h__
#define __logger__callback_output_h__

#include "logger.h"

namespace Datacratic {

/*****************************************************************************/
/* CALLBACK OUTPUT                                                           */
/*****************************************************************************/

/** Class that passes messages to a callback.
*/

struct CallbackOutput : public LogOutput {

    typedef boost::function<void (std::string, std::string)> Callback;

    /** Create a logger with its own zeromq context. */
    CallbackOutput(const Callback & callback);

    virtual ~CallbackOutput();

    virtual void logMessage(const std::string & channel,
                            const std::string & message);

    virtual void close();

    Callback callback;
};




} // namespace Datacratic


#endif /* __logger__callback_output_h__ */
