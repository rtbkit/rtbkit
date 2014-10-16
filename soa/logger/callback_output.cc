/* callback_output.cc
   Jeremy Barnes, 29 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "callback_output.h"


using namespace std;


namespace Datacratic {

/*****************************************************************************/
/* CALLBACK OUTPUT                                                            */
/*****************************************************************************/

CallbackOutput::
CallbackOutput(const Callback & callback)
    : callback(callback)
{
}

CallbackOutput::
~CallbackOutput()
{
}

void
CallbackOutput::
logMessage(const std::string & channel,
           const std::string & message)
{
    if (callback)
        callback(channel, message);
}

void
CallbackOutput::
close()
{
}

} // namespace Datacratic
