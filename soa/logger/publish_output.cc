/* publish_output.cc
   Jeremy Barnes, 29 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "publish_output.h"


using namespace std;


namespace Datacratic {

/*****************************************************************************/
/* PUBLISH OUTPUT                                                            */
/*****************************************************************************/

PublishOutput::
PublishOutput()
    : context(new zmq::context_t(1)),
      sock(*context, ZMQ_PUB)
{
}

PublishOutput::
PublishOutput(zmq::context_t & context)
    : context(ML::make_unowned_std_sp(context)),
      sock(context, ZMQ_PUB)
{
}

PublishOutput::
PublishOutput(std::shared_ptr<zmq::context_t> context)
    : context(context),
      sock(*context, ZMQ_PUB)
{
}

PublishOutput::
~PublishOutput()
{
}

void
PublishOutput::
bind(const std::string & uri)
{
    //cerr << "publishing to " << uri << endl;
    sock.bind(uri.c_str());
}

void
PublishOutput::
logMessage(const std::string & channel,
           const std::string & message)
{
    sendMesg(sock, channel, ZMQ_SNDMORE);
    sendMesg(sock, message, 0);
}

void
PublishOutput::
close()
{
    int res = zmq_close(sock);
    if (res == -1)
        throw ML::Exception("zmq_close: %s", zmq_strerror(zmq_errno()));
}

} // namespace Datacratic
