/* nsq_logger.h   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/

#pragma once

#include "soa/service/ilogger.h"
#include "soa/service/nsq_client.h"   
#include "soa/service/message_loop.h"

namespace Datacratic {


/****************************************************************************/
/* NSQ LOGGER                                                               */
/****************************************************************************/

/* Implementation of the ilogger interface using NSQ client*/
struct NsqLogger : public ILogger {

    NsqLogger(const std::string & loggerUrl,
              const OnMessageReceived & onMessageReceived = nullptr);

    virtual ~NsqLogger();

    virtual void init(const std::string & loggerUrl);
    virtual void subscribe(const std::string & topic, 
    					   const std::string & channel);

    virtual void consumeMessage(const std::string & messageId);
	virtual void publishMessage(const std::string & topic,
                                const std::string & message);

private:

    void onClosed(bool fromPeer, 
                  const std::vector<std::string> & msgs);

    std::unique_ptr<NsqClient> client;
    MessageLoop loop;
    int closed_;
 
};

} // namespace Datacratic
