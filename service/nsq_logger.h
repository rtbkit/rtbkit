/* nsq_logger.h   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/

#pragma once

#include "soa/service/logger.h"
#include "soa/service/nsq_client.h"   

namespace Datacratic {

struct NsqLogger : public Logger {

    NsqLogger(const std::string & loggerUrl,
    	   	  OnClosing onClosing = nullptr,
              const OnMessageReceived & onMessageReceived = nullptr);

    virtual ~NsqLogger(){}

    virtual void init(const std::string & loggerUrl);
    virtual void subscribe(const std::string & topic, 
    					   const std::string & channel);

    virtual void consumeMessage(const std::string & messageId);
	virtual void publishMessage(const std::string & topic,
                                const std::string & message);

private:

	std::unique_ptr<NsqClient> client;
 
};

} // namespace Datacratic
