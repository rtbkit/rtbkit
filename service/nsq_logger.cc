/* nsq_logger.cc   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/

#include "soa/service/nsq_logger.h"


using namespace Datacratic;

NsqLogger::NsqLogger(const std::string & loggerUrl,
   	   	  			 OnClosing onClosing,
          			 const OnMessageReceived & onMessageReceived) 
{
	client.reset(new NsqClient(onClosing, onMessageReceived));
	init(loggerUrl);
	client->connectSync();
}

void NsqLogger::init(const std::string & loggerUrl)
{
	client->init(loggerUrl);
}

void NsqLogger::subscribe(const std::string & topic, 
			   			  const std::string & channel) 
{

	client->sub(topic,channel);
}

void NsqLogger::consumeMessage(const std::string & messageId)
{
	client->fin(messageId);
}

void NsqLogger::publishMessage(const std::string & topic,
                               const std::string & message)
{
    client->pub(topic,message);
}
