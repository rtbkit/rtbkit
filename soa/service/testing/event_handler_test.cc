#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <iostream>
#include <boost/test/unit_test.hpp>
#include "soa/service/event_publisher.h"
#include "soa/service/event_subscriber.h"
#include "jml/arch/timers.h"

using namespace std;
using namespace Datacratic;

const std::string messageSent("Super Message");
bool messageConsumed = false;

struct TestSubscribe {

	TestSubscribe(const std::string & loggerType, const std::string & loggerUrl) {
		auto onMessage = [&] (Date ts, uint16_t attempts,
	                          const string & messageId,
	                          const string & message)
	        {
	        	cout << "Message Received = " << message << endl;
	            if (message.compare(messageSent) == 0) {
	            	messageConsumed = true;
	            	eventSubscriber->consumeMessage(messageId);
	            }
	            else
	            	throw; 
	        };
	    eventSubscriber.reset(new EventSubscriber(loggerType, loggerUrl, onMessage));

	}

	~TestSubscribe() {}

	std::unique_ptr<EventSubscriber> eventSubscriber;
};

#if 0
BOOST_AUTO_TEST_CASE( test_event_handler )
{
	TestSubscribe subscriber("nsqLogging", "http://192.168.168.113:4150");
	subscriber.eventSubscriber->subscribe("Lookalikes-Client", "Clients_LAL");

    EventPublisher publisher("nsqLogging", "http://192.168.168.113:4150");
    publisher.publishMessage("Lookalikes-Client", messageSent);

    ML::sleep(1.0);

	BOOST_CHECK(messageConsumed);
}
#endif