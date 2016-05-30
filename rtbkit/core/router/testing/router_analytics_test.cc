/* router_analytics_test.cc

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <string>
#include <iostream>     // cerr
#include <exception>
#include <dlfcn.h>      // dlopen()
#include <vector>
#include <chrono>
#include <thread>

#include <boost/test/unit_test.hpp>

#include "rtbkit/core/router/router.h"
#include "rtbkit/common/analytics.h"
#include "soa/service/loop_monitor.h"

using namespace RTBKIT;

namespace local {

void loadLib(const std::string & file) 
{
    void * handle = dlopen(file.c_str(), RTLD_NOW);
    if (!handle) {
            std::cerr << dlerror() << std::endl; 
            throw std::runtime_error("Could not load the library");
    }    
}

} // end local

BOOST_AUTO_TEST_CASE( zmq_analytics_test ) 
{
    const char * msg1 = "TestError";
    const char * msg2 = "Anything";
    bool message_received = false;

    // normally this is done using --preload of ServiceProxyArguments
    local::loadLib("libzmq_analytics.so");
    
    // This is the config file given as command line option to router_runner
    std::string analytics_config_file = R"JSON({ "pluginName":"zmq" })JSON";
    Json::Value analytics_config = Json::parse(analytics_config_file);  
  
    
    Router router;
    router.initAnalytics(analytics_config);
    router.init();
    router.bindTcp();
    router.start();
   
    const auto & proxies = router.getServices(); 
    
    proxies->config->dump(std::cerr);
    
    Datacratic::ZmqNamedSubscriber mySubscriber(*(proxies->zmqContext));
    mySubscriber.init(proxies->config);
    mySubscriber.messageHandler = [&](vector<zmq::message_t> && msg) {
        message_received = true;
        std::cerr << "MY MESSAGE =======> ";
        for(const auto & m : msg) {
            std::cerr << m.toString() << " ";
        }
        std::cerr << std::endl;
        BOOST_REQUIRE_EQUAL(msg[2].toString(),std::string(msg1));
        BOOST_REQUIRE_EQUAL(msg[3].toString(),std::string(msg2));
    };

        
    mySubscriber.connectToEndpoint("router/logger");
    mySubscriber.start();
    mySubscriber.subscribe("ERROR");
  
    // Necessary time for the subscriber to be ready to get messages 
    std::this_thread::sleep_for(std::chrono::seconds(1));
   
    router.analytics->logErrorMessage(msg1, std::vector<std::string>{msg2});

    std::this_thread::sleep_for(std::chrono::seconds(5));

    BOOST_REQUIRE(message_received);

}
