/* standard_adserver_connector_test.cc

   Exchange connector test for BidSwitch.
   Based on rubicon_exchange_connector_test.cc
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

//#include "rtbkit/common/testing/exchange_source.h"
#include "rtbkit/plugins/adserver/standard_event_source.h"
#include "rtbkit/plugins/adserver/standard_win_source.h"
#include "rtbkit/plugins/adserver/standard_adserver_connector.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"

#include "jml/arch/info.h"

#include <type_traits>


using namespace RTBKIT;

const std::string win_sample_filename("rtbkit/plugins/adserver/testing/standard_adserver_connector_win.json");
const std::string click_sample_filename("rtbkit/plugins/adserver/testing/standard_adserver_connector_click.json");
const std::string conversion_sample_filename("rtbkit/plugins/adserver/testing/standard_adserver_connector_conversion.json");
const std::string statusOK("HTTP/1.1 200 OK");

std::string loadFile(const std::string & filename)
{
    ML::filter_istream stream(filename);

    std::string result;

    while (stream) {
        std::string line;
        getline(stream, line);
        result += line + "\n";
    }

    return result;
}

struct TestStandardAdServer {

    TestStandardAdServer() : addressWin(18143),
                             addressEvent(18144)
                             { 
        BOOST_TEST_MESSAGE ( "Setting up fixture." ); 
        proxies = std::make_shared<ServiceProxies>();
        connector = std::make_shared<StandardAdServerConnector>
            (proxies, "connector");
    
        connector->init(18143,18144);
        connector->start();

        winSource.reset(new StandardWinSource(addressWin));
        eventSource.reset(new StandardEventSource(addressEvent));
    }

    ~TestStandardAdServer() { 
        BOOST_TEST_MESSAGE ( "Test Suite done. Cleaning up fixture.");
        winSource.reset();
        eventSource.reset();
        connector->shutdown();
        BOOST_TEST_MESSAGE ( "Fixture cleaned up. Exiting."); 
    }

    // Create our exchange connector and configure it to listen on port
    // 18143 for wins and 18144 for events.  Note that we need to ensure 
    // that port 18143 and 18144 is open on our firewall.
    
    std::shared_ptr<ServiceProxies> proxies;
    std::shared_ptr<StandardAdServerConnector> connector;

    NetworkAddress addressWin, addressEvent;
    std::thread winSourceThread;

    std::shared_ptr<WinSource> winSource;
    std::shared_ptr<EventSource> eventSource;
};

BOOST_FIXTURE_TEST_SUITE( standard_adserver_tests, TestStandardAdServer )

BOOST_AUTO_TEST_CASE( test_standard_adserver_win )
{
    // load win json
    std::string strJson = loadFile(win_sample_filename);
    std::cerr << strJson << std::endl;

    std::string httpRequest = ML::format(
                                  "POST / HTTP/1.1\r\n"
                                  "Content-Length: %zd\r\n"
                                  "Content-Type: application/json\r\n"
                                  "\r\n"
                                  "%s",
                                  strJson.size(),
                                  strJson.c_str());

    // and send it
    winSource->write(httpRequest);
    std::string result = winSource->read();

    BOOST_CHECK_EQUAL(result.compare(0, statusOK.length(), statusOK), 0);

    proxies->events->dump(std::cerr);
}

BOOST_AUTO_TEST_CASE( test_standard_adserver_click )
{
    // load click json
    std::string strJson = loadFile(click_sample_filename);
    std::cerr << strJson << std::endl;

    std::string httpRequest = ML::format(
                                  "POST / HTTP/1.1\r\n"
                                  "Content-Length: %zd\r\n"
                                  "Content-Type: application/json\r\n"
                                  "\r\n"
                                  "%s",
                                  strJson.size(),
                                  strJson.c_str());

    // and send it
    eventSource->write(httpRequest);
    std::string result = eventSource->read();

    BOOST_CHECK_EQUAL(result.compare(0, statusOK.length(), statusOK), 0);

    std::cerr << result << std::endl;

    proxies->events->dump(std::cerr);

}

BOOST_AUTO_TEST_CASE( test_standard_adserver_conversion )
{
    // load click json
    std::string strJson = loadFile(conversion_sample_filename);
    std::cerr << strJson << std::endl;

    std::string httpRequest = ML::format(
                                  "POST / HTTP/1.1\r\n"
                                  "Content-Length: %zd\r\n"
                                  "Content-Type: application/json\r\n"
                                  "\r\n"
                                  "%s",
                                  strJson.size(),
                                  strJson.c_str());

    // and send it
    eventSource->write(httpRequest);
    std::string result = eventSource->read();

    BOOST_CHECK_EQUAL(result.compare(0, statusOK.length(), statusOK), 0);

    std::cerr << result << std::endl;

    proxies->events->dump(std::cerr);

}

BOOST_AUTO_TEST_SUITE_END()
