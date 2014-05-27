/* exchange_parsing_from_file_test.cc
   Jean-Sebastien Bejeau, 27 May 2014
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Allow to test batch of Bid Request parsing from a file.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "rtbkit/plugins/bid_request/openrtb_bid_request.h"
#include "rtbkit/plugins/exchange/http_exchange_connector.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "jml/utils/filter_streams.h"
#include "rtbkit/core/router/router.h"
#include "rtbkit/common/testing/exchange_source.h"

#include "jml/utils/file_functions.h"

using namespace std;
using namespace Datacratic;
using namespace RTBKIT;

std::string configurationFile =  "./rtbkit/testing/exchange_parsing_from_file_config.json";

static Json::Value loadJsonFromFile(const std::string & filename) {
    ML::File_Read_Buffer buf(filename);
    return Json::parse(std::string(buf.start(), buf.end()));
}

BOOST_AUTO_TEST_CASE( test_exchange_parsing_multi_requests )
{

    Json::Value config = loadJsonFromFile(configurationFile); 
    
    std::shared_ptr<ServiceProxies> proxies(new ServiceProxies());

    // We need a router for our exchange connector to work
    Router router(proxies, "router");
    router.unsafeDisableMonitor();  // Don't require a monitor service
    router.init();

    // Start the router up
    router.bindTcp();
    router.start();

    // Start exchange
    const std::string type = config["exchangeType"].asString();
    auto exchange = ExchangeConnector::create(type, router , type); 

    std::shared_ptr<ExchangeConnector> connector(exchange.release());
    std::shared_ptr<HttpExchangeConnector> connector1 = dynamic_pointer_cast<HttpExchangeConnector>(connector);

    std::cerr << "Loading " << connector1->exchangeName() << " exchange connector." << std::endl;

    connector1->configureHttp(1, -1, "0.0.0.0");
    connector1->start();
    connector1->enableUntil(Date::positiveInfinity());
    
    // Tell the router about the new exchange connector
    router.addExchange(connector1);
    ML::sleep(1.0);

    // prepare request
    NetworkAddress address(connector1->port());
    BidSource source(address);


    for (auto sample : config["samples"])
    {
        std::string req = sample.asString();
        vector<string> reqs;
        
        ML::filter_istream stream(req);
    
        while (stream) {
            string line;
            getline(stream, line);
            reqs.push_back(line);
        }

        std::stringstream ss;
        std::stringstream ssError;
        ss << "---------------------------------------------------" << endl;
        ss << "Summary of parsing" << endl;
        ss << "Number of bid request : " << reqs.size()-1 << endl;
        int rejected = 0;
        std::string response;

        for (unsigned i = 0;  i < reqs.size()-1;  ++i) {
            try {

                std::string utf8String = reqs[i]; 

                std::ostringstream stream;
                stream << "POST /auctions HTTP/1.1\r\n"
                       << "Content-Length: " << utf8String.size() << "\r\n"
                       << "Content-Type: application/json\r\n"
                       << "x-openrtb-version: 2.1\r\n"
                       << "x-openrtb-verbose: 1\r\n"
                       << "\r\n"
                       << utf8String;

                source.write(stream.str());
                response = source.read();
                
                HttpHeader http;
                if(response.find("Content-Type: none") == std::string::npos) {
                    http.parse(response);
                
                    std::string error;
                    if(http.contentType.find("Content-Type: none") == std::string::npos) {
                        auto json = Json::parse(http.knownData);
                        error = json["error"].asString();
                        rejected++;        
                        ssError << "---------------------------------------------------" << endl;
                        ssError << "At line : " << i+1 << endl;
                        ssError << "Error: " << error << endl;
                        ssError << "Bid_request : " << reqs[i] << endl;
                    }
                }
            }
            catch (const std::exception & exc) {
            }
        }

        ss << "Number of error during parsing : " << rejected << endl;
        if(rejected > 0) {
            ss << "List of errors :\n" << ssError.str() << endl;
        }
        ss << "---------------------------------------------------" << endl;
        std::cerr << ss.str() << endl;

        router.shutdown();

        ML::sleep(1.0);

        BOOST_CHECK_EQUAL(rejected, 0); 

    }
}
