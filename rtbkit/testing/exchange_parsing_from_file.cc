/* exchange_parsing_from_file.ccc
   Jean-Sebastien Bejeau, 27 May 2014
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Allow to test batch of Bid Request parsing from a file.
*/


#include "rtbkit/testing/exchange_parsing_from_file.h"

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

static Json::Value loadJsonFromFile(const std::string & filename) {
    ML::File_Read_Buffer buf(filename);
    return Json::parse(std::string(buf.start(), buf.end()));
}

Exchange_parsing_from_file ::
Exchange_parsing_from_file(const std::string config) : 
    configurationFile(config), 
    error(0)
{
    std::ostringstream stream;

    // Set default Header
    stream << "Content-Type: application/json\r\n"
           << "x-openrtb-version: 2.1\r\n"
           << "x-openrtb-verbose: 1\r\n";

    header = stream.str();
}


void
Exchange_parsing_from_file ::
run() 
{
    Json::Value config = loadJsonFromFile(configurationFile); 

    for ( auto currentConfig : config) {
        std::cerr << "Current Exchange :" <<  currentConfig["exchangeType"] << std::endl;

        std::shared_ptr<ServiceProxies> proxies(new ServiceProxies());

        // We need a router for our exchange connector to work
        Router router(proxies, "router");
        router.unsafeDisableMonitor();  // Don't require a monitor service
        router.unsafeDisableAuctionProbability(); // Disable auction prob to avoid dropping BR
        router.init();

        // Start the router up
        router.bindTcp();
        router.start();

        // Start exchange
        const std::string type = currentConfig["exchangeType"].asString();
        auto exchange = ExchangeConnector::create(type, router , type);

        std::shared_ptr<ExchangeConnector> connector(exchange.release());
        std::shared_ptr<HttpExchangeConnector> connector1 = dynamic_pointer_cast<HttpExchangeConnector>(connector);

        std::cerr << "Loading " << connector1->exchangeName() << " exchange connector." << std::endl;

        connector1->configureHttp(1, -1, "0.0.0.0");
        connector1->start();
        connector1->enableUntil(Date::positiveInfinity());

        // Tell the router about the new exchange connector
        router.addExchange(connector1);
        router.initFilters();
        ML::sleep(1.0);

        // prepare request
        NetworkAddress address(connector1->port());
        BidSource source(address);


        for (auto sample : currentConfig["samples"])
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
                           << header
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

            auto numError = rejected - currentConfig["expectedRejected"].asInt();

            if( numError != 0 ) {
                error += numError;
            
            }
        }
    }




}


int
Exchange_parsing_from_file ::
getNumError() 
{
    return error;
}

void
Exchange_parsing_from_file ::
resetNumError()
{
    error = 0;
}

