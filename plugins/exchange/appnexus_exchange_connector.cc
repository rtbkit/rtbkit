/* appnexus_exchange_connector.cc
   Eric Robert, 23 July 2013

   Implementation of the AppNexus exchange connector.
*/

#include <iostream>
#include <boost/range/irange.hpp>

#include "appnexus_exchange_connector.h"
#include "rtbkit/plugins/bid_request/appnexus_bid_request.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
/*
#include "rtbkit/common/testing/exchange_source.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include "jml/utils/file_functions.h"
#include "jml/arch/info.h"
#include "jml/utils/rng.h"
*/

using namespace std ;
using namespace Datacratic;
/*
namespace Datacratic {

template<typename T, int I, typename S>
Json::Value jsonEncode(const ML::compact_vector<T, I, S> & vec)
{
    Json::Value result(Json::arrayValue);
    for (unsigned i = 0;  i < vec.size();  ++i)
        result[i] = jsonEncode(vec[i]);
    return result;
}

template<typename T, int I, typename S>
ML::compact_vector<T, I, S>
jsonDecode(const Json::Value & val, ML::compact_vector<T, I, S> *)
{
    ExcAssert(val.isArray());
    ML::compact_vector<T, I, S> res;
    res.reserve(val.size());
    for (unsigned i = 0;  i < val.size();  ++i)
        res.push_back(jsonDecode(val[i], (T*)0));
    return res;
}

} // namespace Datacratic
*/

namespace RTBKIT {

//BOOST_STATIC_ASSERT(hasFromJson<Datacratic::Id>::value == true);
//BOOST_STATIC_ASSERT(hasFromJson<int>::value == false);

/*****************************************************************************/
/* OPENRTB EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

AppNexusExchangeConnector::
AppNexusExchangeConnector(ServiceBase & owner, const std::string & name)
    : HttpExchangeConnector(name, owner)
{
}

AppNexusExchangeConnector::
AppNexusExchangeConnector(const std::string & name,
                          std::shared_ptr<ServiceProxies> proxies)
    : HttpExchangeConnector(name, proxies)
{
}

std::shared_ptr<BidRequest>
AppNexusExchangeConnector::
parseBidRequest(HttpAuctionHandler & connection,
                const HttpHeader & header,
                const std::string & payload)
{
    // doest not set a content type.
#if 1
    {
        std::cerr << "**GOT:\n" << Json::parse(payload).toString() << std::endl;
    }
#endif
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    auto rc = AppNexusBidRequestParser::parseBidRequest(context,
              exchangeNameString(), exchangeNameString()) ;

    if (!rc)
        connection.sendErrorResponse("appnexus connector: bad JSON fed");
#if 1
    else
        std::cerr << "\nAPPNEXUS: " << payload
                  << "  RTBkit: " << rc->toJsonStr()
                  << std::endl;
#endif
    return rc;
}

double
AppNexusExchangeConnector::
getTimeAvailableMs(HttpAuctionHandler & connection,
                   const HttpHeader & header,
                   const std::string & payload)
{
    // Scan the payload quickly for the tmax parameter.
    static const std::string toFind = "\"bidder_timeout_ms\":";
    std::string::size_type pos = payload.find(toFind);
    if (pos == std::string::npos)
        return 100.0;

    int tmax = atoi(payload.c_str() + pos + toFind.length());
    return tmax;
}

HttpResponse
AppNexusExchangeConnector::
getResponse(const HttpAuctionHandler & connection,
            const HttpHeader & requestHeader,
            const Auction & auction) const
{
    const Auction::Data * current = auction.getCurrentData();

    cerr << "XXXX = " << auction.id.toString() << "XXX\n";
    for (size_t i =0; i < auction.getResponses().size(); i++)
    {
    	cerr << "RESPONSE: --- " << auction.getResponseJson(i).toString() << endl ;
    }

    if (current->hasError())
        return getErrorResponse(connection, auction,
                                current->error + ": " + current->details);


    Json::Value response ;
    response["no_bid"] = true;
    response["auction_id_64"] = auction.id.toInt();
    Json::Value responses (Json::arrayValue);
    responses.append(response);
    Json::Value bid_response ;
    bid_response["responses"] = responses ;
    Json::Value retval;
    retval["bid_response"] = bid_response;

//    GoogleBidResponse gresp ;
//    gresp.set_processing_time_ms(static_cast<uint32_t>(auction.timeUsed()*1000));

    auto en = exchangeName();

    // Create a spot for each of the bid responses
    for (auto spotNum: boost::irange(0UL, current->responses.size()))
    {

        if (!current->hasValidResponse(spotNum))
            continue ;
        // Get the winning bid
        auto & resp = current->winningResponse(spotNum);

        // Find how the agent is configured.  We need to copy some of the
        // fields into the bid.
        const AgentConfig * config  =
            std::static_pointer_cast<const AgentConfig>(resp.agentConfig).get();

        // Put in the fixed parts from the creative
        int creativeIndex = resp.agentCreativeIndex;

        auto & creative = config->creatives.at(creativeIndex);

        cerr << "spotnum=" << spotNum << ": " << creative.id << endl ;


    }
    /*
    AppNexus::BidResponse response;
    response.id = auction.id;

    // Create a spot for each of the bid responses
    for (unsigned spotNum = 0; spotNum < current->responses.size(); ++spotNum) {
        if (!current->hasValidResponse(spotNum))
            continue;

        setBid(auction, spotNum, response);
    }

    if (response.seatbid.empty())
        return HttpResponse(204, "none", "");

    static Datacratic::DefaultDescription<AppNexus::BidResponse> desc;
    std::ostringstream stream;
    StreamJsonPrintingContext context(stream);
    desc.printJsonTyped(&response, context);
    */

    return HttpResponse(200, "application/json", retval.toString());
}

HttpResponse
AppNexusExchangeConnector::
getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                          const std::string & reason) const
{
    return HttpResponse(204, "application/json", "{}");
}

HttpResponse
AppNexusExchangeConnector::
getErrorResponse(const HttpAuctionHandler & connection,
                 const Auction & auction,
                 const std::string & errorMessage) const
{
    Json::Value response;
    response["error"] = errorMessage;
    return HttpResponse(400, response);
}

} // namespace RTBKIT

namespace {
using namespace RTBKIT;

struct Init {
    Init() {
        ExchangeConnector::registerFactory<AppNexusExchangeConnector>();
    }
} init;
}

