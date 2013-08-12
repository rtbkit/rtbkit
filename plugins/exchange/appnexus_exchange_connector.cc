/* appnexus_exchange_connector.cc
   Eric Robert, 23 July 2013
   
   Implementation of the AppNexus exchange connector.
*/

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
    std::shared_ptr<BidRequest> result;

    // Check for JSON content-type
    if (header.contentType != "application/json") {
        connection.sendErrorResponse("non-JSON request");
        return result;
    }

    auto name = exchangeNameString(); 

    // Parse the bid request
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    result.reset(AppNexusBidRequestParser::parseBidRequest(context, name, name));
    return result;
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

    if (current->hasError())
        return getErrorResponse(connection, auction,
                                current->error + ": " + current->details);

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

    return HttpResponse(200, "application/json", "stream.str()");
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

