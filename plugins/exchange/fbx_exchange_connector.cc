/* fbx_exchange_connector.cc
   Jean-Sebastien Bejeau, 27 June 2013
   
   Implementation of the FBX exchange connector.
*/

#include "fbx_exchange_connector.h"
//#include "rtbkit/common/testing/exchange_source.h"
#include "rtbkit/plugins/bid_request/fbx_bid_request.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
//#include "rtbkit/core/agent_configuration/agent_config.h"
#include "fbx/fbx_parsing.h"
/*#include "soa/types/json_printing.h"
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include "jml/utils/file_functions.h"
#include "jml/arch/info.h"
#include "jml/utils/rng.h"*/

using namespace Datacratic;

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

namespace FBX {

template<typename T>
Json::Value jsonEncode(const FBX::List<T> & vec)
{
    using Datacratic::jsonEncode;
    Json::Value result(Json::arrayValue);
    for (unsigned i = 0;  i < vec.size();  ++i)
        result[i] = jsonEncode(vec[i]);
    return result;
}

template<typename T>
FBX::List<T>
jsonDecode(const Json::Value & val, FBX::List<T> *)
{
    using Datacratic::jsonDecode;
    ExcAssert(val.isArray());
    FBX::List<T> res;
    res.reserve(val.size());
    for (unsigned i = 0;  i < val.size();  ++i)
        res.push_back(jsonDecode(val[i], (T*)0));
    return res;
}

} // namespace FBX

namespace RTBKIT {

BOOST_STATIC_ASSERT(hasFromJson<Datacratic::Id>::value == true);
BOOST_STATIC_ASSERT(hasFromJson<int>::value == false);

/*****************************************************************************/
/* FBX EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

FBXExchangeConnector::
FBXExchangeConnector(ServiceBase & owner, const std::string & name) :
	HttpExchangeConnector(name, owner) {
		//this->auctionResource = "/bids";
		//this->auctionVerb = "POST";string
}

FBXExchangeConnector::
FBXExchangeConnector(const std::string & name,
                         std::shared_ptr<ServiceProxies> proxies)
    : HttpExchangeConnector(name, proxies)
{
}

std::shared_ptr<BidRequest>
FBXExchangeConnector::
parseBidRequest(HttpAuctionHandler & connection,
                const HttpHeader & header,
                const std::string & payload)
{
    std::shared_ptr<BidRequest> res;

    // Check for JSON content-type
    if (header.contentType != "application/json") {
        connection.sendErrorResponse("non-JSON request");
        return res;
    }

    std::cerr << "got request" << std::endl << header << std::endl;


    // Parse the bid request
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    res.reset(FbxBidRequestParser::parseBidRequest(context,
                                                       exchangeName(),
                                                       exchangeName()));
        
    std::cerr << res->toJson() << std::endl;

    return res;
}


HttpResponse
FBXExchangeConnector::
getResponse(const HttpAuctionHandler & connection,
            const HttpHeader & requestHeader,
            const Auction & auction) const
{
    const Auction::Data * current = auction.getCurrentData();

    if (current->hasError())
        return getErrorResponse(connection, auction,
                                current->error + ": " + current->details);

    FBX::BidResponse response;
    response.requestId = auction.id;
/*
    // Create a spot for each of the bid responses
    for (unsigned spotNum = 0; spotNum < current->responses.size(); ++spotNum) {
        if (!current->hasValidResponse(spotNum))
            continue;

        setSeatBid(auction, spotNum, response);
    }

    if (response.seatbid.empty())
        return HttpResponse(204, "none", "");
*/
    static Datacratic::DefaultDescription<FBX::BidResponse> desc;
    std::ostringstream stream;
    StreamJsonPrintingContext context(stream);
    desc.printJsonTyped(&response, context);

    std::cerr << Json::parse(stream.str());

    return HttpResponse(200, "application/json", stream.str());
}

} // namespace RTBKIT
