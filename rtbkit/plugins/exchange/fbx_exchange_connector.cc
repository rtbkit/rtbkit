/* fbx_exchange_connector.cc
   Jean-Sebastien Bejeau, 27 June 2013
   
   Implementation of the FBX exchange connector.
*/

#include "fbx_exchange_connector.h"
#include "rtbkit/plugins/bid_request/fbx_parsing.h"
#include "rtbkit/plugins/bid_request/fbx_bid_request.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"

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
namespace FBX {
/*
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
*/
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

    // Parse the bid request
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    res.reset(FbxBidRequestParser::parseBidRequest(context,
                                                   exchangeName(),
                                                   exchangeName()));
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
        return getErrorResponse(connection, 
                                current->error + ": " + current->details);

    FBX::BidResponse response;
    response.requestId = auction.id;

    // Create a spot for each of the bid responses
    for (unsigned spotNum = 0; spotNum < current->responses.size(); ++spotNum) {
        if (!current->hasValidResponse(spotNum))
            continue;

        const Auction::Data * data = auction.getCurrentData();

        auto & resp = data->winningResponse(spotNum);

        response.bids.emplace_back();
        auto & bid = response.bids.back();

        bid.adId = Id(resp.creativeId);
        // will be casted in int, according to the comment (for bidNative), this
        // is expected
        bid.bidNative.val = double(getAmountIn<CPM>(resp.price.maxPrice));
    }

    static Datacratic::DefaultDescription<FBX::BidResponse> desc;
    std::ostringstream stream;
    StreamJsonPrintingContext context(stream);
    desc.printJsonTyped(&response, context);

    return HttpResponse(200, "application/json", stream.str());
}

} // namespace RTBKIT

namespace {
    using namespace RTBKIT;

    struct AtInit {
        AtInit() {
            ExchangeConnector::registerFactory<FBXExchangeConnector>();
        }
    } atInit;
}

