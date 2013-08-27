/* openrtb_exchange_connector.cc
   Eric Robert, 7 May 2013
   
   Implementation of the OpenRTB exchange connector.
*/

#include "openrtb_exchange_connector.h"
#include "rtbkit/common/testing/exchange_source.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_source.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include "jml/utils/file_functions.h"
#include "jml/arch/info.h"
#include "jml/utils/rng.h"

using namespace Datacratic;

namespace RTBKIT {

BOOST_STATIC_ASSERT(hasFromJson<Datacratic::Id>::value == true);
BOOST_STATIC_ASSERT(hasFromJson<int>::value == false);

/*****************************************************************************/
/* OPENRTB EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

OpenRTBExchangeConnector::
OpenRTBExchangeConnector(ServiceBase & owner, const std::string & name)
    : HttpExchangeConnector(name, owner)
{
}

OpenRTBExchangeConnector::
OpenRTBExchangeConnector(const std::string & name,
                         std::shared_ptr<ServiceProxies> proxies)
    : HttpExchangeConnector(name, proxies)
{
}

std::shared_ptr<BidRequest>
OpenRTBExchangeConnector::
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

    // Check for the x-openrtb-version header
    auto it = header.headers.find("x-openrtb-version");
    if (it == header.headers.end()) {
        connection.sendErrorResponse("no OpenRTB version header supplied");
        return res;
    }

    // Check that it's version 2.1
    std::string openRtbVersion = it->second;
    if (openRtbVersion != "2.1") {
        connection.sendErrorResponse("expected OpenRTB version 2.1; got " + openRtbVersion);
        return res;
    }

    // Parse the bid request
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    res.reset(OpenRtbBidRequestParser::parseBidRequest(context,
                                                       exchangeName(),
                                                       exchangeName()));
    return res;
}

double
OpenRTBExchangeConnector::
getTimeAvailableMs(HttpAuctionHandler & connection,
                   const HttpHeader & header,
                   const std::string & payload)
{
    // Scan the payload quickly for the tmax parameter.
    static const std::string toFind = "\"tmax\":";
    std::string::size_type pos = payload.find(toFind);
    if (pos == std::string::npos)
        return 10.0;
    
    int tmax = atoi(payload.c_str() + pos + toFind.length());
    return tmax;
}

HttpResponse
OpenRTBExchangeConnector::
getResponse(const HttpAuctionHandler & connection,
            const HttpHeader & requestHeader,
            const Auction & auction) const
{
    const Auction::Data * current = auction.getCurrentData();

    if (current->hasError())
        return getErrorResponse(connection, auction,
                                current->error + ": " + current->details);

    OpenRTB::BidResponse response;
    response.id = auction.id;

    response.ext = getResponseExt(connection, auction);

    // Create a spot for each of the bid responses
    for (unsigned spotNum = 0; spotNum < current->responses.size(); ++spotNum) {
        if (!current->hasValidResponse(spotNum))
            continue;

        setSeatBid(auction, spotNum, response);
    }

    if (response.seatbid.empty())
        return HttpResponse(204, "none", "");

    static Datacratic::DefaultDescription<OpenRTB::BidResponse> desc;
    std::ostringstream stream;
    StreamJsonPrintingContext context(stream);
    desc.printJsonTyped(&response, context);

    return HttpResponse(200, "application/json", stream.str());
}

Json::Value
OpenRTBExchangeConnector::
getResponseExt(const HttpAuctionHandler & connection,
               const Auction & auction) const
{
    return {};
}

HttpResponse
OpenRTBExchangeConnector::
getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                          const std::string & reason) const
{
    return HttpResponse(204, "application/json", "{}");
}

HttpResponse
OpenRTBExchangeConnector::
getErrorResponse(const HttpAuctionHandler & connection,
                 const Auction & auction,
                 const std::string & errorMessage) const
{
    Json::Value response;
    response["error"] = errorMessage;
    return HttpResponse(400, response);
}

std::string
OpenRTBExchangeConnector::
getBidSourceConfiguration() const
{
    auto suffix = std::to_string(port());
    return ML::format("{\"type\":\"openrtb\",\"url\":\"%s\"}",
                      ML::fqdn_hostname(suffix) + ":" + suffix);
}

void
OpenRTBExchangeConnector::
setSeatBid(Auction const & auction,
           int spotNum,
           OpenRTB::BidResponse & response) const
{
    const Auction::Data * data = auction.getCurrentData();

    // Get the winning bid
    auto & resp = data->winningResponse(spotNum);

    int seatIndex = 0;
    if(response.seatbid.empty()) {
        response.seatbid.emplace_back();
    }

    OpenRTB::SeatBid & seatBid = response.seatbid.at(seatIndex);

    // Add a new bid to the array
    seatBid.bid.emplace_back();

    // Put in the variable parts
    auto & b = seatBid.bid.back();
    b.cid = Id(resp.agent);
    b.id = Id(auction.id, auction.request->imp[0].id);
    b.impid = auction.request->imp[spotNum].id;
    b.price.val = USD_CPM(resp.price.maxPrice);
}

} // namespace RTBKIT

namespace {
    using namespace RTBKIT;

    struct Init {
        Init() {
            ExchangeConnector::registerFactory<OpenRTBExchangeConnector>();
        }
    } init;
}

