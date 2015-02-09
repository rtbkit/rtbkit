/* gumgum_exchange_connector.cc
   Anatoly Kochergin, 20 April 2013
   
   Implementation of Gumgum In-Image exchange connector.
*/

#include "gumgum_exchange_connector.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request_parser.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"


using namespace std;
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

namespace OpenRTB {

template<typename T>
Json::Value jsonEncode(const OpenRTB::List<T> & vec)
{
    using Datacratic::jsonEncode;
    Json::Value result(Json::arrayValue);
    for (unsigned i = 0;  i < vec.size();  ++i)
        result[i] = jsonEncode(vec[i]);
    return result;
}

template<typename T>
OpenRTB::List<T>
jsonDecode(const Json::Value & val, OpenRTB::List<T> *)
{
    using Datacratic::jsonDecode;
    ExcAssert(val.isArray());
    OpenRTB::List<T> res;
    res.reserve(val.size());
    for (unsigned i = 0;  i < val.size();  ++i)
        res.push_back(jsonDecode(val[i], (T*)0));
    return res;
}

} // namespace OpenRTB

namespace RTBKIT {

BOOST_STATIC_ASSERT(hasFromJson<Datacratic::Id>::value == true);
BOOST_STATIC_ASSERT(hasFromJson<int>::value == false);


/*****************************************************************************/
/* GUMGUM EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

GumgumExchangeConnector::
GumgumExchangeConnector(ServiceBase & owner, const std::string & name)
    : HttpExchangeConnector(name, owner)
{
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
}

GumgumExchangeConnector::
GumgumExchangeConnector(const std::string & name,
                        std::shared_ptr<ServiceProxies> proxies)
    : HttpExchangeConnector(name, proxies)
{
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
}

std::shared_ptr<BidRequest>
GumgumExchangeConnector::
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

    // Check it's version
    string openRtbVersion = it->second;
    float version = atof(openRtbVersion.c_str());
    cerr << "version" << version << endl;
    if (openRtbVersion != "2.0" && version < 2.0 ) {
        connection.sendErrorResponse("expected OpenRTB version 2.0 or higher; got " + openRtbVersion);
        return res;
    }

    cerr << "got request" << endl << header << endl << payload << endl;

    // Parse the bid request
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    res.reset(OpenRTBBidRequestParser::openRTBBidRequestParserFactory(openRtbVersion)->parseBidRequest(context, exchangeName(), exchangeName()));
        
    cerr << res->toJson() << endl;

    return res;
}

double
GumgumExchangeConnector::
getTimeAvailableMs(HttpAuctionHandler & connection,
                   const HttpHeader & header,
                   const std::string & payload)
{
    // Scan the payload quickly for the tmax parameter.
    static const string toFind = "\"tmax\":";
    string::size_type pos = payload.find(toFind);
    if (pos == string::npos) {
        cerr << "tmax not found in request, using default value" << endl;
        return 100.0;
		}
        
    int tmax = atoi(payload.c_str() + pos + toFind.length());
        
    return tmax;
}

HttpResponse
GumgumExchangeConnector::
getResponse(const HttpAuctionHandler & connection,
            const HttpHeader & requestHeader,
            const Auction & auction) const
{
    const Auction::Data * current = auction.getCurrentData();

    if (current->hasError())
        return getErrorResponse(connection, current->error + ": " + current->details);
    
    OpenRTB::BidResponse response;
    response.id = auction.id;

    std::map<Id, int> seatToBid;

    string en = exchangeName();

    // Create a spot for each of the bid responses
    for (unsigned spotNum = 0; spotNum < current->responses.size();
         ++spotNum) {
        if (!current->hasValidResponse(spotNum))
            continue;
        
        // Get the winning bid
        auto & resp = current->winningResponse(spotNum);

        // Find how the agent is configured.  We need to copy some of the
        // fields into the bid.
        const AgentConfig * config
            = std::static_pointer_cast<const AgentConfig>
            (resp.agentConfig).get();

        // Get the exchange specific data for this campaign (if it exists)
        bool cpinfo_ok = false;
        Id seat("gumgum_dummy_seat"); 
        try {
            auto cpinfo = config->getProviderData<CampaignInfo>(en);
            seat = cpinfo->seat;
            cpinfo_ok = true;
        } catch(ML::Exception& ex) {}

        // Put in the fixed parts from the creative
        int creativeIndex = resp.agentCreativeIndex;

        auto & creative = config->creatives.at(creativeIndex);

        // Get the exchange specific data for this creative (if it exists)
        bool crinfo_ok = false;
        Id adid; 
        std::string adm; 
        std::string nurl; 
        try {
            auto crinfo = creative.getProviderData<CreativeInfo>(en);
            adid = crinfo->adid;
            adm = crinfo->adm;
            nurl = crinfo->nurl;
            crinfo_ok = true;
        } catch(ML::Exception& ex) {}

        // Find the index in the seats array
        int seatIndex = -1;
        {
            auto it = seatToBid.find(seat);
            if (it == seatToBid.end()) {
                seatIndex = seatToBid.size();
                seatToBid[seat] = seatIndex;
                response.seatbid.emplace_back();
                if(cpinfo_ok) response.seatbid.back().seat = seat;
            }
            else seatIndex = it->second;
        }
        
        // Get the seatBid object
        OpenRTB::SeatBid & seatBid = response.seatbid.at(seatIndex);
        
        // Add a new bid to the array
        seatBid.bid.emplace_back();
        auto & b = seatBid.bid.back();
        
        // Put in the variable parts
        b.id = Id(auction.id, auction.request->imp[0].id);
        b.impid = auction.request->imp[spotNum].id;
        b.price.val = getAmountIn<CPM>(resp.price.maxPrice);
        if(crinfo_ok && adid != Id("")) b.adid = adid;
        if(crinfo_ok && !adm.empty()) b.adm = adm;
        if(crinfo_ok && !nurl.empty()) b.nurl = nurl;
    }

    if (seatToBid.empty())
        return HttpResponse(204, "none", "");

    static Datacratic::DefaultDescription<OpenRTB::BidResponse> desc;
    std::ostringstream stream;
    StreamJsonPrintingContext context(stream);
    desc.printJsonTyped(&response, context);

    cerr << Json::parse(stream.str());

    return HttpResponse(200, "application/json", stream.str());
}

HttpResponse
GumgumExchangeConnector::
getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                          const std::string & reason) const
{
    Json::Value response;
    response["error"] = reason;
    return HttpResponse(204, response);
}

HttpResponse
GumgumExchangeConnector::
getErrorResponse(const HttpAuctionHandler & connection,
                 const std::string & errorMessage) const
{
    Json::Value response;
    response["error"] = errorMessage;
    return HttpResponse(400, response);
}


} // namespace RTBKIT

namespace {
    using namespace RTBKIT;

    struct AtInit {
        AtInit() {
            ExchangeConnector::registerFactory<GumgumExchangeConnector>();
        }
    } atInit;
}

