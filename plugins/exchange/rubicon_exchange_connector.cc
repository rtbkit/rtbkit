/* rubicon_exchange_connector.cc
   Jeremy Barnes, 15 March 2013
   
   Implementation of the Rubicon exchange connector.
*/

#include "rubicon_exchange_connector.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include "jml/utils/file_functions.h"

#include "crypto++/blowfish.h"
#include "crypto++/modes.h"
#include "crypto++/filters.h"

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
/* RUBICON EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

RubiconExchangeConnector::
RubiconExchangeConnector(ServiceBase & owner, const std::string & name)
    : HttpExchangeConnector(name, owner)
{
}

RubiconExchangeConnector::
RubiconExchangeConnector(const std::string & name,
                         std::shared_ptr<ServiceProxies> proxies)
    : HttpExchangeConnector(name, proxies)
{
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
}

std::shared_ptr<BidRequest>
RubiconExchangeConnector::
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
    string openRtbVersion = it->second;
    if (openRtbVersion != "2.1") {
        connection.sendErrorResponse("expected OpenRTB version 2.1; got " + openRtbVersion);
        return res;
    }

    // Parse the bid request
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    res.reset(OpenRtbBidRequestParser::parseBidRequest(context, exchangeName(), exchangeName()));
        
    cerr << res->toJson() << endl;

    return res;
}

double
RubiconExchangeConnector::
getTimeAvailableMs(HttpAuctionHandler & connection,
                   const HttpHeader & header,
                   const std::string & payload)
{
    // Scan the payload quickly for the tmax parameter.
    static const string toFind = "\"tmax\":";
    string::size_type pos = payload.find(toFind);
    if (pos == string::npos)
        return 10.0;
        
    int tmax = atoi(payload.c_str() + pos + toFind.length());
        
    return tmax;
}

HttpResponse
RubiconExchangeConnector::
getResponse(const HttpAuctionHandler & connection,
            const HttpHeader & requestHeader,
            const Auction & auction) const
{
    const Auction::Data * current = auction.getCurrentData();

    if (current->hasError())
        return getErrorResponse(connection, auction, current->error + ": " + current->details);
    
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

        // Get the exchange specific data for this campaign
        auto cpinfo = config->getProviderData<CampaignInfo>(en);

        // Put in the fixed parts from the creative
        int creativeIndex = resp.agentCreativeIndex;

        auto & creative = config->creatives.at(creativeIndex);

        // Get the exchange specific data for this creative
        auto crinfo = creative.getProviderData<CreativeInfo>(en);
        
        // Find the index in the seats array
        int seatIndex = -1;
        {
            auto it = seatToBid.find(cpinfo->seat);
            if (it == seatToBid.end()) {
                seatIndex = seatToBid.size();
                seatToBid[cpinfo->seat] = seatIndex;
                response.seatbid.emplace_back();
                response.seatbid.back().seat = cpinfo->seat;
            }
            else seatIndex = it->second;
        }
        
        // Get the seatBid object
        OpenRTB::SeatBid & seatBid = response.seatbid.at(seatIndex);
        
        // Add a new bid to the array
        seatBid.bid.emplace_back();
        auto & b = seatBid.bid.back();
        
        // Put in the variable parts
        b.cid = Id(resp.agent);
        b.id = Id(auction.id, auction.request->imp[0].id);
        b.impid = auction.request->imp[spotNum].id;
        b.price.val = USD_CPM(resp.price.maxPrice);
        b.adm = crinfo->adm;
        b.adomain = crinfo->adomain;
        b.crid = crinfo->crid;
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
RubiconExchangeConnector::
getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                          const Auction & auction,
                          const std::string & reason) const
{
    return HttpResponse(204, "application/json", "{}");
}

HttpResponse
RubiconExchangeConnector::
getErrorResponse(const HttpAuctionHandler & connection,
                 const Auction & auction,
                 const std::string & errorMessage) const
{
    Json::Value response;
    response["error"] = errorMessage;
    return HttpResponse(400, response);
}

ExchangeConnector::ExchangeCompatibility
RubiconExchangeConnector::
getCampaignCompatibility(const AgentConfig & config,
                         bool includeReasons) const
{
    ExchangeCompatibility result;
    result.setCompatible();

    auto cpinfo = std::make_shared<CampaignInfo>();

    const Json::Value & pconf = config.providerConfig["rubicon"];

    try {
        cpinfo->seat = Id(pconf["seat"].asString());
        if (!cpinfo->seat)
            result.setIncompatible("providerConfig.rubicon.seat is null",
                                   includeReasons);
    } catch (const std::exception & exc) {
        result.setIncompatible
            (string("providerConfig.rubicon.seat parsing error: ")
             + exc.what(), includeReasons);
        return result;
    }
    
    result.info = cpinfo;
    
    return result;
}

namespace {

using Datacratic::jsonDecode;

/** Given a configuration field, convert it to the appropriate JSON */
template<typename T>
void getAttr(ExchangeConnector::ExchangeCompatibility & result,
             const Json::Value & config,
             const char * fieldName,
             T & field,
             bool includeReasons)
{
    try {
        if (!config.isMember(fieldName)) {
            result.setIncompatible
                ("creative[].providerConfig.rubicon." + string(fieldName)
                 + " must be specified", includeReasons);
            return;
        }
        
        const Json::Value & val = config[fieldName];
        
        jsonDecode(val, field);
    }
    catch (const std::exception & exc) {
        result.setIncompatible("creative[].providerConfig.rubicon."
                               + string(fieldName) + ": error parsing field: "
                               + exc.what(), includeReasons);
        return;
    }
}
    
} // file scope

ExchangeConnector::ExchangeCompatibility
RubiconExchangeConnector::
getCreativeCompatibility(const Creative & creative,
                         bool includeReasons) const
{
    ExchangeCompatibility result;
    result.setCompatible();

    auto crinfo = std::make_shared<CreativeInfo>();

    const Json::Value & pconf = creative.providerConfig["rubicon"];

    // 1.  Must have rubicon.attr containing creative attributes.  These
    //     turn into RubiconCreativeAttribute filters.
    getAttr(result, pconf, "attr", crinfo->attr, includeReasons);

    // TODO: create filter from these...

    // 2.  Must have rubicon.adm that includes Rubicon's macro
    getAttr(result, pconf, "adm", crinfo->adm, includeReasons);
    if (crinfo->adm.find("${AUCTION_PRICE:BF}") == string::npos)
        result.setIncompatible
            ("creative[].providerConfig.rubicon.adm ad markup must contain "
             "encrypted win price macro ${AUCTION_PRICE:BF}",
             includeReasons);
    
    // 3.  Must have creative ID in rubicon.crid
    getAttr(result, pconf, "crid", crinfo->crid, includeReasons);
    if (!crinfo->crid)
        result.setIncompatible
            ("creative[].providerConfig.rubicon.crid is null",
             includeReasons);
            
    // 4.  Must have advertiser names array in rubicon.adomain
    getAttr(result, pconf, "adomain", crinfo->adomain,  includeReasons);
    if (crinfo->adomain.empty())
        result.setIncompatible
            ("creative[].providerConfig.rubicon.adomain is empty",
             includeReasons);

    // Cache the information
    result.info = crinfo;

    return result;
}

float
RubiconExchangeConnector::
decodeWinPrice(const std::string & sharedSecret,
               const std::string & winPriceStr)
{
    ExcAssertEqual(winPriceStr.length(), 16);
        
    auto tox = [] (char c)
        {
            if (c >= '0' && c <= '9')
                return c - '0';
            else if (c >= 'A' && c <= 'F')
                return 10 + c - 'A';
            else if (c >= 'a' && c <= 'f')
                return 10 + c - 'a';
            throw ML::Exception("invalid hex digit");
        };

    unsigned char input[8];
    for (unsigned i = 0;  i < 8;  ++i)
        input[i]
            = tox(winPriceStr[i * 2]) * 16
            + tox(winPriceStr[i * 2 + 1]);
        
    CryptoPP::ECB_Mode<CryptoPP::Blowfish>::Decryption d;
    d.SetKey((byte *)sharedSecret.c_str(), sharedSecret.size());
    CryptoPP::StreamTransformationFilter
        filt(d, nullptr,
             CryptoPP::StreamTransformationFilter::NO_PADDING);
    filt.Put(input, 8);
    filt.MessageEnd();
    char recovered[9];
    size_t nrecovered = filt.Get((byte *)recovered, 8);

    ExcAssertEqual(nrecovered, 8);
    recovered[nrecovered] = 0;

    float res = boost::lexical_cast<float>(recovered);

    return res;
}

} // namespace RTBKIT
