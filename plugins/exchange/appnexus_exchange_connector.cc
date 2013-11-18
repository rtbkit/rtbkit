/* appnexus_exchange_connector.cc
   Eric Robert, 23 July 2013

   Implementation of the AppNexus exchange connector.
*/

#include <iostream>
#include <boost/range/irange.hpp>
#include <boost/tokenizer.hpp>

#include "appnexus_exchange_connector.h"
#include "rtbkit/plugins/bid_request/appnexus_bid_request.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"

using namespace std ;
using namespace Datacratic;

namespace RTBKIT {

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
        std::cerr << "*** THEIRS :\n" << Json::parse(payload).toString() << std::endl;
    }
#endif
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    auto rc = AppNexusBidRequestParser::parseBidRequest(context,
              exchangeNameString(), exchangeNameString()) ;

    if (!rc)
        connection.sendErrorResponse("appnexus connector: bad JSON fed");
#if 1
    else
        std::cerr << "***   OURS :\n" << rc->toJsonStr() << std::endl;
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

	if (current->hasError())
		return getErrorResponse(connection, auction,
				current->error + ": " + current->details);

	Json::Value responses (Json::arrayValue);

	auto en = exchangeName();

	// Create a spot for each of the bid responses
	for (auto spotNum: boost::irange(0UL, current->responses.size()))
	{

		if (!current->hasValidResponse(spotNum))
			continue ;

		Json::Value response ;
		response["no_bid"] = false;
		response["auction_id_64"] = auction.id.toInt();

		// Get the winning bid
		auto & resp = current->winningResponse(spotNum);
		// TODO:
	    // figure out what to do w.r.t. members etc.
		//
		response["member_id"] = 2187;
		response["price"] = resp.price.maxPrice.value;
		response["creative_id"] = resp.creativeId;
		response["creative_code"] = resp.creativeName;

		responses.append(response);
	}
	// Is there a nicer way to do this?
	Json::Value bid_response ;
	bid_response["responses"] = responses ;
	Json::Value retval;
	retval["bid_response"] = bid_response;
#if 7
	cerr << " ---> BID_RESPONSE=" << retval.toString() << endl ;
#endif
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

using namespace boost;

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
            ("creative[].providerConfig.appnexus." + string(fieldName)
             + " must be specified", includeReasons);
            return;
        }

        const Json::Value & val = config[fieldName];

        jsonDecode(val, field);
    }
    catch (const std::exception & exc) {
        result.setIncompatible("creative[].providerConfig.appnexus."
                               + string(fieldName) + ": error parsing field: "
                               + exc.what(), includeReasons);
        return;
    }
}
} // file scope

ExchangeConnector::ExchangeCompatibility
AppNexusExchangeConnector::
getCreativeCompatibility(const Creative & creative,
                         bool includeReasons) const
{
    ExchangeCompatibility result;
    result.setCompatible();

    auto crinfo = std::make_shared<CreativeInfo>();

    if (!creative.providerConfig.isMember("appnexus")) {
        result.setIncompatible();
        return result;
    }

    const Json::Value & pconf = creative.providerConfig["appnexus"];

    // 1.  Must have appnexus.externalId containing creative attributes.
    getAttr(result, pconf, "externalId", crinfo->buyer_creative_id_, includeReasons);

    // 2.  Must have appnexus.htmlTemplate that includes AdX's macro
    getAttr(result, pconf, "htmlTemplate", crinfo->html_snippet_, includeReasons);
    if (crinfo->html_snippet_.find("%%WINNING_PRICE%%") == string::npos)
        result.setIncompatible
        ("creative[].providerConfig.appnexus.html_snippet must contain "
         "encrypted win price macro %%WINNING_PRICE%%",
         includeReasons);

    // 3.  Must have appnexus.clickThroughUrl
    getAttr(result, pconf, "clickThroughUrl", crinfo->click_through_url_, includeReasons);

    // 4.  Must have appnexus.agencyId
    //     according to the .proto file this could also be set
    //     to 1 if nothing has been provided in the providerConfig
    getAttr(result, pconf, "agencyId", crinfo->agency_id_, includeReasons);
    if (!crinfo->agency_id_) crinfo->agency_id_ = 1;

    string tmp;
    const auto to_int = [] (const string& str) {
        return atoi(str.c_str());
    };

    // 5.  Must have vendorType
    getAttr(result, pconf, "vendorType", tmp, includeReasons);
    if (!tmp.empty())
    {
        tokenizer<> tok(tmp);
        auto& ints = crinfo->vendor_type_;
        transform(tok.begin(), tok.end(),
        std::inserter(ints, ints.begin()),[&](const std::string& s) {
            return atoi(s.data());
        });
    }

    tmp.clear();
    // 6.  Must have attribute
    getAttr(result, pconf, "attribute", tmp, includeReasons);
    if (!tmp.empty())
    {
        tokenizer<> tok(tmp);
        auto& ints = crinfo->attribute_;
        transform(tok.begin(), tok.end(),
        std::inserter(ints, ints.begin()),[&](const std::string& s) {
            return atoi(s.data());
        });
    }

    tmp.clear();
    // 7.  Must have sensitiveCategory
    getAttr(result, pconf, "sensitiveCategory", tmp, includeReasons);
    if (!tmp.empty())
    {
        tokenizer<> tok(tmp);
        auto& ints = crinfo->category_;
        transform(tok.begin(), tok.end(),
        std::inserter(ints, ints.begin()),[&](const std::string& s) {
            return atoi(s.data());
        });
    }

    if (result.isCompatible) {
        // Cache the information
        result.info = crinfo;
    }

    return result;
}

bool
AppNexusExchangeConnector::
bidRequestCreativeFilter(const BidRequest & request,
                         const AgentConfig & config,
                         const void * info) const
{
    const auto crinfo = reinterpret_cast<const CreativeInfo*>(info);

    // This function is called once per BidRequest.
    // However a bid request can return multiple AdSlot.
    // The creative restrictions do apply per AdSlot.
    // We then check that *all* the AdSlot present in this BidRequest
    // do pass the filter.
    // TODO: verify performances of the implementation.
    for (const auto& spot: request.imp)
    {

        const auto& excluded_attribute_seg = spot.restrictions.get("excluded_attribute");
        for (auto atr: crinfo->attribute_)
            if (excluded_attribute_seg.contains(atr))
            {
                this->recordHit ("attribute_excluded");
                return false ;
            }

        const auto& excluded_sensitive_category_seg =
            spot.restrictions.get("excluded_sensitive_category");
        for (auto atr: crinfo->category_)
            if (excluded_sensitive_category_seg.contains(atr))
            {
                this->recordHit ("sensitive_category_excluded");
                return false ;
            }

        const auto& allowed_vendor_type_seg =
            spot.restrictions.get("allowed_vendor_type");
        for (auto atr: crinfo->vendor_type_)
            if (!allowed_vendor_type_seg.contains(atr))
            {
                this->recordHit ("vendor_type_not_allowed");
                return false ;
            }
    }
    return true;
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

