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
#if 0
    {
        std::cerr << "*** THEIRS :\n" << Json::parse(payload).toString() << std::endl;
    }
#endif
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    auto rc = AppNexusBidRequestParser::parseBidRequest(context,
              exchangeNameString(), exchangeNameString()) ;

    if (!rc)
        connection.sendErrorResponse("appnexus connector: bad JSON fed");
#if 0
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
		return getErrorResponse(connection, 
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
#if 0
	cerr << " ---> BID_RESPONSE=" << retval.toString() << endl ;
#endif
	return HttpResponse(200, "application/json", retval.toString());
}

HttpResponse
AppNexusExchangeConnector::
getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                          const std::string & reason) const
{
    return HttpResponse(204, "none", "");
}

HttpResponse
AppNexusExchangeConnector::
getErrorResponse(const HttpAuctionHandler & connection,
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
		const Json::Value & config, const char * fieldName, T & field,
		bool includeReasons)
{
	try {
		if (!config.isMember(fieldName)) {
			result.setIncompatible(
					"creative[].providerConfig['appnexus']." + string(fieldName)
							+ " must be specified", includeReasons);
			return;
		}

		const Json::Value & val = config[fieldName];

		jsonDecode(val, field);
	} catch (const std::exception & exc) {
		result.setIncompatible(
				"creative[].providerConfig['appnexus']." + string(fieldName)
						+ ": error parsing field: " + exc.what(),
				includeReasons);
		return;
	}
}
} // file scope

ExchangeConnector::ExchangeCompatibility
AppNexusExchangeConnector::getCreativeCompatibility(
		const Creative & creative, bool includeReasons) const
{

	ExchangeCompatibility result;
	result.setCompatible();

	if (!creative.providerConfig.isMember("appnexus")) {
		result.setIncompatible("creative[].providerConfig['appnexus'] missing",
				includeReasons);
		return result;;
	}

	const Json::Value & pconf = creative.providerConfig["appnexus"];
	auto crinfo = std::make_shared<CreativeInfo>();
    string tmp;

	// 1. must have a member id.
	getAttr(result, pconf, "memberId", crinfo->member_id_, includeReasons);
	if (!result.isCompatible)
		goto out;

	// 2. check for either creativeId, or creativeCode (or both)
	{
		auto gotit = false;
		// 2. check for either creativeId, or creativeCode
		if (pconf.isMember("creativeId")) {
			getAttr(result, pconf, "creativeId", crinfo->creative_id_,
					includeReasons);
			if (result.isCompatible)
				gotit = true;
		}
		if (pconf.isMember("creativeCode")) {
			getAttr(result, pconf, "creativeCode", crinfo->creative_code_,
					includeReasons);
			if (result.isCompatible)
				gotit = true;
		}
		if (!gotit) {
			result.setIncompatible("creative[].providerConfig['appnexus']: "
					"either 'creativeId' or 'creativeCode' must be configured",
					includeReasons);
		}
		if (!result.isCompatible)
			goto out;
	}

	if (pconf.isMember("clickUrl")) {
		getAttr(result, pconf, "clickUrl", crinfo->click_url_, includeReasons);
		if (!result.isCompatible)
			goto out;
	}
	if (pconf.isMember("pixelUrl")) {
		getAttr(result, pconf, "pixelUrl", crinfo->pixel_url_, includeReasons);
		if (!result.isCompatible)
			goto out;
	}


	getAttr(result, pconf, "attributes", tmp, includeReasons);
    if (!tmp.empty())
    {
        tokenizer<> tok(tmp);
        auto& ints = crinfo->attrs_;
        transform(tok.begin(), tok.end(),
        std::inserter(ints, ints.begin()),[&](const std::string& s) {
            return atoi(s.data());
        });
    }
	if (result.isCompatible)
	{
		result.info = crinfo;
	}
out:
	return result;
}

bool
AppNexusExchangeConnector::
bidRequestCreativeFilter(const BidRequest & request,
                         const AgentConfig & config,
                         const void * info) const
{
    const auto crinfo = reinterpret_cast<const CreativeInfo*>(info);

    // 1. filter attributes
    const auto& excluded_attribute_seg = request.restrictions.get("excluded_attributes");
    for (auto atr: crinfo->attrs_)
        if (excluded_attribute_seg.contains(atr))
        {
            this->recordHit ("attribute_excluded");
            return false ;
        }

    // 2. filter member etc.
    const auto& member_list = request.restrictions.get("members");
    if (!member_list.contains(crinfo->member_id_))
    {
    	this->recordHit ("unlisted_member: " + to_string(crinfo->member_id_));
    	return false;
    }
    return true;
}

} // namespace RTBKIT

namespace {
using namespace RTBKIT;

struct AtInit {
    AtInit() {
        ExchangeConnector::registerFactory<AppNexusExchangeConnector>();
    }
} atInit;
}

