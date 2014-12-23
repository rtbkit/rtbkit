/* smaato_exchange_connector.cc
   Matthieu Labour, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved. */


#include "smaato_exchange_connector.h"

#include "rtbkit/plugins/bid_request/openrtb_bid_request_parser.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"
#include "soa/service/logs.h"
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include "jml/utils/file_functions.h"

 
using namespace std;
using namespace Datacratic;

namespace {
Logging::Category smaatoExchangeConnectorTrace("Smaato Exchange Connector");
Logging::Category smaatoExchangeConnectorError("[ERROR] Smaato Exchange Connector", smaatoExchangeConnectorTrace);
}
 
namespace RTBKIT  { 

/*****************************************************************************/
/* MOPUB EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

std::string SmaatoExchangeConnector::nobid = "nobid";

SmaatoExchangeConnector::
SmaatoExchangeConnector(ServiceBase & owner, const std::string & name)
    : OpenRTBExchangeConnector(owner, name) {
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
}

SmaatoExchangeConnector::
SmaatoExchangeConnector(const std::string & name,
                       std::shared_ptr<ServiceProxies> proxies)
    : OpenRTBExchangeConnector(name, proxies) {
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
}
 
std::shared_ptr<BidRequest>
SmaatoExchangeConnector::
parseBidRequest(HttpAuctionHandler & connection,
                const HttpHeader & header,
                const std::string & payload)
{
    std::shared_ptr<BidRequest> none;

    size_t found = header.queryParams.uriEscaped().find(SmaatoExchangeConnector::nobid);
    if (found != string::npos) {
      connection.dropAuction("nobid");
      return none;
    }   

    // Check for JSON content-type
    if (!header.contentType.empty()) {
        static const std::string delimiter = ";";

        std::string::size_type posDelim = header.contentType.find(delimiter);
        std::string content;

        if(posDelim == std::string::npos)
            content = header.contentType;
        else {
            content = header.contentType.substr(0, posDelim);
            #if 0
            std::string charset = header.contentType.substr(posDelim, header.contentType.length());
            #endif
        }

        if(content != "application/json") {
            connection.sendErrorResponse("UNSUPPORTED_CONTENT_TYPE", "The request is required to use the 'Content-Type: application/json' header");
            return none;
        }
    }
    else {
        connection.sendErrorResponse("MISSING_CONTENT_TYPE_HEADER", "The request is missing the 'Content-Type' header");
        return none;
    }

    // Check for the x-openrtb-version header
    auto it = header.headers.find("x-openrtb-version");
    if (it == header.headers.end()) {
        connection.sendErrorResponse("MISSING_OPENRTB_HEADER", "The request is missing the 'x-openrtb-version' header");
        return none;
    }

    // Check that it's version 2.0
    std::string openRtbVersion = it->second;
    if (openRtbVersion != "2.0") {
        connection.sendErrorResponse("UNSUPPORTED_OPENRTB_VERSION", "The request is required to be using version 2.0 of the OpenRTB protocol but requested " + openRtbVersion);
        return none;
    }

    if(payload.empty()) {
        this->recordHit("error.emptyBidRequest");
        connection.sendErrorResponse("EMPTY_BID_REQUEST", "The request is empty");
        return none;
    }

    // Parse the bid request
    std::shared_ptr<BidRequest> result;
    try {
        ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
        result.reset(OpenRTBBidRequestParser::openRTBBidRequestParserFactory(openRtbVersion)->parseBidRequest(context,
                                                                                              exchangeName(),
                                                                                              exchangeName()));
    }
    catch(ML::Exception const & e) {
        this->recordHit("error.parsingBidRequest");
        throw;
    }
    catch(...) {
        throw;
    }

    // Check if we want some reporting
    auto verbose = header.headers.find("x-openrtb-verbose");
    if(header.headers.end() != verbose) {
        if(verbose->second == "1") {
            if(!result->auctionId.notNull()) {
                connection.sendErrorResponse("MISSING_ID", "The bid request requires the 'id' field");
                return none;
            }
        }
    }
    


    return result;
}



ExchangeConnector::ExchangeCompatibility
SmaatoExchangeConnector::
getCampaignCompatibility(const AgentConfig & config,
                         bool includeReasons) const {
    ExchangeCompatibility result;
    result.setCompatible();

    auto cpinfo = std::make_shared<CampaignInfo>();
 
    const Json::Value & pconf = config.providerConfig["smaato"];
 
    try {
      cpinfo->seat = Id(pconf["seat"].asString());
      if (!cpinfo->seat)
	result.setIncompatible("providerConfig.smaato.seat is null",
			       includeReasons);
    } catch (const std::exception & exc) {
        result.setIncompatible
	  (string("providerConfig.smaato.seat parsing error: ")
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
             bool includeReasons) {

    try {
        if (!config.isMember(fieldName)) {
            result.setIncompatible
            ("creative[].providerConfig.smaato." + string(fieldName)
             + " must be specified", includeReasons);
            return;
        }

        const Json::Value & val = config[fieldName];

        jsonDecode(val, field);
    } catch (const std::exception & exc) {
        result.setIncompatible("creative[].providerConfig.smaato."
                               + string(fieldName) + ": error parsing field: "
                               + exc.what(), includeReasons);
        return;
    }
}

} // file scope


 
ExchangeConnector::ExchangeCompatibility
  SmaatoExchangeConnector::
  getCreativeCompatibility(const Creative & creative, bool includeReasons) const {
    ExchangeCompatibility result;
    result.setCompatible();
    
    auto crinfo = std::make_shared<CreativeInfo>();
    const Json::Value & pconf = creative.providerConfig["smaato"];
    
    getAttr(result, pconf, "adm", crinfo->adm, includeReasons);
    getAttr(result, pconf, "nurl", crinfo->nurl, includeReasons);
    getAttr(result, pconf, "adid", crinfo->adid, includeReasons);
    getAttr(result, pconf, "adomain", crinfo->adomain, includeReasons);
    getAttr(result, pconf, "mimeTypes", crinfo->mimeTypes, includeReasons);
    result.info = crinfo;
 
    return result;
  }

  bool
  SmaatoExchangeConnector::
  bidRequestCreativeFilter(const BidRequest & request,
            const AgentConfig & config,
            const void * info) const {
      const auto crinfo = reinterpret_cast<const CreativeInfo*>(info);

      // now go through the spots.
      for (const auto& spot: request.imp) {
          //const auto& mime_types = spot.banner->mimes;
	for (const auto& mimeType : spot.banner->mimes) { 
              if (std::find(crinfo->mimeTypes.begin(), crinfo->mimeTypes.end(), mimeType.type)
                      != crinfo->mimeTypes.end()) {
                  this->recordHit ("blockedMime");
                  return true;
              }
	}
      }

       return false;
  }

void
SmaatoExchangeConnector::
setSeatBid(Auction const & auction,
           int spotNum,
           OpenRTB::BidResponse & response) const {

    const Auction::Data * current = auction.getCurrentData();

    // Get the winning bid
    auto & resp = current->winningResponse(spotNum);

    // Find how the agent is configured.  We need to copy some of the
    // fields into the bid.
    const AgentConfig * config =
        std::static_pointer_cast<const AgentConfig>(resp.agentConfig).get();

    std::string en = exchangeName();

    // Get the exchange specific data for this campaign
    auto cpinfo = config->getProviderData<CampaignInfo>(en);

    // Put in the fixed parts from the creative
    int creativeIndex = resp.agentCreativeIndex;

    auto & creative = config->creatives.at(creativeIndex);

    // Get the exchange specific data for this creative
    auto crinfo = creative.getProviderData<CreativeInfo>(en);

    // Find the index in the seats array
    int seatIndex = 0;
    while(response.seatbid.size() != seatIndex) {
        if(response.seatbid[seatIndex].seat == cpinfo->seat) break;
        ++seatIndex;
    }

    // Create if required
    if(seatIndex == response.seatbid.size()) {
        response.seatbid.emplace_back();
        response.seatbid.back().seat = cpinfo->seat;
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
    b.price.val = getAmountIn<CPM>(resp.price.maxPrice);
    b.adm = crinfo->adm;
    b.adomain = crinfo->adomain;
    b.nurl = crinfo->nurl;
}

} // namespace RTBKIT
 
namespace {
  using namespace RTBKIT;
 
  struct Init {
    Init() {
      ExchangeConnector::registerFactory<SmaatoExchangeConnector>();
    }
  } init;
}
