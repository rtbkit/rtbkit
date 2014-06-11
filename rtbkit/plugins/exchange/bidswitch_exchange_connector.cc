/* bidswitch_exchange_connector.cc
   Jeremy Barnes, 15 March 2013

   Implementation of the BidSwitch exchange connector.
*/

#include "bidswitch_exchange_connector.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include "jml/utils/file_functions.h"

#include "crypto++/blowfish.h"
#include "crypto++/modes.h"
#include "crypto++/filters.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

/*****************************************************************************/
/* BIDSWITCH EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

BidSwitchExchangeConnector::
BidSwitchExchangeConnector(ServiceBase & owner, const std::string & name)
     : OpenRTBExchangeConnector(owner, name)
{
     this->auctionResource = "/auctions";
     this->auctionVerb = "POST";
}

BidSwitchExchangeConnector::
BidSwitchExchangeConnector(const std::string & name,
                           std::shared_ptr<ServiceProxies> proxies)
     : OpenRTBExchangeConnector(name, proxies)
{
     this->auctionResource = "/auctions";
     this->auctionVerb = "POST";
}

ExchangeConnector::ExchangeCompatibility
BidSwitchExchangeConnector::
getCampaignCompatibility(const AgentConfig & config,
                         bool includeReasons) const
{
     ExchangeCompatibility result;
     result.setCompatible();

     auto cpinfo = std::make_shared<CampaignInfo>();

     const Json::Value & pconf = config.providerConfig["bidswitch"];

     try {
          cpinfo->iurl = pconf["iurl"].asString();
          if (!cpinfo->iurl.size())
               result.setIncompatible("providerConfig.bidswitch.iurl is null",
                                      includeReasons);
     } catch (const std::exception & exc) {
          result.setIncompatible
          (string("providerConfig.bidswitch.iurl parsing error: ")
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
               ("creative[].providerConfig.bidswitch." + string(fieldName)
                + " must be specified", includeReasons);
               return;
          }

          const Json::Value & val = config[fieldName];

          jsonDecode(val, field);
     } catch (const std::exception & exc) {
          result.setIncompatible("creative[].providerConfig.bidswitch."
                                 + string(fieldName) + ": error parsing field: "
                                 + exc.what(), includeReasons);
          return;
     }
}

} // file scope

ExchangeConnector::ExchangeCompatibility
BidSwitchExchangeConnector::
getCreativeCompatibility(const Creative & creative,
                         bool includeReasons) const
{
     ExchangeCompatibility result;
     result.setCompatible();

     auto crinfo = std::make_shared<CreativeInfo>();

     const Json::Value & pconf = creative.providerConfig["bidswitch"];

     // 1.  Must have bidswitch.nurl that includes BidSwitch's macro
     getAttr(result, pconf, "nurl", crinfo->nurl, includeReasons);
     if (crinfo->nurl.find("${AUCTION_PRICE}") == string::npos)
          result.setIncompatible
          ("creative[].providerConfig.bidswitch.nurl ad markup must contain "
           "encrypted win price macro ${AUCTION_PRICE}",
           includeReasons);

     // 2.  Must have creative ID in bidswitch.crid
     getAttr(result, pconf, "adid", crinfo->adid, includeReasons);
     if (!crinfo->adid)
          result.setIncompatible
          ("creative[].providerConfig.bidswitch.adid is null",
           includeReasons);


     // 3.  Must have AdvertiserDomain in bidswitch.crid
     getAttr(result, pconf, "adomain", crinfo->adomain, includeReasons);
     if (crinfo->adomain.empty())
          result.setIncompatible
          ("creative[].providerConfig.bidswitch.adomain is null",
           includeReasons);
     // Cache the information
     result.info = crinfo;

     return result;
}

namespace {

struct GoogleObject {
     std::vector<int> allowed_vendor_type;
     std::vector<std::pair<int,double>> detected_vertical;
     std::vector<int> excluded_attribute;
     void dump() const {
          cerr << "allowed_vendor_type: " << allowed_vendor_type << endl ;
          cerr << "excluded_attribute: " << excluded_attribute << endl ;
          cerr << "detected_vertical  : [" ;
          for (auto ii: detected_vertical)
               cerr << '(' << ii.first << ',' << ii.second << ')';
          cerr << ']' << endl ;

     }
};


GoogleObject
parseGoogleObject(const Json::Value& gobj)
{
     GoogleObject rc;
     if (gobj.isMember("allowed_vendor_type")) {
          const auto& avt = gobj["allowed_vendor_type"];
          if (avt.isArray()) {
               for (auto ii: avt) {
                    rc.allowed_vendor_type.push_back (ii.asInt());
               }
          }
     }
     if (gobj.isMember("excluded_attribute")) {
          const auto& avt = gobj["excluded_attribute"];
          if (avt.isArray()) {
               for (auto ii: avt) {
                    rc.excluded_attribute.push_back (ii.asInt());
               }
          }
     }
     if (gobj.isMember("detected_vertical")) {
          const auto& avt = gobj["detected_vertical"];
          if (avt.isArray()) {
               for (auto ii: avt) {
                    rc.detected_vertical.push_back ( {ii["id"].asInt(),ii["weight"].asDouble()});
               }
          }
     }
     rc.dump();
     return rc;
}
}

std::shared_ptr<BidRequest>
BidSwitchExchangeConnector::
parseBidRequest(HttpAuctionHandler & connection,
                const HttpHeader & header,
                const std::string & payload)
{
     std::shared_ptr<BidRequest> res;
//
     // Check for JSON content-type
     if (header.contentType != "application/json") {
          connection.sendErrorResponse("non-JSON request");
          return res;
     }

#if 0
     /*
      * Unfortunately, x-openrtb-version isn't sent in the real traffic
      */
     // Check for the x-openrtb-version header
     auto it = header.headers.find("x-openrtb-version");
     if (it == header.headers.end()) {
          connection.sendErrorResponse("no OpenRTB version header supplied");
          return res;
     }

     // Check that it's version 2.1
     std::string openRtbVersion = it->second;
     if (openRtbVersion != "2.0") {
          connection.sendErrorResponse("expected OpenRTB version 2.0; got " + openRtbVersion);
          return res;
     }
#endif

     // Parse the bid request
     ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
     res.reset(OpenRtbBidRequestParser::parseBidRequest(context, exchangeName(), exchangeName()));

     const auto& ext = res->ext;

     if (ext.isMember("ssp")) {
          if (ext.isMember("google")) {
               const auto& gobj = ext["google"];
               cerr << gobj << endl ;
               parseGoogleObject (gobj);
          }
          if (ext.isMember("adtruth")) {
               const auto& adt = ext["adtruth"];
               cerr << adt.toString() << endl ;
          }
     }

     return res;
}


void
BidSwitchExchangeConnector::
setSeatBid(Auction const & auction,
           int spotNum,
           OpenRTB::BidResponse & response) const
{

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
     b.price.val = USD_CPM(resp.price.maxPrice);
     b.nurl = crinfo->nurl;
     b.adid = crinfo->adid;
     b.adomain = crinfo->adomain;
     b.iurl = cpinfo->iurl;
}

} // namespace RTBKIT

namespace {
using namespace RTBKIT;

struct Init {
     Init() {
          ExchangeConnector::registerFactory<BidSwitchExchangeConnector>();
     }
} init;
}

