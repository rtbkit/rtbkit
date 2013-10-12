 /* nexage_exchange_connector.cc                                -*- C++ -*-
   Syed Murtaza Hussain Kazmi, 21 June 2013
   Copyright (c) 2013 Virtual Force PVT.  All rights reserved.
*/
#include "nexage_exchange_connector.h"
#include "rtbkit/plugins/bid_request/nexage_bid_request.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

/*****************************************************************************/
/* NeXage EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

	NeXageExchangeConnector::
	NeXageExchangeConnector(ServiceBase & owner, const std::string & name)
		: HttpExchangeConnector(name, owner)
	{
		this->auctionResource = "/auctions";
		this->auctionVerb = "POST";
	}

	NeXageExchangeConnector::
	NeXageExchangeConnector(const std::string & name,
							 std::shared_ptr<ServiceProxies> proxies)
		: HttpExchangeConnector(name, proxies)
	{
		this->auctionResource = "/auctions";
		this->auctionVerb = "POST";
	}


	std::shared_ptr<BidRequest>
	NeXageExchangeConnector::
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
		cerr << endl << endl << "Header Content-Type:" << header.contentType << endl;
		cerr << endl << endl << "Bid Request:::" << endl << payload.c_str() << endl;
		// Parse the bid request
		ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
		result.reset(NexageBidRequestParser::parseBidRequest(context, name, name));
		return result;
	}

	/**     Return the available time for the bid request in milliseconds.
			This method should not parse the bid request, as when 
			shedding load we want to do as little work as possible.
       
	*/

	double
	NeXageExchangeConnector::
	getTimeAvailableMs(HttpAuctionHandler & connection,
					   const HttpHeader & header,
					   const std::string & payload)
	{    
 
	  return 100.0;  //forced by NexAge Exchange
	}

	void
	NeXageExchangeConnector::
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
		b.id        =   Id(auction.id, auction.request->imp[0].id); //Bidder's bid ID to identify bid. 
		b.impid     =   auction.request->imp[spotNum].id; // ID of the impression we're bidding on. 
		b.price.val =   USD_CPM(resp.price.maxPrice); //Price
		b.adid      =   Id(crinfo->adid); //Id of ad to be served if won. 
		b.nurl      =  "http://54.214.251.166:18143"; //Win notice/ad markup URL. For each bid, the “nurl” attribute contains the win notice URL. If the bidder wins the impression, the exchange calls this notice URL a) to inform the bidder of the win and b) to convey certain information using substitution macros
		b.adm       =   crinfo->adm; //Ad markup.
		b.adomain   =   crinfo->adomain; //Advertiser domains.
		b.iurl      =   crinfo->iurl; //parameter per the OpenRTB spec.The "iurl" parameter should be a representation of the creative which you're serving. 
		
		b.cid       =   Id(resp.agent); //Campaign Agent ID. 
		b.crid      =   crinfo->crid; //Creative ID. 
		b.attr      =   crinfo->attr; 
      
			//----------------------------------------------------------------------
	}

	ExchangeConnector::ExchangeCompatibility
	NeXageExchangeConnector::
	getCampaignCompatibility(const AgentConfig & config,
							 bool includeReasons) const
	{
		ExchangeCompatibility result;
		result.setCompatible();

		auto cpinfo = std::make_shared<CampaignInfo>();

		const Json::Value & pconf = config.providerConfig["nexage"];

		try {
			cpinfo->seat = Id(pconf["seat"].asString());
			if (!cpinfo->seat)
				result.setIncompatible("providerConfig.nexage.seat is null",
									   includeReasons);
		} catch (const std::exception & exc) {
			result.setIncompatible
				(string("providerConfig.nexage.seat parsing error: ")
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
					("creative[].providerConfig.nexage." + string(fieldName)
					 + " must be specified", includeReasons);
				return;
			}
        
			const Json::Value & val = config[fieldName];
        
			jsonDecode(val, field);
		}
		catch (const std::exception & exc) {
			result.setIncompatible("creative[].providerConfig.nexage."
								   + string(fieldName) + ": error parsing field: "
								   + exc.what(), includeReasons);
			return;
		}
	}
    
} // file scope

	ExchangeConnector::ExchangeCompatibility
	NeXageExchangeConnector::
	getCreativeCompatibility(const Creative & creative,
							 bool includeReasons) const
	{
		ExchangeCompatibility result;
		result.setCompatible();

		auto crinfo = std::make_shared<CreativeInfo>();

		const Json::Value & pconf = creative.providerConfig["nexage"];

		// 1.  Must have nexage.attr containing creative attributes.  These
		//     turn into NeXageCreativeAttribute filters.
		getAttr(result, pconf, "attr", crinfo->attr, includeReasons);

		// TODO: create filter from these...

		// 2.  Must have nexage.adm that includes NeXage's macro
		getAttr(result, pconf, "adm", crinfo->adm, includeReasons);

        
		// 3.  Must have creative ID in nexage.crid
		getAttr(result, pconf, "crid", crinfo->crid, includeReasons);
		if (!crinfo->crid)
			result.setIncompatible
				("creative[].providerConfig.nexage.crid is null",
				 includeReasons);
            
		// 4.  Must have advertiser names array in nexage.adomain
		getAttr(result, pconf, "adomain", crinfo->adomain,  includeReasons);
		if (crinfo->adomain.empty())
			result.setIncompatible
				("creative[].providerConfig.nexage.adomain is empty",
				 includeReasons);
		//5. Get iurl
		getAttr(result, pconf, "iurl", crinfo->iurl,  includeReasons);
		if (crinfo->iurl.empty())
			result.setIncompatible
				("creative[].providerConfig.nexage.iurl is empty",
				 includeReasons);

		//6. Get AdID
		getAttr(result, pconf, "adid", crinfo->adid,  includeReasons);
	
		// Cache the information
		result.info = crinfo;

		return result;
	}



	HttpResponse
	NeXageExchangeConnector::
	getResponse(const HttpAuctionHandler & connection,
				const HttpHeader & requestHeader,
				const Auction & auction) const
	{

		const Auction::Data * current = auction.getCurrentData();

		if (current->hasError())
			return getErrorResponse(connection, auction, current->error + ": " + current->details);

		OpenRTB::BidResponse response;
		response.id = auction.id;

		cerr<< current->responses.size() <<"size" << endl; 
    
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
		
		cerr << "Bid Response=" << endl <<Json::parse(stream.str());
		
		return HttpResponse(200, "application/json", stream.str());

	}


	HttpResponse
	NeXageExchangeConnector::
	getDroppedAuctionResponse(const HttpAuctionHandler & connection,
							  const std::string & reason) const
	{
		return HttpResponse(204, "application/json", "{}");
	}

	HttpResponse
	NeXageExchangeConnector::
	getErrorResponse(const HttpAuctionHandler & connection,
					 const Auction & auction,
					 const std::string & errorMessage) const
	{
		Json::Value response;
		response["error"] = errorMessage;
		return HttpResponse(400, response);
	}

	Json::Value
	NeXageExchangeConnector::
	getResponseExt(const HttpAuctionHandler & connection,
				const Auction & auction) const
	{
		return {};
	}



} // namespace RTBKIT
namespace {
    using namespace RTBKIT;

    struct Init {
        Init() {
            ExchangeConnector::registerFactory<NeXageExchangeConnector>();
        }
    } init;
}

