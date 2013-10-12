 /* mopub_exchange_connector.h     -*- C++ -*-
   Syed Murtaza Hussain Kazmi, 21 June 2013
   Copyright (c) 2013 Virtual Force PVT.  All rights reserved.

*/

#pragma once

#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"

namespace RTBKIT {


/*****************************************************************************/
/* MOPUB EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

/** Exchange connector for MoPub.  This speaks their flavour of the
    OpenRTB 2.1 protocol.

    Configuration options are the same as the HttpExchangeConnector on which
    it is based.
*/

struct MoPubExchangeConnector: public OpenRTBExchangeConnector {

    MoPubExchangeConnector(ServiceBase & owner, const std::string & name);
    MoPubExchangeConnector(const std::string & name, std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "mopub";
    }

    virtual std::string exchangeName() const {
        return exchangeNameString();
    }


	/** This is the information that the Rubicon exchange needs to keep
        for each campaign (agent).
    */
    struct CampaignInfo {
        Id seat;       ///< ID of the Rubicon exchange seat
    };

    virtual double
    getTimeAvailableMs(HttpAuctionHandler & connection,
                       const HttpHeader & header,
                       const std::string & payload);
	   
   
    virtual ExchangeCompatibility
    getCampaignCompatibility(const AgentConfig & config,
                             bool includeReasons) const;
    
    /** This is the information that MoPub needs in order to properly
        filter and serve a creative.
    */
    struct CreativeInfo {
        std::string adm;                  ///< Ad markup
        std::vector<std::string> adomain; ///< Advertiser domains
        Id crid;                          ///< Creative ID
        std::string iurl;                ///< The iURL should be a representation of the creative which is being served. 
        Id adid;                          ///< Ad ID
        OpenRTB::List<OpenRTB::CreativeAttribute> attr; ///< Creative attributes
        std::string ext_creativeapi;      ///< Creative API
    };

    virtual ExchangeCompatibility
    getCreativeCompatibility(const Creative & creative,
                             bool includeReasons) const;

   private:
    virtual void setSeatBid(Auction const & auction,
                            int spotNum,
                            OpenRTB::BidResponse & response) const;
};

} // namespace RTBKIT

