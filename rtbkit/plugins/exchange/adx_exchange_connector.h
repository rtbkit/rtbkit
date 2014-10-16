/*
 * adx_exchange_connector.h
 *
 *  Created on: May 29, 2013
 *      Author: jan sulmont
 */

#ifndef ADX_EXCHANGE_CONNECTOR_H_
#define ADX_EXCHANGE_CONNECTOR_H_

#include <sstream>
#include <unordered_set>

#include "rtbkit/common/creative_configuration.h"
#include "rtbkit/plugins/exchange/http_exchange_connector.h"

namespace RTBKIT {


/**************************************************************************/
/* ADX EXCHANGE CONNECTOR                                                */
/**************************************************************************/

/**
 * Exchange connector for AdX.
 * Configuration options are the same as the HttpExchangeConnector on which
 * it is based.
 *
 * About RTT (route trip time), latencies, and Google AdX:
 *
 * 1) Google requires the bidder side to implement a simple Ping protocol,
 *    working as following: upon receiving of a bid request where is_ping
 *    is set to True, the bidder side should immediately returns a empty bid
 *    response, in which processing_time_ms is set.
 * 2) This protocol allows Google to keep track of the latency to a given
 *    bidder, and throttle accordingly the traffic it's sending to it.
 *    Unlike perhaps other exchanges, Google does not embed RTT in HTTP
 *    header, and thus this information isn't available to a given bidder.
 * 3) In other words, the flow control will be driven from Google, based
 *    on its estimation of the RTT, and also the processing_time_ms contained
 *    in the bid response.
 * 4) Google has a 100 ms deadline policy, including processing time and RTT,
 *    yielding to a 10 bid requests/second worse case scenario.
 *    Without an estimate of the RTT, we chose a conservative approach and
 *    set this RTT estimate to 50ms, resulting in an effective deadline of
 *    50 ms for computing a bid response.
 * 5) based on above, obviously, the lower the processing_time_ms, the more
 *    bid requests Google will send.
*/


struct AdXExchangeConnector: public HttpExchangeConnector {
    AdXExchangeConnector(ServiceBase & owner, const std::string & name);
    AdXExchangeConnector(const std::string & name,
                         std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString()  {
        return "adx";
    }

    virtual std::string exchangeName() const {
        return exchangeNameString();
    }

    void init();

    virtual std::shared_ptr<BidRequest>
    parseBidRequest(HttpAuctionHandler & connection,
                    const HttpHeader & header,
                    const std::string & payload) ;

    virtual double
    getTimeAvailableMs(HttpAuctionHandler & connection,
                       const HttpHeader & header,
                       const std::string & payload) ;

    virtual double
    getRoundTripTimeMs(HttpAuctionHandler & connection,
                       const HttpHeader & header) ;

    virtual HttpResponse
    getResponse(const HttpAuctionHandler & connection,
                const HttpHeader & requestHeader,
                const Auction & auction) const ;

    virtual HttpResponse
    getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                              const std::string & reason) const ;

    virtual HttpResponse
    getErrorResponse(const HttpAuctionHandler & connection,
                     const std::string & errorMessage) const ;

    /** This is the information that AdX needs in order to properly
        filter and serve a creative.
    */
    struct CreativeInfo {
        uint32_t agency_id_ ;                       ///< Agency Id
        std::string buyer_creative_id_ ;            ///< Buyer Creative Id
        std::string html_snippet_;                  ///< landing Url
        std::string click_through_url_;             ///< Click
        std::string adgroup_id_;                    ///< Ad Group Id
        std::unordered_set<int32_t> vendor_type_ ;  ///< Vendor Type
        std::unordered_set<int32_t> category_ ;     ///< Category
        std::unordered_set<int32_t> attribute_;     ///< Attribute
        std::unordered_set<int32_t> 
                        restricted_category_;       ///< Restricted category
    };

    virtual bool
    bidRequestCreativeFilter(const BidRequest & request,
                             const AgentConfig & config,
                             const void * info) const;

    virtual ExchangeCompatibility
    getCreativeCompatibility(const Creative & creative,
                             bool includeReasons) const;

private:

    typedef CreativeConfiguration<CreativeInfo> AdxCreativeConfiguration;
    AdxCreativeConfiguration configuration_;

    /**
     * see class comments
     */
    static double deadline_ms()  {
        return 100.0;
    }
    static double rtt_ms()       {
        return 50;
    }
};



inline
double
AdXExchangeConnector::getTimeAvailableMs(HttpAuctionHandler & ,
        const HttpHeader & , const std::string & )
{
    return deadline_ms ();
}

inline
double
AdXExchangeConnector::getRoundTripTimeMs(HttpAuctionHandler &, const HttpHeader & )
{
    return rtt_ms ();
}


} // namespace RTBKIT


#endif /* ADX_EXCHANGE_CONNECTOR_H_ */
