/* openrtb_exchange_connector.h                                    -*- C++ -*-
   Eric Robert, 7 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include "rtbkit/plugins/exchange/http_exchange_connector.h"

namespace RTBKIT {

/*****************************************************************************/
/* OPENRTB EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

/** Generic exchange connector using the OpenRTB protocol.

    Configuration options are the same as the HttpExchangeConnector on which
    it is based.
*/

struct OpenRTBExchangeConnector : public HttpExchangeConnector {
    OpenRTBExchangeConnector(ServiceBase & owner, const std::string & name);
    OpenRTBExchangeConnector(const std::string & name,
                             std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "openrtb";
    }

    virtual std::string exchangeName() const {
        return exchangeNameString();
    }

    virtual std::shared_ptr<BidRequest>
    parseBidRequest(HttpAuctionHandler & connection,
                    const HttpHeader & header,
                    const std::string & payload);

    virtual double
    getTimeAvailableMs(HttpAuctionHandler & connection,
                       const HttpHeader & header,
                       const std::string & payload);

    virtual HttpResponse
    getResponse(const HttpAuctionHandler & connection,
                const HttpHeader & requestHeader,
                const Auction & auction) const;

    virtual HttpResponse
    getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                              const std::string & reason) const;

    virtual HttpResponse
    getErrorResponse(const HttpAuctionHandler & connection,
                     const std::string & errorMessage) const;

    virtual std::string getBidSourceConfiguration() const;

private:

    virtual Json::Value
    getResponseExt(const HttpAuctionHandler & connection,
                   const Auction & auction) const;
protected:

    virtual void setSeatBid(Auction const & auction,
                            int spotNum,
                            OpenRTB::BidResponse & response) const;
};

} // namespace RTBKIT
