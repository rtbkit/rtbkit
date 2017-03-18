//
// Created by Tom Clark on 2017-03-06.
//

#pragma once

#include "rtbkit/plugins/exchange/creative_configuration.h"
#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"

namespace RTBKIT {
    struct SmartRTBExchangeConnector: public OpenRTBExchangeConnector {
        SmartRTBExchangeConnector(ServiceBase& owner, const std::string &name);
        SmartRTBExchangeConnector(const std::string &name,
                                std::shared_ptr<ServiceProxies> proxies);

        static std::string nobid;

        static std::string exchangeNameString() {
            return "smartrtb";
        }

        virtual std::string exchangeName() const {
            return exchangeNameString();
        }

        virtual std::shared_ptr<BidRequest>
        parseBidRequest(HttpAuctionHandler & connection,
                        const HttpHeader & header,
                        const std::string & payload);


        double getTimeAvailableMs(HttpAuctionHandler & handler,
                                  const HttpHeader & header,
                                  const std::string & payload) {
            return 100.0;
        }

        double getRoundTripTimeMs(HttpAuctionHandler & handler,
                                  const HttpHeader & header) {
            return 35.0;
        }

        struct CampaignInfo {
            Id seat;                ///< ID of the exchange seat
        };

        virtual ExchangeCompatibility
        getCampaignCompatibility(const AgentConfig & config,
                                 bool includeReasons) const;

        struct CreativeInfo {
            Id adid;                ///< ID for ad to be service if bid wins
            std::string adm;        ///< Actual XHTML ad markup
            std::string nurl;       ///< Win notice URL
            std::vector<std::string> adomain; ///< Advertiser Domain
            std::vector<std::string> mimeTypes; ///< MIME Types for this creative
            std::string iurl;
            std::string crid;
        };

        virtual ExchangeCompatibility
        getCreativeCompatibility(const Creative & creative, bool includeReasons) const;

        virtual bool
        bidRequestCreativeFilter(const BidRequest & request,
                                 const AgentConfig & config,
                                 const void * info) const;

        private:
            void init();

            typedef TypedCreativeConfiguration<CreativeInfo> SmartrtbCreativeConfiguration;
            SmartrtbCreativeConfiguration configuration_;

            virtual void setSeatBid(Auction const & auction,
                                    int spotNum,
                                    OpenRTB::BidResponse &response) const;
    };
}
