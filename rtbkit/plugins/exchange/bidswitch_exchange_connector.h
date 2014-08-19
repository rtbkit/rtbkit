/* bidswitch_exchange_connector.h                                    -*- C++ -*-
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

#pragma once
#include <set>
#include <unordered_map>
#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/core/router/filters/generic_filters.h"
#include "rtbkit/common/creative_configuration.h"

namespace RTBKIT {


/*****************************************************************************/
/* BIDSWITCH EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

/** Exchange connector for BidSwitch.  This speaks their flavour of the
    OpenRTB 2.1 protocol.
*/

struct BidSwitchExchangeConnector: public OpenRTBExchangeConnector {
    BidSwitchExchangeConnector(ServiceBase & owner, const std::string & name);
    BidSwitchExchangeConnector(const std::string & name,
                               std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "bidswitch";
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
                       const std::string & payload) {
        // TODO: check that is at it seems
        return 200.0;
    }

    /** This is the information that the BidSwitch exchange needs to keep
        for each campaign (agent).
    */
    struct CampaignInfo {
        Id seat;          ///< ID of the BidSwitch exchange seat
        std::string iurl; ///< Image URL for content checking
    };

    virtual ExchangeCompatibility
    getCampaignCompatibility(const AgentConfig & config,
                             bool includeReasons) const;

    /** This is the information that BidSwitch needs in order to properly
        filter and serve a creative.
    */
    struct CreativeInfo {
        Id adid;                ///< ID for ad to be service if bid wins
        std::string nurl;       ///< Win notice URL
        std::vector<std::string> adomain;    ///< Advertiser Domain
        struct {
            std::vector<int32_t> vendor_type_;
            std::set<int32_t> attribute_ ;
        } Google;
    };

    typedef CreativeConfiguration<CreativeInfo> BidSwitchCreativeConfiguration;
    
    virtual ExchangeCompatibility
    getCreativeCompatibility(const Creative & creative,
                             bool includeReasons) const;

    virtual bool
    bidRequestCreativeFilter(const BidRequest & request,
                             const AgentConfig & config,
                             const void * info) const;


    // BidSwitch win price decoding function.
    static float decodeWinPrice(const std::string & sharedSecret,
                                const std::string & winPriceStr);

  private:

    BidSwitchCreativeConfiguration configuration_;

    virtual void setSeatBid(Auction const & auction,
                            int spotNum,
                            OpenRTB::BidResponse & response) const;

    void init();
};

struct BidSwitchWSeatFilter : public FilterBaseT<BidSwitchWSeatFilter>
{
    static constexpr const char* name = "bidswitch-wseat";
    unsigned priority() const { return 10; }

    std::unordered_map<std::string, ConfigSet> data;
    ConfigSet emptyConfigSet;

    void setConfig(unsigned configIndex, const AgentConfig& config, bool value)
    {
        Json::Value cfg = config.providerConfig["bidswitch"];
        if(!cfg.empty() && !cfg["seat"].empty()) {
            // Might change depending if the providerConfig allows more than one seat.
            data[config.providerConfig["bidswitch"]["seat"].asString()].set(configIndex, value);
        }
        else {
            emptyConfigSet.set(configIndex, value);
        }
    }

    void filter(FilterState& state) const
    {
        ConfigSet mask(emptyConfigSet);

        auto& segs = state.request.segments.get("openrtb-wseat");
 
        // Calls the filter for every wseat in the BR.
        segs.forEach([&](int, const std::string &str, float) {
            auto it = data.find(str);
            if(!(it == data.end())) {
                auto& configs = it->second;
                mask |= configs;
            }
        });
        
        state.narrowConfigs(mask);
    }
};

} // namespace RTBKIT
