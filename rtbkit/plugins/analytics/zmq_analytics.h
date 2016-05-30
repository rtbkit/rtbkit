/* zmq_analytics.h
   Sirma Cagil Altay, 7 Mar 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.

   This is a logger class that publishes events through Zmq
*/

#pragma once

#include <memory>
#include <functional>
#include <string>

#include "rtbkit/common/analytics.h"
#include "soa/service/zmq_named_pub_sub.h"

namespace RTBKIT {
    class Router; 
    struct MatchedWinLoss; 
    struct MatchedCampaignEvent;
    struct UnmatchedEvent;
    struct PostAuctionErrorEvent;
}
namespace Datacratic {
    struct Id;
    class ServiceProxies;
}

namespace RTBKIT {

class ZmqAnalytics : public Analytics {

public:
    ZmqAnalytics(const std::string & service_name, std::shared_ptr<Datacratic::ServiceProxies> proxies);
    virtual ~ZmqAnalytics(); 

    virtual void init();
    virtual void bindTcp(const std::string & port_range = "logs");
    virtual void start();
    virtual void shutdown();

    // USED IN ROUTER 
    virtual void logMarkMessage(const Router & router,
                                const double & last_check);
    virtual void logBidMessage(const std::string & agent,
                               const Id & auctionId,
                               const std::string & bids,
                               const std::string & meta);
    virtual void logAuctionMessage(const Id & auctionId,
                                   const std::string & auctionRequest);
    virtual void logConfigMessage(const std::string & agent,
                                  const std::string & config);
    virtual void logNoBudgetMessage(const std::string agent,
                                    const Id & auctionId,
                                    const std::string & bids,
                                    const std::string & meta);
    virtual void logMessage(const std::string & msg,
                            const std::string agent,
                            const Id & auctionId,
                            const std::string & bids,
                            const std::string & meta);
    virtual void logUsageMessage(Router & router,
                                 const double & period);
    virtual void logErrorMessage(const std::string & error,
                                 const std::vector<std::string> & message);
    virtual void logRouterErrorMessage(const std::string & function,
                                       const std::string & exception, 
                                       const std::vector<std::string> & message);

    // USED IN PA
    virtual void logMatchedWinLoss(const MatchedWinLoss & matchedWinLoss);
    virtual void logMatchedCampaignEvent(const MatchedCampaignEvent & matchedCampaignEvent);
    virtual void logUnmatchedEvent(const UnmatchedEvent & unmatchedEvent);
    virtual void logPostAuctionErrorEvent(const PostAuctionErrorEvent & postAuctionErrorEvent); 
    virtual void logPAErrorMessage(const std::string & function,
                                   const std::string & exception, 
                                   const std::vector<std::string> & message);

    // USED IN MOCK ADSERVER CONNECTOR
    virtual void logMockWinMessage(const std::string & eventAuctionId,
                                   const std::string & eventWinPrice);

    // USED IN STANDARD ADSERVER CONNECTOR 
    virtual void logStandardWinMessage(const std::string & timestamp,
                                       const std::string & bidRequestId,
                                       const std::string & impId,
                                       const std::string & winPrice);
    virtual void logStandardEventMessage(const std::string & eventType,
                                         const std::string & timestamp,
                                         const std::string & bidRequestId,
                                         const std::string & impId,
                                         const std::string & userIds);

    // USED IN OTHER ADSERVER CONNECTORS
    virtual void logAdserverEvent(const std::string & type,
                                  const std::string & bidRequestId,
                                  const std::string & impId);
    virtual void logAdserverWin(const std::string & timestamp,
                                const std::string & auctionId,
                                const std::string & adSpotId,
                                const std::string & accountKey,
                                const std::string & winPrice,
                                const std::string & dataCost);
    virtual void logAuctionEventMessage(const std::string & event,
                                        const std::string & timestamp,
                                        const std::string & auctionId,
                                        const std::string & adSpotId,
                                        const std::string & userId);
    virtual void logEventJson(const std::string & event,
                              const std::string & timestamp,
                              const std::string & json);
    virtual void logDetailedWin(const std::string timestamp,
                                const std::string & json,
                                const std::string & auctionId,
                                const std::string & spotId,
                                const std::string & price,
                                const std::string & userIds,
                                const std::string & campaign,
                                const std::string & strategy,
                                const std::string & bidTimeStamp);

private:
    Datacratic::ZmqNamedPublisher zmq_publisher_;

}; // class ZmqEventLogger

} // namespace RTBKIT
