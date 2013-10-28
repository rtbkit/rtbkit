/*
 * BiddingAgent.h
 *
 *  Created on: Oct 26, 2013
 *      Author: jan
 */

#ifndef BIDDINGAGENT_H_
#define BIDDINGAGENT_H_

#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace RTBKIT {
namespace api {

enum BidStatus
{
    WIN,        ///< Bid was won
    LOSS,       ///< Bid was lost
    TOOLATE,    ///< Bid was too late and so not accepted
    INVALID,    ///< Bid was invalid and so not accepted
    LOSTBID,    ///< Bid was lost somewhere
    DROPPEDBID, ///< Bid was dropped as way too late
    NOBUDGET,   ///< No budget
    BUG         ///< Bug request
};

;

struct BidResult
{
    BidStatus                 result;        ///> Result of our bid
    double                    timestamp;     ///> Time at which the event occured
    std::string               auctionId;     ///> Unique auction id for the original bid
    int                       spotNum;       ///> Spot index into the bidRequest.imp or ourBid
    std::string               secondPrice;   ///> Depends on result from router or the exchange
    std::string               bidRequest;    ///> Original request we bid on
    std::string               ourBid;        ///> Original bids that was placed
    std::string               metadata;      ///> Metadata that was attached to the bid
    std::string               augmentations; ///> Original Augmentations sent with the request
};

struct DeliveryEvent
{
    std::string               event;
    double                    timestamp;      // number of seconds since epoch
    std::string               auctionId;
    std::string               spotId;
    int                       spotIndex;
    std::string               bidRequest;
    std::string               augmentations;
    std::string               bid;
    std::string               win;
    std::string               campaignEvents;
    std::vector<std::string>  visits;
};

class Bidder
{
public:
    /**
     *    Create a bidder.
     *    \param name name give to the bidder.
     */
     Bidder(const std::string& service_proxy_json,
    		const std::string& name);

    virtual ~Bidder();

    void init();

    void shutdown();

    /** Notify the AgentConfigurationService that the configuration of the
     *  bidding agent has changed.
     *  \param config string rep. of a AgentConfig json object
     *  Note that bidding agent will remember the given configuration which will
     *  be used to answer any further configuration requests that are received.
     *  This function is thread-safe and can be called at any time to change the
     *  bid request which will be received by the agent. Note that update are
     *  done asynchronously and changes might not take effect immediately.
     */
    void doConfig(const std::string& config);

    /** Send a bid response to the router in answer to a received auction.
     * \param id  string rep.  of an auction id given in the auction callback.
     * \param bids string rep. of a Bids struct converted to json.
     * \param meta string rep. of a json blob that will be returned as is in the bid result.
     * \param wcm string rep of a Win cost model to be used.
     */
    void doBid(const std::string& id,
               const std::string& bids,
               const std::string& meta = "",
               const std::string& wmc  = "");

    /**
     *   the following callback if set,  will be used
     *   for the delivery of relevant bid requests
     */
    std::function<
    void( double               timestamp,    // Start time of the auction.
          const std::string&   id,           // Auction id
          const std::string&   bidRequest,
          const std::string&   bids,         // Impressions available for bidding
          double               timeLeftMs,   // Time left of the bid request
          const std::string&   augmentations,// data for the augmentors
          const std::string&   wcm)          // Win cost model
    >                                         bid_request_cb_;


    /**
     *  the following callback, if set, will  be used i/o
     *  dispatch results coming from back from the router.
     */
    std::function<void(const BidResult&)>     bid_result_cb_;

    /**
     *  the following callback, if set, will  be used i/o
     *  dispatch results coming from back from the router.
     */
    std::function<void(const DeliveryEvent&)> delivery_event_cb_;

    /**
     * whenever  router receives an invalid message from the
     * agent. This can either be caused by an invalid config, and invalid bid
     */
    std::function<
    void(double timestamp,
         std::string description,
         std::vector<std::string> originalError)
    >                                         error_cb_;

private:
    const std::string&    name_;
    class impl;
    std::unique_ptr<impl> pimpl_;
    // unwanted.
    Bidder(const Bidder&);
    Bidder& operator=(const Bidder&);
};

} /* namespace api */
} /* namespace RTBKIT */
#endif /* BIDDINGAGENT_H_ */
