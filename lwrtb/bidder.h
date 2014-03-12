/*
 * Bidder.h
 *
 *  Created on: Oct 26, 2013
 *      Author: jan
 */

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace lwrtb
{

// enum class  BidStatus: char
enum BidStatus {
    WIN,        ///< Bid was won
    LOSS,       ///< Bid was lost
    TOOLATE,    ///< Bid was too late and so not accepted
    INVALID,    ///< Bid was invalid and so not accepted
    LOSTBID,    ///< Bid was lost somewhere
    DROPPEDBID, ///< Bid was dropped as way too late
    NOBUDGET,   ///< No budget
    BUG         ///< Bug request
};

inline
std::string BidStatusToString (BidStatus status)
{
    switch (status) {
    case BidStatus::WIN:
        return "WIN";
    case BidStatus::LOSS:
        return "LOSS";
    case BidStatus::TOOLATE:
        return "TOOLATE";
    case BidStatus::INVALID:
        return "INVALID";
    case BidStatus::LOSTBID:
        return "LOSTBID";
    case BidStatus::DROPPEDBID:
        return "DROPPEDBID";
    case BidStatus::NOBUDGET:
        return "NOBUDGET";
    case BidStatus::BUG:
        return "BUG";
    default:
        return "NOPE";
    }
}

// forward
class Bidder;

//
//     DELIVERY
//
struct DeliveryEvent {
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
class DeliveryCb
{
public:
    DeliveryCb() {}
    virtual ~DeliveryCb() {}
    virtual void call(Bidder& , const DeliveryEvent&) {}
};

//
//     BID REQUEST
//
struct BidRequestEvent {
    double      timestamp;     // Start time of the auction.
    std::string id;            // Auction id
    std::string bidRequest;
    std::string bids;          // Impressions available for bidding
    double      timeLeftMs;    // Time left of the bid request.
    std::string augmentations; // Data from the augmentors.
    std::string winCostModel;  // Win cost model.
};

class BidRequestCb
{
public:
    BidRequestCb() {}
    virtual ~BidRequestCb() {}
    virtual void call(Bidder& , const BidRequestEvent&) { }
};

//
//     BID RESULT
//
struct BidResultEvent {
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

class BidResultCb
{
public:
    BidResultCb() {}
    virtual ~BidResultCb() {}
    virtual void call(Bidder& , const BidResultEvent&) {}
};

//
//     ERROR
//
struct ErrorEvent {
    double                    timestamp;
    std::string               description;
    std::vector<std::string>  originalError;
};
class ErrorCb
{
public:
    ErrorCb() {}
    virtual ~ErrorCb() {}
    virtual void call(Bidder& , const ErrorEvent&) {}
};


class Bidder
{
public:
    /**
     *    Create a bidder.
     *    \param name name give to the bidder.
     */
    Bidder(const std::string& name,
           const std::string& service_proxy_config = "");

    virtual ~Bidder();

    void init();

    void start(bool sync=false );

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
    std::function<void(const BidRequestEvent&)>   bid_request_cb_;


    /**
     *  the following callback, if set, will  be used i/o
     *  dispatch results coming from back from the router.
     */
    std::function<void(const BidResultEvent&)>     bid_result_cb_;

    /**
     *  the following callback, if set, will  be used i/o
     *  dispatch results coming from back from the router.
     */
    std::function<void(const DeliveryEvent&)>      delivery_event_cb_;

    /**
     * whenever  router receives an invalid message from the
     * agent. This can either be caused by an invalid config,
     * and invalid bid
     */
    std::function<void(const ErrorEvent&)>         error_cb_;

    void setBidRequestCb  (BidRequestCb&);
    void setDeliveryCb    (DeliveryCb&);
    void setBidResultCb   (BidResultCb&);
    void setErrorCb       (ErrorCb&);

private:
    const std::string     name_;
    class impl;
    std::unique_ptr<impl> pimpl_;

    // SWIG plumbing
    BidResultCb*          swig_bres_cb_ ;
    DeliveryCb*           swig_devr_cb_ ;
    ErrorCb*              swig_err_cb_  ;
    BidRequestCb*         swig_breq_cb_ ;

    // unwanted.
    Bidder(const Bidder&);
    Bidder& operator=(const Bidder&);
};

} /* namespace lwrtb */

