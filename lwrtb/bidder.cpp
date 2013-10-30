/*
 * BiddingAgent.cpp
 *
 *  Created on: Oct 26, 2013
 *      Author: rtbkit
 */

#include <string>
#include <memory>

#include "api/bidder.h"
#include "soa/jsoncpp/value.h"
#include "plugins/bidding_agent/bidding_agent.h"

using namespace std;

namespace RTBKIT {
namespace api {
struct Bidder::impl
{
    shared_ptr<ServiceProxies> prx_;
    unique_ptr<BiddingAgent>   bidding_agent_;
};

Bidder::Bidder(const string& name,
               const string& svc_prx_config)
    : name_  (name)
    , pimpl_ (new impl())
{
    pimpl_->prx_.reset (new ServiceProxies);
    if (svc_prx_config.length())
        pimpl_->prx_->bootstrap (Json::parse(svc_prx_config));
    pimpl_->bidding_agent_.reset (new BiddingAgent(pimpl_->prx_, name));
    // pimpl_->bidding_agent_->strictMode(false);


}

Bidder::~Bidder()
{

}

void Bidder::start(bool sync)
{
    if (sync)
        pimpl_->bidding_agent_->startSync();
    else
        pimpl_->bidding_agent_->start();
}

void
Bidder::init()
{
	pimpl_->bidding_agent_->strictMode(false);
    if (bid_request_cb_)
        pimpl_->bidding_agent_->onBidRequest = [this] (double timestamp,
                                               Id & id,
                                               shared_ptr<RTBKIT::BidRequest> br,
                                               const Bids& bids,
                                               double timeLeftMs,
                                               const Json::Value & augmentations,
        const WinCostModel& wcm) {
    	this->bid_request_cb_ (timestamp,
                               id.toString(),
                               br->toJsonStr(),
                               bids.toJson().toString(),
                               timeLeftMs,
                               augmentations.toString(),
                               wcm.toJson().toString());
    };

    if (error_cb_)
        pimpl_->bidding_agent_->onError = this->error_cb_;

    if (bid_result_cb_)
    {
        static auto my_convert = [] (const RTBKIT::BidStatus& s) {
            if (s== RTBKIT::BidStatus::BS_WIN) return RTBKIT::api::BidStatus::WIN;
            if (s== RTBKIT::BidStatus::BS_LOSS) return RTBKIT::api::BidStatus::LOSS;
            if (s== RTBKIT::BidStatus::BS_TOOLATE) return RTBKIT::api::BidStatus::TOOLATE;
            if (s== RTBKIT::BidStatus::BS_INVALID) return RTBKIT::api::BidStatus::INVALID;
            if (s== RTBKIT::BidStatus::BS_LOSTBID) return RTBKIT::api::BidStatus::LOSTBID;
            if (s== RTBKIT::BidStatus::BS_DROPPEDBID) return RTBKIT::api::BidStatus::DROPPEDBID;
            if (s== RTBKIT::BidStatus::BS_NOBUDGET) return RTBKIT::api::BidStatus::NOBUDGET;
            return RTBKIT::api::BidStatus::BUG;
        };
        pimpl_->bidding_agent_->onWin = [this] (const RTBKIT::BidResult &br) {
            RTBKIT::api::BidResult res {
                my_convert(br.result),
                br.timestamp,
                br.auctionId.toString(),
                br.spotNum,
                br.secondPrice.toString(),
                br.request->toJsonStr(),
                br.ourBid.toJson().toString(),
                br.metadata.toString(),
                br.augmentations.toString()
            };
            this->bid_result_cb_ (res);
        };
        pimpl_->bidding_agent_->onLoss = pimpl_->bidding_agent_->onWin;
        pimpl_->bidding_agent_->onNoBudget = pimpl_->bidding_agent_->onWin;
        pimpl_->bidding_agent_->onTooLate = pimpl_->bidding_agent_->onWin;
        pimpl_->bidding_agent_->onDroppedBid = pimpl_->bidding_agent_->onWin;
        pimpl_->bidding_agent_->onInvalidBid = pimpl_->bidding_agent_->onWin;
    }

    if (delivery_event_cb_)
    {
        pimpl_->bidding_agent_->onImpression = [this] (const RTBKIT::DeliveryEvent& de) {
            RTBKIT::api::DeliveryEvent ode {
                de.event,
                de.timestamp.secondsSinceEpoch(),
                de.auctionId.toString(),
                de.spotId.toString(),
                de.spotIndex,
                de.bidRequest->toJsonStr(),
                de.augmentations,
                de.bid.toJson().toString(),
                de.win.toJson().toString(),
                de.campaignEvents.toJson().toString()
            };
            for (const auto& v: de.visits)
                ode.visits.emplace_back(v.toJson().toString());
        };
        pimpl_->bidding_agent_->onClick = pimpl_->bidding_agent_->onImpression;
        pimpl_->bidding_agent_->onVisit = pimpl_->bidding_agent_->onImpression;
    }
    pimpl_->bidding_agent_->init ();
}

void
Bidder::shutdown()
{
    pimpl_->bidding_agent_->shutdown ();
}

void
Bidder::doConfig(const string& config)
{
    pimpl_->bidding_agent_->doConfigJson(Json::parse(config));
}

void
Bidder::doBid(const string& id,
              const string& bids,
              const string& meta,
              const string& wmc)
{
    pimpl_->bidding_agent_->doBid (
        Id(id),
        Bids::fromJson(bids),
        meta.size() ? Json::parse(meta) : Json::Value(),
        wmc.size() ? WinCostModel::fromJson(Json::parse(wmc)) : WinCostModel());
}


} /* namespace api */
} /* namespace RTBKIT */
