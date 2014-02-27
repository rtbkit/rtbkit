/*
 * Bidder.cpp
 *
 *  Created on: Oct 26, 2013
 *      Author: jan
 */

#include <string>
#include <memory>

#include "lwrtb/bidder.h"
#include "soa/jsoncpp/value.h"
#include "plugins/bidding_agent/bidding_agent.h"

using namespace std;
using namespace RTBKIT;
using namespace Datacratic;

namespace lwrtb
{
struct Bidder::impl {
    shared_ptr<ServiceProxies> prx_;
    unique_ptr<BiddingAgent>   bidding_agent_;
};
std::unique_ptr<int> pl;

Bidder::Bidder(const string& name,
               const string& svc_prx_config)
    : name_         (name)
    , pimpl_        (new impl())
    , swig_bres_cb_ (nullptr)
    , swig_devr_cb_ (nullptr)
    , swig_err_cb_  (nullptr)
    , swig_breq_cb_ (nullptr)
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
    if (bid_request_cb_) {
        pimpl_->bidding_agent_->onBidRequest = [this] (double timestamp,
                                               Id & id,
                                               shared_ptr<RTBKIT::BidRequest> br,
                                               const Bids& bids,
                                               double timeLeftMs,
                                               const Json::Value & augmentations,
        const WinCostModel& wcm) {
            lwrtb::BidRequestEvent res {
                timestamp,
                id.toString(),
                br->toJsonStr(),
                bids.toJson().toString(),
                timeLeftMs,
                augmentations.toString(),
                wcm.toJson().toString()
            };
            this->bid_request_cb_ (res);
        };
    }

    if (error_cb_) {
        pimpl_->bidding_agent_->onError = [this](double timestamp,
                                          const std::string& description,
        const std::vector<std::string> originalError) {
            lwrtb::ErrorEvent res {
                timestamp,
                description,
                originalError
            };
            this->error_cb_(res);
        };
    }


    if (bid_result_cb_) {
        static auto my_convert = [] (const RTBKIT::BidStatus& s) {
            if (s== RTBKIT::BidStatus::BS_WIN) return lwrtb::BidStatus::WIN;
            if (s== RTBKIT::BidStatus::BS_LOSS) return lwrtb::BidStatus::LOSS;
            if (s== RTBKIT::BidStatus::BS_TOOLATE) return lwrtb::BidStatus::TOOLATE;
            if (s== RTBKIT::BidStatus::BS_INVALID) return lwrtb::BidStatus::INVALID;
            if (s== RTBKIT::BidStatus::BS_LOSTBID) return lwrtb::BidStatus::LOSTBID;
            if (s== RTBKIT::BidStatus::BS_DROPPEDBID) return lwrtb::BidStatus::DROPPEDBID;
            if (s== RTBKIT::BidStatus::BS_NOBUDGET) return lwrtb::BidStatus::NOBUDGET;
            return lwrtb::BidStatus::BUG;
        };
        pimpl_->bidding_agent_->onWin = [this] (const RTBKIT::BidResult &br) {
            lwrtb::BidResultEvent res {
                my_convert(br.result),
                br.timestamp,
                br.auctionId.toString(),
                br.spotNum,
                br.secondPrice.toString(),
                br.request.get() ? br.request->toJsonStr() : "",
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

    if (delivery_event_cb_) {
        pimpl_->bidding_agent_->onImpression = [this] (const RTBKIT::DeliveryEvent& de) {
            lwrtb::DeliveryEvent ode {
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


void Bidder::setBidRequestCb  (BidRequestCb& cb)
{
    this->swig_breq_cb_ = &cb ;
    bid_request_cb_ = [&] (const lwrtb::BidRequestEvent& ev) {
        this->swig_breq_cb_->call (*this, ev);
    };
}


void Bidder::setDeliveryCb  (DeliveryCb& cb)
{
    this->swig_devr_cb_ = &cb ;
    delivery_event_cb_ = [&] (const DeliveryEvent& de) {
        this->swig_devr_cb_->call (*this,de);
    };
}


void Bidder::setBidResultCb  (BidResultCb& cb)
{
    this->swig_bres_cb_ = &cb ;
    bid_result_cb_ = [&] (const BidResultEvent& br) {
        this->swig_bres_cb_->call (*this,br);
    };
}


void Bidder::setErrorCb  (ErrorCb& cb)
{
    this->swig_err_cb_ = &cb ;
    error_cb_ = [&] (const ErrorEvent& err) {
        this->swig_err_cb_->call (*this,err);
    };
}

} /* namespace lwrtb */
