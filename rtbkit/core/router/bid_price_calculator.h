/* bid_price_calculator.h
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#pragma once

#include "rtbkit/core/router/router.h"

namespace RTBKIT {

class Router;

struct BidPriceCalculator
{
    BidPriceCalculator(Router * router);

    void sendAuctionMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction,
                            double timeLeftMs,
                            RTBKIT::BiddableSpots const & spots);

    void sendWinMessage(std::string const & agent,
                        std::string const & id,
                        Amount price);

    void sendLossMessage(std::string const & agent,
                         std::string const & id);

    void sendBidLostMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    void sendBidDroppedMessage(std::string const & agent,
                               std::shared_ptr<Auction> const & auction);

    void sendBidInvalidMessage(std::string const & agent,
                               std::string const & reason,
                               std::shared_ptr<Auction> const & auction);

    void sendNoBudgetMessage(std::string const & agent,
                             std::shared_ptr<Auction> const & auction);

    void sendTooLateMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    void sendMessage(std::string const & agent,
                     std::string const & message);

    void sendErrorMessage(std::string const & agent,
                          std::string const & error,
                          std::vector<std::string> const & payload);

    void sendPingMessage(std::string const & agent,
                         int ping);

private:
    Router * router;
};

}

