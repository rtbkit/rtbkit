
$(eval $(call library,agents_bidder,agents_bidder_interface.cc,rtb_router))
$(eval $(call library,http_bidder,http_bidder_interface.cc,rtb_router openrtb_bid_request))

