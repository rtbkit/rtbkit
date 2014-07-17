$(eval $(call library,agents_bidder,agents_bidder_interface.cc,rtb_router))
$(eval $(call library,http_bidder,http_bidder_interface.cc,openrtb_bid_request))

$(BIN)/router_runner $(BIN)/post_auction_runner: $(LIB)/libagents_bidder.so $(LIB)/libhttp_bidder.so
