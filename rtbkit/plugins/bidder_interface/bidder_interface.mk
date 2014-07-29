$(eval $(call library,agents_bidder,agents_bidder_interface.cc,rtb_router))
$(eval $(call library,http_bidder,http_bidder_interface.cc,openrtb_bid_request,rtb_router))
$(eval $(call library,multi_bidder,multi_bidder_interface.cc,))


bidder_interface_plugins: $(LIB)/libagents_bidder.so $(LIB)/libhttp_bidder.so


.PHONY: bidder_interface_plugins
