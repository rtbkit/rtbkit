$(eval $(call library,agents_bidder,agents_bidder_interface.cc,rtb_router))
$(eval $(call library,http_bidder,http_bidder_interface.cc,rtb_router openrtb_bid_request))
$(eval $(call library,multi_bidder,multi_bidder_interface.cc,agent_configuration))

bidder_interface_plugins: $(LIB)/libagents_bidder.so $(LIB)/libhttp_bidder.so $(LIB)/libmulti_bidder.so

.PHONY: bidder_interface_plugins
