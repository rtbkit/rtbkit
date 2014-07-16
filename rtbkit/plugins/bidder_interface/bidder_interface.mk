$(eval $(call library,agents_bidder,agents_bidder_interface.cc,))
$(eval $(call library,http_bidder,http_bidder_interface.cc,openrtb_bid_request))

# the code loading the plugins above is part of librtb (bidder_interface.cc)
$(LIB)/librtb.so: $(LIB)/libagents_bidder.so $(LIB)/libhttp_bidder.so
