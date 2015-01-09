# RTBKIT common makefile

LIBBIDREQUEST_SOURCES := \
	bid_request.cc \
	segments.cc \
	json_holder.cc \
	currency.cc \
	expand_variable.cc 

LIBBIDREQUEST_LINK := \
	types boost_regex db openrtb value_description

$(eval $(call library,bid_request,$(LIBBIDREQUEST_SOURCES),$(LIBBIDREQUEST_LINK)))

LIBRTB_SOURCES := \
	auction.cc \
	augmentation.cc \
	account_key.cc \
	bids.cc \
	auction_events.cc \
	exchange_connector.cc \
	bidder_interface.cc \
	win_cost_model.cc \
	post_auction_proxy.cc \
	analytics_publisher.cc

LIBRTB_LINK := \
	ACE arch utils jsoncpp boost_thread endpoint boost_regex zmq opstats bid_request

$(eval $(call library,rtb,$(LIBRTB_SOURCES),$(LIBRTB_LINK)))

$(eval $(call library,filter_registry,filter.cc,arch utils rtb))

$(eval $(call include_sub_make,testing,,common_testing.mk))
