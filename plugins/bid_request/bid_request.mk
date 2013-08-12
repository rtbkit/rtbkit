# bid_request.mk
# Jeremy Barnes, 19 February 2013
# Copyright (c) 2013 Datacratic Inc.  All rights reserved.
#
# Makefile for bid request deserializers and parsers

$(eval $(call library,openrtb_bid_request,openrtb_bid_request.cc,bid_request openrtb))
$(eval $(call library,fbx_bid_request,fbx_bid_request.cc fbx_parsing.cc,bid_request))
$(eval $(call library,appnexus_bid_request,appnexus_bid_request.cc appnexus_parsing.cc,bid_request openrtb))

$(eval $(call include_sub_make,bid_request_testing,testing,bid_request_testing.mk))


