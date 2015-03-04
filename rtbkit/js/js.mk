#------------------------------------------------------------------------------#
# js.mk
# Rémi Attab, 01 Mar 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# Makefile for the various js wrappers
#------------------------------------------------------------------------------#

$(eval $(call nodejs_addon,config_validator,config_validator_js.cc,rtb_router))
$(eval $(call nodejs_addon,bid_request,bid_request_js.cc currency_js.cc,bid_request openrtb_bid_request js))

RTBJS_SOURCE := \
	rtb_js.cc \
	rtb_router_js.cc \
	auction_js.cc \
	bidding_agent_js.cc \
	banker_js.cc \
	bids_js.cc \
	win_cost_model_js.cc

$(eval $(call nodejs_addon,rtb,$(RTBJS_SOURCE),rtb_router bidding_agent,bid_request services opstats))
