$(eval $(call nodejs_addon,bid_request,bid_request_js.cc,bid_request js))

$(eval $(call nodejs_addon,rtb,rtb_js.cc rtb_router_js.cc auction_js.cc bidding_agent_js.cc banker_js.cc,rtb_router bidding_agent,bid_request services opstats))

$(eval $(call nodejs_addon,config_validator,config_validator_js.cc,rtb_router))
