# exchange_testing.mk

$(eval $(call test,rubicon_exchange_connector_test,rubicon_exchange openrtb_bid_request bidding_agent rtb_router cairomm-1.0 cairo sigc-2.0,boost manual))