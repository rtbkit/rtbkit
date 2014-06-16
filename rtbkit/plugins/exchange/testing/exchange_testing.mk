# exchange_testing.mk

$(eval $(call test,rubicon_exchange_connector_test,rubicon_exchange bid_test_utils openrtb_exchange openrtb_bid_request bidding_agent rtb_router cairomm-1.0 cairo sigc-2.0,boost manual))
$(eval $(call test,gumgum_exchange_connector_test,gumgum_exchange bid_test_utils openrtb_bid_request bidding_agent rtb_router cairomm-1.0 cairo sigc-2.0,boost))
$(eval $(call test,bidswitch_exchange_connector_adx_test,bidswitch_exchange bid_test_utils openrtb_bid_request bidding_agent rtb_router cairomm-1.0 cairo sigc-2.0,boost))
$(eval $(call test,bidswitch_exchange_connector_test,bidswitch_exchange bid_test_utils openrtb_bid_request bidding_agent rtb_router cairomm-1.0 cairo sigc-2.0,boost))
$(eval $(call test,nexage_exchange_connector_test,nexage_exchange bid_test_utils openrtb_bid_request bidding_agent rtb_router cairomm-1.0 cairo sigc-2.0,boost))
$(eval $(call test,adx_exchange_connector_test,adx_exchange bid_test_utils bidding_agent rtb_router,boost))
$(eval $(call test,openrtb_exchange_connector_test,openrtb_exchange bid_test_utils bidding_agent rtb_router,boost))
$(eval $(call test,rtbkit_exchange_connector_test,rtbkit_exchange bid_test_utils bidding_agent rtb_router,boost))
