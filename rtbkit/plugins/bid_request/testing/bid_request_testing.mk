# bid_request_testing.mk

$(eval $(call test,openrtb_bid_request_test,openrtb_bid_request,boost))
$(eval $(call test,appnexus_bid_request_test,appnexus_bid_request,boost))
$(eval $(call test,fbx_bid_request_test,fbx_bid_request,boost))
