# RTBKIT plugins makefile

$(eval $(call include_sub_make,bidding_agent))
$(eval $(call include_sub_make,rtb_augmentor,augmentor,rtb_augmentor.mk))
$(eval $(call include_sub_make,data_logger))
$(eval $(call include_sub_make,bid_request))
$(eval $(call include_sub_make,bidder_interface))
$(eval $(call include_sub_make,exchange))
$(eval $(call include_sub_make,adserver))
$(eval $(call include_sub_make,analytics))
