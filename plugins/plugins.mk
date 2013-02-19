# RTBKIT plugins makefile

$(eval $(call include_sub_make,bidding_agent))
$(eval $(call include_sub_make,rtb_augmentor,augmentor,rtb_augmentor.mk))
$(eval $(call include_sub_make,data_logger))
$(eval $(call include_sub_make,exchange))
