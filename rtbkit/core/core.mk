# RTBKIT core makefile

$(eval $(call include_sub_make,monitor))
$(eval $(call include_sub_make,banker))
$(eval $(call include_sub_make,agent_configuration))
$(eval $(call include_sub_make,post_auction))
$(eval $(call include_sub_make,rtb_router,router,rtb_router.mk))
