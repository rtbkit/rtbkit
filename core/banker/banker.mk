# RTBKIT banker makefile

LIBBANKER_SOURCES := \
	account.cc \
	banker.cc \
	null_banker.cc \
	slave_banker.cc \
	master_banker.cc \

LIBBANKER_LINK := \
	types services redis monitor

$(eval $(call library,banker,$(LIBBANKER_SOURCES),$(LIBBANKER_LINK)))

$(eval $(call program,banker_service_runner,banker boost_program_options))

$(eval $(call include_sub_make,migration,migration,migration.mk))
$(eval $(call include_sub_make,banker_testing,testing,banker_testing.mk))
