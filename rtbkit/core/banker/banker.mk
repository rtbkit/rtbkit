# RTBKIT banker makefile

LIBBANKER_SOURCES := \
	account.cc \
	banker.cc \
	null_banker.cc \
	slave_banker.cc \
	master_banker.cc \
	application_layer.cc

LIBBANKER_LINK := \
	types services redis monitor boost_program_options

$(eval $(call library,banker,$(LIBBANKER_SOURCES),$(LIBBANKER_LINK)))

$(eval $(call program,banker_service_runner,banker boost_program_options))

$(eval $(call library,gobanker,go_account.cc local_banker.cc))

$(eval $(call python_program,banker_backup,banker_backup.py))
$(eval $(call python_program,banker_restore,banker_restore.py))

$(eval $(call include_sub_make,migration,migration,migration.mk))
$(eval $(call include_sub_make,banker_testing,testing,banker_testing.mk))
