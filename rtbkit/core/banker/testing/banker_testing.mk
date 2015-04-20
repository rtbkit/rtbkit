# Banker testing makefile
# Jeremy Barnes, 19 October 2012

# libmockbankerpersistence.so
LIBMOCKBANKERPERSISTENCE_SOURCES := \
	mock_banker_persistence.cc

LIBMOCKBANKERPERSISTENCE_LINK := \
	banker

$(eval $(call library,mock_banker_persistence, \
	$(LIBMOCKBANKERPERSISTENCE_SOURCES), \
	$(LIBMOCKBANKERPERSISTENCE_LINK)))

# libbanker_temporary_server.so
LIBBANKERTEMPORARYSERVER_SOURCES := \
	banker_temporary_server.cc

LIBBANKERTEMPORARYSERVER_LINK := \
	banker

$(eval $(call library,banker_temporary_server, \
	$(LIBBANKERTEMPORARYSERVER_SOURCES), \
	$(LIBBANKERTEMPORARYSERVER_LINK)))

#$(eval $(call test,redis_banker_test,banker,boost))
#$(eval $(call test,redis_banker_race_test,banker,boost))
#$(eval $(call test,redis_banker_deadlock_test,banker,boost))
$(eval $(call test,master_banker_test,banker mock_banker_persistence,boost))
$(eval $(call test,slave_banker_test,banker mock_banker_persistence,boost manual))
$(eval $(call test,banker_account_test,banker,boost))
$(eval $(call test,banker_behaviour_test,banker banker_temporary_server,boost manual))
$(eval $(call test,redis_persistence_test,banker,boost))
$(eval $(call test,local_banker_test,gobanker banker,boost manual))

banker_tests: master_banker_test slave_banker_test banker_account_test banker_behaviour_test redis_persistence_test
