# RTBKIT adserver makefile

LIBADSERVERCONNECTOR_SOURCES := \
	adserver_connector.cc \
	http_adserver_connector.cc

LIBADSERVERCONNECTOR_LINK := \
	zeromq boost_thread utils endpoint services rtb gc

$(eval $(call library,adserver_connector, \
	$(LIBADSERVERCONNECTOR_SOURCES),  \
	$(LIBADSERVERCONNECTOR_LINK)))

$(eval $(call library,mock_adserver,mock_adserver_connector.cc mock_win_source.cc mock_event_source.cc,adserver_connector bid_test_utils))
$(eval $(call library,standard_adserver,standard_adserver_connector.cc standard_win_source.cc standard_event_source.cc,adserver_connector bid_test_utils))
$(eval $(call program,adserver_runner,adserver_connector boost_program_options services))

$(eval $(call include_sub_make,adserver_testing,testing,adserver_testing.mk))

