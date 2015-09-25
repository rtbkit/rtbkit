# logger makefile
# Jeremy Barnes, 20 May 2011


LIBLOGGER_SOURCES := \
	logger.cc remote_output.cc remote_input.cc \
	file_output.cc publish_output.cc \
	filter.cc json_filter.cc stats_output.cc callback_output.cc \
	rotating_output.cc cloud_output.cc compressor.cc compressing_output.cc \
	multi_output.cc 

LIBLOGGER_LINK := \
	ACE arch utils boost_thread boost_regex zeromq endpoint lzma boost_filesystem opstats cloud gc

$(eval $(call library,logger,$(LIBLOGGER_SOURCES),$(LIBLOGGER_LINK)))

$(eval $(call nodejs_addon,logger,logger_js.cc filter_js.cc,logger js sigslot))

LIBLOG_METRICS_SOURCES := \
    kvp_logger_interface.cc easy_kvp_logger.cc logger_metrics_interface.cc \
    logger_metrics_term.cc

LIBLOG_METRICS_LINK := \
    mongoclient boost_filesystem boost_program_options types

$(eval $(call library,log_metrics,$(LIBLOG_METRICS_SOURCES),$(LIBLOG_METRICS_LINK)))

$(eval $(call include_sub_make,js))
$(eval $(call include_sub_make,logger_testing,testing,logger_testing.mk))
