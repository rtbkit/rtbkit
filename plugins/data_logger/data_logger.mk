# data loggers makefile
# Sunil Rottoo 

LIBRTBKIT_DATA_LOGGER_SOURCES := \
	data_logger.cc

LIBRTBKIT_DATA_LOGGER_LINK := \
	ACE arch utils logger boost_thread zmq opstats services monitor

$(eval $(call library,data_logger,$(LIBRTBKIT_DATA_LOGGER_SOURCES),$(LIBRTBKIT_DATA_LOGGER_LINK)))
