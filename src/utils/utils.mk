LIBUTILS_SOURCES := \
        environment.cc \
        file_functions.cc \
        filter_streams.cc \
        string_functions.cc \
        parse_context.cc \
	info.cc \
	configuration.cc

LIBUTILS_LINK :=	ACE arch boost_iostreams-mt

$(eval $(call library,utils,$(LIBUTILS_SOURCES),$(LIBUTILS_LINK)))

$(eval $(call include_sub_make,utils_testing,testing))
