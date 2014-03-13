#------------------------------------------------------------------------------#
# utils.mk
# Rémi Attab, 26 Jul 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# Makefile of soa's misc utilities.
#------------------------------------------------------------------------------#


LIB_TEST_UTILS_SOURCES := \
        fixtures.cc \
        threaded_test.cc

LIB_TEST_UTILS_LINK := \
	arch utils boost_filesystem boost_thread

$(eval $(call library,test_utils,$(LIB_TEST_UTILS_SOURCES),$(LIB_TEST_UTILS_LINK)))

$(eval $(call library,variadic_hash,variadic_hash.cc,cityhash))

$(eval $(call include_sub_make,testing))

