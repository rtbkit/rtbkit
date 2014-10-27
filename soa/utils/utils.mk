#------------------------------------------------------------------------------#
# utils.mk
# Rémi Attab, 26 Jul 2013
# Copyright (c) 2013 Datacratic.  All rights reserved.
#
# Makefile of soa's misc utilities.
#------------------------------------------------------------------------------#

LIB_TEST_UTILS_SOURCES := \
        benchmarks.cc \
        fixtures.cc \
        threaded_test.cc

LIB_TEST_UTILS_LINK := \
	arch utils boost_filesystem boost_thread

$(eval $(call library,test_utils,$(LIB_TEST_UTILS_SOURCES),$(LIB_TEST_UTILS_LINK)))

$(eval $(call library,variadic_hash,variadic_hash.cc,cityhash))
$(eval $(call library,string_encryption,string_encryption.cc,crypto++))
$(eval $(call program,string_encryption_keygen,string_encryption))

ifeq ($(PYTHON_ENABLED),1)

$(eval $(call library,py_util,py.cc,boost_python python$(PYTHON_VERSION)))
$(eval $(call set_compile_option,py.cc,-I$(PYTHON_INCLUDE_PATH)))

endif # PYTHON_ENABLED

$(eval $(call include_sub_make,testing))

