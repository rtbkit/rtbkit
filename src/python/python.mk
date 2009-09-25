# python.mk
# Jeremy Barnes, 24 September 2009
# Copyright (c) 2009 Jeremy Barnes.  All rights reserved.
#
# Makefile for the python wrappers for jml

PYTHON_SOURCES := \
        classifier.i

PYTHON_LINK :=	ACE arch boost_iostreams-mt

$(eval $(call library,python,$(PYTHON_SOURCES),$(PYTHON_LINK)))

