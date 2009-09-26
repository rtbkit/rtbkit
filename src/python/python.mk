# python.mk
# Jeremy Barnes, 24 September 2009
# Copyright (c) 2009 Jeremy Barnes.  All rights reserved.
#
# Makefile for the python wrappers for jml

PYTHON_SOURCES := \
        jml_wrap_python.i

PYTHON_LINK :=	boosting

$(eval $(call library,_jml,$(PYTHON_SOURCES),$(PYTHON_LINK),_jml))
