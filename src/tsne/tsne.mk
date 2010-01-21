# Makefile for tsne functionality
# Jeremy Barnes, 16 January 2010
# Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

LIBTSNE_SOURCES := \
        tsne.cc \

LIBTSNE_LINK :=	utils algebra arch boost_thread-mt stats

$(eval $(call library,tsne,$(LIBTSNE_SOURCES),$(LIBTSNE_LINK)))

ifeq ($(PYTHON_ENABLED),1)


TSNE_PYTHON_SOURCES := \
	tsne_python.cc

$(eval $(call set_compile_option,$(TSNE_PYTHON_SOURCES),-I$(PYTHON_INCLUDE_PATH)))

TSNE_PYTHON_LINK := tsne

$(eval $(call library,tsne_python,$(TSNE_PYTHON_SOURCES),$(TSNE_PYTHON_LINK),_tsne))

endif # PYTHON_ENABLED

$(eval $(call include_sub_make,tsne_testing,testing))
