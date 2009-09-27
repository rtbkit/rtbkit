# python.mk
# Jeremy Barnes, 24 September 2009
# Copyright (c) 2009 Jeremy Barnes.  All rights reserved.
#
# Makefile for the python wrappers for jml

PYTHON_SOURCES := \
        jml_wrap_python.i

PYTHON_LINK :=	boosting

# The next line is necessary to stop the following error with SWIG 1.3.36 and
# g++ 4.3.5.  It's not really a problem, but the compiler gets confused and
# since we have -Werror, we can't compile.
# ../build/x86_64/obj/python/jml_wrap_python_wrap.cxx: In function ‘PyObject* _wrap_new_Classifier(PyObject*, PyObject*)’:
# ../build/x86_64/obj/python/jml_wrap_python_wrap.cxx:7193: error: ‘argv[0]’ may be used uninitialized in this function
OPTIONS_python/jml_wrap_python_wrap.cxx := -Wno-uninitialized

$(eval $(call library,_jml,$(PYTHON_SOURCES),$(PYTHON_LINK),_jml))
