# Makefile for Jeremy's Machine Learning library
# Copyright (c) 2006 Jeremy Barnes.  All rights reserved.

-include local.mk

default: all
.PHONY: default

BUILD 	?= ../build
ARCH 	?= $(shell uname -m)
OBJ 	:= $(BUILD)/$(ARCH)/obj
BIN 	:= $(BUILD)/$(ARCH)/bin
TESTS 	:= $(BUILD)/$(ARCH)/tests
SRC 	:= .

include arch/$(ARCH).mk

ifeq ($(MAKECMDGOALS),failed)
include .target.mk
failed:
	+make $(FAILED) $(GOALS)
else

include functions.mk
include rules.mk

$(eval $(call include_sub_makes,math arch utils db algebra stats judy boosting))

$(shell echo GOALS := $(MAKECMDGOALS) > .target.mk)
endif
