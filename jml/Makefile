# Makefile for Jeremy's Machine Learning library
# Copyright (c) 2006 Jeremy Barnes.  All rights reserved.

FC=gfortran
-include local.mk

LOCAL_LIB_DIR?=$(HOME)/local/lib /usr/local/lib

default: all
.PHONY: default

BUILD 	?= ./build
ARCH 	?= $(shell uname -m)
OBJ 	:= $(BUILD)/$(ARCH)/obj
BIN 	:= $(BUILD)/$(ARCH)/bin
TESTS 	:= $(BUILD)/$(ARCH)/tests
SRC 	:= .
PWD     := $(shell pwd)
TEST_TMP:= $(TESTS)

JML_TOP := .
JML_BUILD := ./jml-build
INCLUDE := -I.

export BUILD
export BIN
export JML_TOP
export JML_BUILD
export TEST_TMP

include $(JML_BUILD)/arch/$(ARCH).mk

include $(JML_BUILD)/functions.mk
include $(JML_BUILD)/rules.mk
include $(JML_BUILD)/python.mk
include $(JML_BUILD)/node.mk

include $(JML_TOP)/jml.mk
