CXX ?= g++
CC ?= gcc
FC := gfortran
NODEJS_ENABLED:=1
PYTHON_ENABLED:=1

LOCAL_DIR?=$(HOME)/local
NODE_PREFIX:=$(LOCAL_DIR)
VOWS?=./node_modules/vows/bin/vows
COFFEE?=./node_modules/coffee-script/bin/coffee
LOCAL_LIB_DIR?=$(LOCAL_DIR)/lib /usr/local/lib
LOCAL_INCLUDE_DIR?=$(LOCAL_DIR)/include

MACHINE_NAME:=$(shell uname -n)

-include local.mk
VIRTUALENV ?= $(LOCAL_DIR)/platform_virtualenv
PYTHON ?= $(VIRTUALENV)/bin/python
PIP ?= $(VIRTUALENV)/bin/pip

default: all
.PHONY: default

BUILD   ?= build
ARCH    ?= $(shell uname -m)
OBJ     := $(BUILD)/$(ARCH)/obj
BIN     := $(BUILD)/$(ARCH)/bin
TESTS   := $(BUILD)/$(ARCH)/tests
SRC     := .
TMP     ?= $(BUILD)/$(ARCH)/tmp
TEST_TMP := $(TESTS)

JML_BUILD := jml-build
JML_TOP := jml
INCLUDE := -I.

export JML_TOP
export BIN
export BUILD
export TEST_TMP
export TMP

include $(JML_BUILD)/arch/$(ARCH).mk

CXX_VERSION?=$(shell g++ --version | head -n1 | sed 's/.* //g')

CFLAGS += -fno-strict-overflow -msse4.2

CXXFLAGS += -Wno-deprecated -Wno-uninitialized -Winit-self -fno-omit-frame-pointer -std=c++0x -fno-deduce-init-list -I$(NODE_PREFIX)/include/node -msse3 -Wno-unused-but-set-variable -I$(LOCAL_INCLUDE_DIR) -I/usr/local/include -Wno-psabi -D__GXX_EXPERIMENTAL_CXX0X__=1
CXXLINKFLAGS += -Wl,--copy-dt-needed-entries -Wl,--no-as-needed -L/usr/local/lib
CFLAGS +=  -Wno-unused-but-set-variable

VALGRINDFLAGS := --suppressions=valgrind.supp --error-exitcode=1 --leak-check=full

$(if $(findstring x4.5,x$(CXX_VERSION)),$(eval CXXFLAGS += -Dnoexcept= -Dnullptr=NULL))

include $(JML_BUILD)/functions.mk
include $(JML_BUILD)/rules.mk
include $(JML_BUILD)/node.mk
include $(JML_BUILD)/python.mk
include $(JML_BUILD)/tcmalloc.mk

SUBDIRS := jml tinyxml2 googleurl soa

PREMAKE := 1

$(eval $(call include_sub_makes,$(SUBDIRS)))

PREMAKE := 0

$(eval $(call include_sub_makes,$(SUBDIRS)))
