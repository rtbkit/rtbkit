include $(JML_BUILD)/gmsl/gmsl

TMPBIN ?= $(BIN)
LIB ?= $(BIN)

dollars=$$

SHELL := /bin/bash

ifeq ($(TERM),xterm)

ESC :=

COLOR_RED :=$(ESC)[31m
COLOR_GREEN :=$(ESC)[32m
COLOR_YELLOW :=$(ESC)[33m
COLOR_BLUE :=$(ESC)[34m
COLOR_VIOLET :=$(ESC)[35m
COLOR_CYAN :=$(ESC)[36m
COLOR_RESET := $(ESC)[0m
COLOR_BOLD :=$(ESC)[1m

endif

ifeq ($(TERM),rxvt)

ESC :=

COLOR_RED :=$(ESC)[31m
COLOR_GREEN :=$(ESC)[32m
COLOR_YELLOW :=$(ESC)[33m
COLOR_BLUE :=$(ESC)[34m
COLOR_VIOLET :=$(ESC)[35m
COLOR_CYAN :=$(ESC)[36m
COLOR_RESET := $(ESC)[0m
COLOR_BOLD :=$(ESC)[1m

endif

-include .make_hash_cache

# Command to hash the name of a command.
NOTHING :=
SPACE := $(NOTHING) $(NOTHING)

hash_command2 = $(wordlist 1,1,$(shell echo $(strip $(1)) | md5sum))

hash_command1 = $(eval HASH:=$(call hash_command2,$(1)))$(shell echo $(1)_hash:=$(HASH) >> .make_hash_cache)$(eval $(1)_hash:=$(HASH))

command_key = $(subst =,_,$(subst $(SPACE),_,$(strip $(1))))

hash_command = $(eval KEY=$(call command_key,$(1)))$(if $($(KEY)_hash),,$(call hash_command1,$(KEY)))$(if $($(KEY)_hash),,$(error hash_command1 didnt set variable $(KEY)_hash))$($(KEY)_hash)

# arg 1: names
define include_sub_makes
$$(foreach name,$(1),$$(eval $$(call include_sub_make,$$(name))))
endef

# arg 1: name
# arg 2: dir (optional, is the same as $(1) if not given)
# arg 3: makefile (optional, is $(2)/$(1).mk if not given)
define include_sub_make
$(if $(trace3),$$(warning called include_sub_make "$(1)" "$(2)" "$(3)" CWD=$(CWD)))
DIRNAME:=$(if $(2),$(2),$(1))
MAKEFILE:=$(if $(3),$(3),$(1).mk)
$$(call push,DIRS,$$(call peek,DIRS)$$(if $$(call peek,DIRS),/,)$$(DIRNAME))
CWD:=$$(call peek,DIRS)
$$(call push,MKPATH,$(1))
CURRENT:=$$(subst _testing,,$(1))
#CURRENT_TEST_TARGETS:=$$(if $$(findstring,xtestingx,$(1)),$$(CURRENT_TEST_TARGETS),$$(CURRENT_TEST_TARGETS $(1)_test))
include $$(if $$(CWD),$$(CWD)/,)/$$(MAKEFILE)
$$(CWD_NAME)_SRC :=	$(SRC)/$$(CWD)
$$(CWD_NAME)_OBJ :=	$(OBJ)/$$(CWD)
#$$(warning stack contains $(__gmsl_stack_DIRS))
CWD:=$$(call pop,DIRS)
CURRENT:=$$(call pop,MKPATH)
CURRENT_TEST_TARGETS := 
endef

# add a c++ source file
# $(1): filename of source file
# $(2): basename of the filename
# $(3): directory under which the source lives; default $(SRC)
# $(4): extra compiler options

define add_c++_source
ifneq ($(PREMAKE),1)

$$(eval tmpDIR := $$(if $(3),$(3),$(SRC)))

$(if $(trace),$$(warning called add_c++_source "$(1)" "$(2)" "$(3)" "$(4)"))
BUILD_$(CWD)/$(2).lo_COMMAND:=$$(CXX) $$(CXXFLAGS) -o $(OBJ)/$(CWD)/$(2).lo -c $$(tmpDIR)/$(CWD)/$(1) -MP -MMD -MF $(OBJ)/$(CWD)/$(2).d -MQ $(OBJ)/$(CWD)/$(2).lo $$(OPTIONS_$(CWD)/$(1)) $(4) $(if $(findstring $(strip $(1)),$(DEBUG_FILES)),$(warning compiling $(1) for debug)$$(CXXDEBUGFLAGS),$$(CXXNODEBUGFLAGS))
$(if $(trace),$$(warning BUILD_$(CWD)/$(2).lo_COMMAND := "$$(BUILD_$(CWD)/$(2).lo_COMMAND)"))

BUILD_$(CWD)/$(2).lo_HASH := $$(call hash_command,$$(BUILD_$(CWD)/$(2).lo_COMMAND))
BUILD_$(CWD)/$(2).lo_OBJ  := $$(OBJ)/$(CWD)/$(2).$$(BUILD_$(CWD)/$(2).lo_HASH).lo

BUILD_$(CWD)/$(2).lo_COMMAND2 := $$(subst $(OBJ)/$(CWD)/$(2).lo,$$(BUILD_$(CWD)/$(2).lo_OBJ),$$(BUILD_$(CWD)/$(2).lo_COMMAND))

$(OBJ)/$(CWD)/$(2).d:
$$(BUILD_$(CWD)/$(2).lo_OBJ):	$$(tmpDIR)/$(CWD)/$(1) $(OBJ)/$(CWD)/.dir_exists $$(dir $$(OBJ)/$(CWD)/$(2))/.dir_exists
	$$(if $(verbose_build),@echo $$(BUILD_$(CWD)/$(2).lo_COMMAND2),@echo "           $(COLOR_CYAN)[C++]$(COLOR_RESET) $(CWD)/$(1)")
	@$$(BUILD_$(CWD)/$(2).lo_COMMAND2)
	@if [ -f $(2).d ] ; then mv $(2).d $(OBJ)/$(CWD)/$(2).d; fi

-include $(OBJ)/$(CWD)/$(2).d
endif
endef

# add a c source file
# $(1): filename of source file
# $(2): basename of the filename
# $(3): directory under which the source lives; default $(SRC)
# $(4): extra compiler options

define add_c_source
ifneq ($(PREMAKE),1)

$$(eval tmpDIR := $$(if $(3),$(3),$(SRC)))

$(if $(trace),$$(warning called add_c_source "$(1)" "$(2)" "$(3)" "$(4)"))
BUILD_$(CWD)/$(2).lo_COMMAND:=$$(CC) $$(CFLAGS) -o $(OBJ)/$(CWD)/$(2).lo -c $$(tmpDIR)/$(CWD)/$(1) -MP -MMD -MF $(OBJ)/$(CWD)/$(2).d -MQ $(OBJ)/$(CWD)/$(2).lo $$(OPTIONS_$(CWD)/$(1)) $(4) $(if $(findstring $(strip $(1)),$(DEBUG_FILES)),$(warning compiling $(1) for debug)$$(CDEBUGFLAGS),$$(CNODEBUGFLAGS))
$(if $(trace),$$(warning BUILD_$(CWD)/$(2).lo_COMMAND := "$$(BUILD_$(CWD)/$(2).lo_COMMAND)"))

BUILD_$(CWD)/$(2).lo_HASH := $$(call hash_command,$$(BUILD_$(CWD)/$(2).lo_COMMAND))
BUILD_$(CWD)/$(2).lo_OBJ  := $$(OBJ)/$(CWD)/$(2).$$(BUILD_$(CWD)/$(2).lo_HASH).lo

BUILD_$(CWD)/$(2).lo_COMMAND2 := $$(subst $(OBJ)/$(CWD)/$(2).lo,$$(BUILD_$(CWD)/$(2).lo_OBJ),$$(BUILD_$(CWD)/$(2).lo_COMMAND))

$(OBJ)/$(CWD)/$(2).d:
$$(BUILD_$(CWD)/$(2).lo_OBJ):	$$(tmpDIR)/$(CWD)/$(1) $(OBJ)/$(CWD)/.dir_exists
	$$(if $(verbose_build),@echo $$(BUILD_$(CWD)/$(2).lo_COMMAND2),@echo "             $(COLOR_CYAN)[C]$(COLOR_RESET) $(CWD)/$(1)")
	@$$(BUILD_$(CWD)/$(2).lo_COMMAND2)
	@if [ -f $(2).d ] ; then mv $(2).d $(OBJ)/$(CWD)/$(2).d; fi

-include $(OBJ)/$(CWD)/$(2).d
endif
endef

# add a fortran source file
define add_fortran_source
ifneq ($(PREMAKE),1)
$(if $(trace),$$(warning called add_fortran_source "$(1)" "$(2)"))
BUILD_$(CWD)/$(2).lo_COMMAND:=$(FC) $(FFLAGS) -o $(OBJ)/$(CWD)/$(2).lo -c $(SRC)/$(CWD)/$(1)
$(if $(trace),$$(warning BUILD_$(CWD)/$(2).lo_COMMAND := "$$(BUILD_$(CWD)/$(2).lo_COMMAND)"))

BUILD_$(CWD)/$(2).lo_HASH := $$(call hash_command,$$(BUILD_$(CWD)/$(2).lo_COMMAND))
BUILD_$(CWD)/$(2).lo_OBJ  := $$(OBJ)/$(CWD)/$(2).$$(BUILD_$(CWD)/$(2).lo_HASH).lo

BUILD_$(CWD)/$(2).lo_COMMAND2 := $$(subst $(OBJ)/$(CWD)/$(2).lo,$$(BUILD_$(CWD)/$(2).lo_OBJ),$$(BUILD_$(CWD)/$(2).lo_COMMAND))


$(OBJ)/$(CWD)/$(2).d:
$$(BUILD_$(CWD)/$(2).lo_OBJ):	$(SRC)/$(CWD)/$(1) $(OBJ)/$(CWD)/.dir_exists
	$$(if $(verbose_build),@echo $$(BUILD_$(CWD)/$(2).lo_COMMAND2),@echo "       $(COLOR_CYAN)[FORTRAN]$(COLOR_RESET) $(CWD)/$(1)")
	@$$(BUILD_$(CWD)/$(2).lo_COMMAND2)
endif
endef

define add_cuda_source
ifneq ($(PREMAKE),1)
$(if $(trace),$$(warning called add_cuda_source "$(1)" "$(2)"))
$(OBJ)/$(CWD)/$(2).d: $(SRC)/$(CWD)/$(1) $(OBJ)/$(CWD)/.dir_exists
	($(NVCC) $(NVCCFLAGS) -D__CUDACC__ -M $$< | awk 'NR == 1 { print "$$(BUILD_$(CWD)/$(2).lo_OBJ)", "$$@", ":", $$$$3, "\\"; next; } /usr/ { next; } /\/ \\$$$$/ { next; } { files[$$$$1] = 1; print; } END { print("\n"); for (file in files) { printf("%s: \n\n", file); } }') > $$@~
	mv $$@~ $$@

BUILD_$(CWD)/$(2).lo_COMMAND:=$(NVCC) $(NVCCFLAGS) -c -o $(OBJ)/$(CWD)/$(2).lo --verbose $(SRC)/$(CWD)/$(1)
$(if $(trace),$$(warning BUILD_$(CWD)/$(2).lo_COMMAND := "$$(BUILD_$(CWD)/$(2).lo_COMMAND)"))

BUILD_$(CWD)/$(2).lo_HASH := $$(call hash_command,$$(BUILD_$(CWD)/$(2).lo_COMMAND))
BUILD_$(CWD)/$(2).lo_OBJ  := $$(OBJ)/$(CWD)/$(2).$$(BUILD_$(CWD)/$(2).lo_HASH).lo

BUILD_$(CWD)/$(2).lo_COMMAND2 := $$(subst $(OBJ)/$(CWD)/$(2).lo,$$(BUILD_$(CWD)/$(2).lo_OBJ),$$(BUILD_$(CWD)/$(2).lo_COMMAND))


$$(BUILD_$(CWD)/$(2).lo_OBJ):	$(SRC)/$(CWD)/$(1) $(OBJ)/$(CWD)/.dir_exists
	$$(if $(verbose_build),@echo $$(BUILD_$(CWD)/$(2).lo_COMMAND2),@echo "$(COLOR_CYAN)[CUDA]$(COLOR_RESET) $(CWD)/$(1)")
	@$$(BUILD_$(CWD)/$(2).lo_COMMAND2)


-include $(OBJ)/$(CWD)/$(2).d
endif
endef

# add a (google) protobuf source file
# intermediary files will be left in the source directory
# $(1): filename of source file
# $(2): basename of the filename
define add_pbuf_source
ifneq ($(PREMAKE),1)
$(if $(trace),$$(warning called add_pbuf_source "$(1)" "$(2)"))

# Call protoc to generate the source file
BUILD_$(GEN)/$(CWD)/$(2).pb.cc_COMMAND := "protoc -I$(SRC)/$(CWD) --cpp_out=$(GEN)/$(CWD) $(SRC)/$(CWD)/$(1)"

$(GEN)/$(CWD)/$(2).pb.cc:	$(SRC)/$(CWD)/$(1)
	@mkdir -p $(GEN)/$(CWD)
	$$(if $(verbose_build),@echo $$(BUILD_$(GEN)/$(CWD)/$(2).pb.cc),@echo "      $(COLOR_CYAN)[PBUF c++]$(COLOR_RESET) $(CWD)/$(1)")
	@eval $$(BUILD_$(GEN)/$(CWD)/$(2).pb.cc_COMMAND)

# We use the add_c++_source to do most of the work, then simply point
# to the file
$$(eval $$(call add_c++_source,$(2).pb.cc,$(2).pb,$(GEN),-IXX))


# Point to the object file produced by the previous macro
BUILD_$(CWD)/$(2).lo_OBJ  := $$(BUILD_$(CWD)/$(2).pb.lo_OBJ)

-include $(OBJ)/$(CWD)/$(2).d

endif
endef


# Set up the map to map an extension to the name of a function to call
$(call set,EXT_FUNCTIONS,.cc,add_c++_source)
$(call set,EXT_FUNCTIONS,.cpp,add_c++_source)
$(call set,EXT_FUNCTIONS,.c,add_c_source)
$(call set,EXT_FUNCTIONS,.f,add_fortran_source)
$(call set,EXT_FUNCTIONS,.cu,add_cuda_source)
$(call set,EXT_FUNCTIONS,.i,add_swig_source)
$(call set,EXT_FUNCTIONS,.proto,add_pbuf_source)

# add a single source file
# $(1): filename
# $(2): suffix of the filename
define add_source
ifneq ($(PREMAKE),1)
$$(if $(trace),$$(warning called add_source "$(1)" "$(2)"))
$$(if $$(ADDED_SOURCE_$(CWD)_$(1)),,\
    $$(if $$(call defined,EXT_FUNCTIONS,$(2)),\
	$$(eval $$(call $$(call get,EXT_FUNCTIONS,$(2)),$(1),$$(basename $(1))))\
	    $$(eval ADDED_SOURCE_$(CWD)_$(1):=$(true)),\
	$$(error Extension "$(2)" is not known adding source file $(1))))
endif
endef


# add a list of source files
# $(1): list of filenames
define add_sources
ifneq ($(PREMAKE),1)
$$(if $(trace),$$(warning called add_sources "$(1)"))
$$(foreach file,$$(strip $(1)),$$(eval $$(call add_source,$$(file),$$(suffix $$(file)))))
endif
endef

# set compile options for a single source file
# $(1): filename
# $(2): compile option
define set_single_compile_option
OPTIONS_$(CWD)/$(1) += $(2)
#$$(warning setting OPTIONS_$(CWD)/$(1) += $(2))
endef

# set compile options for a given list of source files
# $(1): list of filenames
# $(2): compile option
define set_compile_option
$$(foreach file,$(1),$$(eval $$(call set_single_compile_option,$$(file),$(2))))
endef

# add a library
# $(1): name of the library
# $(2): source files to include in the library
# $(3): libraries to link with
# $(4): output name; default lib$(1)
# $(5): output extension; default .so
# $(6): build name; default SO

define library
ifneq ($(PREMAKE),1)
$$(if $(trace),$$(warning called library "$(1)" "$(2)" "$(3)"))
$$(eval $$(call add_sources,$(2)))

$$(eval tmpLIBNAME := $(if $(4),$(4),lib$(1)))
$$(eval so := $(if $(5),$(5),.so))

LIB_$(1)_BUILD_NAME := $(if $(6),$(6),"            $(COLOR_YELLOW)[SO]$(COLOR_RESET)")

OBJFILES_$(1):=$$(foreach file,$(addsuffix .lo,$(basename $(2:%=$(CWD)/%))),$$(BUILD_$$(file)_OBJ))

LINK_$(1)_COMMAND:=$$(CXX) $$(CXXFLAGS) $$(CXXLIBRARYFLAGS) $$(CXXNODEBUGFLAGS) -o $(LIB)/$$(tmpLIBNAME)$$(so) $$(OBJFILES_$(1)) $$(foreach lib,$(3), -l$$(lib))

LINK_$(1)_HASH := $$(call hash_command,$$(LINK_$(1)_COMMAND))
LIB_$(1)_SO   := $(TMPBIN)/$$(tmpLIBNAME).$$(LINK_$(1)_HASH)$$(so)

-include $(TMPBIN)/$$(tmpLIBNAME)$$(so).version.mk

$(LIB)/$$(tmpLIBNAME)$$(so).version.mk:	$(LIB)/.dir_exists 
	@echo LIB_$(1)_CURRENT_VERSION:=$$(LINK_$(1)_HASH) > $$@


# We need the library so names to stay the same, so we copy the correct one
# into our version
$(LIB)/$$(tmpLIBNAME)$$(so): $$(LIB_$(1)_SO) $$(dir $(LIB)/$$(tmpLIBNAME))/.dir_exists
	@$(RM) $$@
	@ln $$< $$@
	@echo LIB_$(1)_CURRENT_VERSION:=$$(LINK_$(1)_HASH) > $(TMPBIN)/$$(tmpLIBNAME)$$(so).version.mk

LINK_$(1)_COMMAND2 := $$(subst $(LIB)/$$(tmpLIBNAME)$$(so),$$(LIB_$(1)_SO),$$(LINK_$(1)_COMMAND))

LIB_$(1)_FILENAME := $$(tmpLIBNAME)$$(so)

$$(LIB_$(1)_SO):	$$(dir $$(LIB_$(1)_SO))/.dir_exists $$(OBJFILES_$(1)) $$(foreach lib,$(3),$$(LIB_$$(lib)_DEPS))
	$$(if $(verbose_build),@echo $$(LINK_$(1)_COMMAND2),@echo $$(LIB_$(1)_BUILD_NAME) $$(LIB_$(1)_FILENAME))
	@$$(LINK_$(1)_COMMAND2)

LIB_$(1)_DEPS := $(LIB)/$$(tmpLIBNAME)$$(so)

libraries: $(LIB)/$$(tmpLIBNAME)$$(so)
endif
endef


# add a program
# $(1): name of the program
# $(2): libraries to link with
# $(3): name of files to include in the program.  If not included or empty,
#       $(1).cc assumed
# $(4): list of targets to add this program to
define program
ifneq ($(PREMAKE),1)
$$(if $(trace4),$$(warning called program "$(1)" "$(2)" "$(3)"))

$(1)_PROGFILES:=$$(if $(3),$(3),$(1:%=%.cc))

$$(eval $$(call add_sources,$$($(1)_PROGFILES)))

#$$(warning $(1)_PROGFILES = $$($(1)_PROGFILES))

$(1)_OBJFILES:=$$(foreach file,$$(addsuffix .lo,$$(basename $$($(1)_PROGFILES:%=$(CWD)/%))),$$(BUILD_$$(file)_OBJ))

#$$(warning $(1)_OBJFILES = $$($(1)_OBJFILES))
#$$(warning $(1)_PROGFILES = "$$($(1)_PROGFILES)")

LINK_$(1)_COMMAND:=$$(CXX) $$(CXXFLAGS) $$(CXXEXEFLAGS) $$(CXXNODEBUGFLAGS) -o $(BIN)/$(1) -lexception_hook $(MALLOC_LIBRARY) -L$(LIB) -ldl $$($(1)_OBJFILES) $$(foreach lib,$(2), -l$$(lib)) $$(CXXEXEPOSTFLAGS)


$(BIN)/$(1):	$(BIN)/.dir_exists $$($(1)_OBJFILES) $$(foreach lib,$(2),$$(LIB_$$(lib)_DEPS)) $$(if $$(HAS_EXCEPTION_HOOK),$$(LIB)/libexception_hook.so)
	$$(if $(verbose_build),@echo $$(LINK_$(1)_COMMAND),@echo "           $(COLOR_BLUE)[BIN]$(COLOR_RESET) $(1)")
	@$$(LINK_$(1)_COMMAND)

$$(foreach target,$(4) programs,$$(eval $$(target): $(BIN)/$(1)))

$(1): $(BIN)/$(1)
.PHONY:	$(1)

run_$(1): $(BIN)/$(1)
	$(PREARGS) $(BIN)/$(1) $($(1)_ARGS) $(ARGS)

endif
endef

# Options to go before a testing command for a test
# $(1): the options
TEST_PRE_OPTIONS_ = $(w arning TEST_PRE_OPTIONS $(1))$(if $(findstring timed,$(1)),/usr/bin/time )$(if $(findstring valgrind,$(1)),$(VALGRIND) $(VALGRINDFLAGS) )

TEST_PRE_OPTIONS = $(w arning TEST_PRE_OPTIONS $(1) returned $(c all TEST_PRE_OPTIONS_,$(1)))$(call TEST_PRE_OPTIONS_,$(1))

# Build the command for a test
# $(1): the name of the test
# $(2): the command to run
# $(3): the test options
BUILD_TEST_COMMAND = rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && $(call TEST_PRE_OPTIONS,$(3))$(2) > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "           $(1) FAILED" && cat $(TESTS)/$(1).failed && false))


# add a test case
# $(1) name of the test
# $(2) libraries to link with
# $(3) test style.  boost = boost test framework, and options: manual, valgrind
# $(4) testing targets to add it to

define test
ifneq ($(PREMAKE),1)
$$(if $(trace),$$(warning called test "$(1)" "$(2)" "$(3)"))

$$(if $(3),,$$(error test $(1) needs to define a test style))

$$(eval $$(call add_sources,$(1).cc))

$(1)_OBJFILES:=$$(BUILD_$(CWD)/$(1).lo_OBJ)

LINK_$(1)_COMMAND:=$$(CXX) $$(CXXFLAGS) $$(CXXEXEFLAGS) $$(CXXNODEBUGFLAGS) -o $(TESTS)/$(1) -lexception_hook $(MALLOC_LIBRARY) -ldl  $$($(1)_OBJFILES) $$(foreach lib,$(2), -l$$(lib)) $(if $(findstring boost,$(3)), -lboost_unit_test_framework) $$(CXXEXEPOSTFLAGS)

$(TESTS)/$(1):	$(TESTS)/.dir_exists $(TEST_TMP)/.dir_exists  $$($(1)_OBJFILES) $$(foreach lib,$(2),$$(LIB_$$(lib)_DEPS)) $$(if $$(HAS_EXCEPTION_HOOK),$$(LIB)/libexception_hook.so)
	$$(if $(verbose_build),@echo $$(LINK_$(1)_COMMAND),@echo "           $(COLOR_BLUE)[BIN]$(COLOR_RESET) $(1)")
	@$$(LINK_$(1)_COMMAND)

tests:	$(TESTS)/$(1)
$$(CURRENT)_tests: $(TESTS)/$(1)

TEST_$(1)_COMMAND := rm -f $(TESTS)/$(1).{passed,failed} && ((set -o pipefail && $(call TEST_PRE_OPTIONS,$(3))$(TESTS)/$(1) $(TESTS)/$(1) > $(TESTS)/$(1).running 2>&1 && mv $(TESTS)/$(1).running $(TESTS)/$(1).passed) || (mv $(TESTS)/$(1).running $(TESTS)/$(1).failed && echo "                 $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && cat $(TESTS)/$(1).failed && echo "                       $(COLOR_RED)$(1) FAILED$(COLOR_RESET)" && false))

$(TESTS)/$(1).passed:	$(TESTS)/$(1)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "      $(COLOR_VIOLET)[TESTCASE]$(COLOR_RESET) $(1)")
	@$$(TEST_$(1)_COMMAND)
	$$(if $(verbose_build),@echo '$$(TEST_$(1)_COMMAND)',@echo "                 $(COLOR_GREEN)$(1) passed$(COLOR_RESET)")

$(1):	$(TESTS)/$(1)
	$(call TEST_PRE_OPTIONS,$(3))$(TESTS)/$(1)

.PHONY: $(1)

#$$(warning $(1) $$(CURRENT))

$(if $(findstring manual,$(3)),manual,test $(if $(findstring noauto,$(3)),,autotest) ) $(CURRENT_TEST_TARGETS) $$(CURRENT)_test_all $(4):	$(TESTS)/$(1).passed
endif
endef


compile: programs libraries tests
