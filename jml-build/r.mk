ifeq ($(R_ENABLED),1)

R_LIBS_DIR ?= r_libs
RSCRIPT ?= Rscript
R_REPO ?= http://cran.skazkaforyou.com/

R_PKG_CMD = x=setdiff(commandArgs(trailingOnly=TRUE),rownames(installed.packages())); if(length(x)) install.packages(x,repos='$(R_REPO)')

r_dependencies: r_requirements.txt
	mkdir -p $(R_LIBS_DIR)
	echo "$(R_PKG_CMD)" | TMP=/tmp R_LIBS=$(R_LIBS_DIR) R_LIBS_USER=$(R_LIBS_DIR) $(RSCRIPT) - `cat r_requirements.txt`

dependencies: r_dependencies


# $(1): name of R program
# $(2): R source file to copy

define r_program
ifneq ($(PREMAKE),1)
$$(if $(trace),$$(warning called r_program "$(1)" "$(2)"))


run_$(1):	$(BIN)/$(1)
	R_LIBS=$(R_LIBS_DIR) R_LIBS_USER=$(R_LIBS_DIR) $(RSCRIPT) $(BIN)/$(1)  $($(1)_ARGS)

$(BIN)/$(1): $(CWD)/$(2) $(BIN)/.dir_exists
	@echo "$(COLOR_BLUE)     [R_PROGRAM]$(COLOR_RESET) $(1)"
	@cp $$< $$@~
	@chmod +x $$@~
	@mv $$@~ $$@

$(1): $(BIN)/$(1)

r_programs: $(1)

all compile:	r_programs
endif
endef

endif # ifeq ($(R_ENABLED),1)
