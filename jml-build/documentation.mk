ifeq ($(DOCUMENTATION_ENABLED),1)

# copy documentation
# copies ./$(1) -> $(DOC)/$(1)
# $(1): file name

define copy_doc

ifneq ($(PREMAKE),1)

$(DOC)/$(1): $(CWD)/$(1)
	@mkdir -p $(DOC)
	cp $(CWD)/$(1) $(DOC)/$(1)

documentation: $(DOC)/$(1)

endif
endef

# generate literate API documentation
# results in transformation ./$(1).md -> $(DOC)/$(1).html
# $(1): target name
# $(2): targets this target depends on (i.e. command this literate doc runs)

define literate_api_doc

ifneq ($(PREMAKE),1)

$(DOC)/$(1).html: $(BIN)/literate_api_doc $(CWD)/$(1).md $(2)
	@mkdir -p $(DOC)
	$(PYTHON) $(BIN)/literate_api_doc $(CWD)/$(1).md $(DOC)/$(1).html

documentation: $(DOC)/$(1).html

endif
endef


endif # ifeq ($(DOCUMENTATION_ENABLED),1)
