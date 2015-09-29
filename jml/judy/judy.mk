LIBJUDY_SOURCES := \
        JudyLByCount.cc \
        JudyLCreateBranch.cc \
        JudyLCascade.cc \
        JudyLCount.cc \
        judy_malloc_allocator.cc \
        JudyLDecascade.cc \
        JudyLDel.cc \
        JudyLFirst.cc \
        JudyLFreeArray.cc \
        JudyLGet.cc \
        JudyLIns.cc \
        JudyLInsArray.cc \
        JudyLInsertBranch.cc \
        JudyLMallocIF.cc \
        JudyLMemActive.cc \
        JudyLMemUsed.cc \
        JudyLNext.cc \
        JudyLNextEmpty.cc \
        JudyLPrev.cc \
        JudyLPrevEmpty.cc \
        JudyLTables.cc \
        JudyLTablesGen.cc \
        j__udyLGet.cc

LIBJUDY_LINK :=

$(eval $(call set_compile_option,$(LIBJUDY_SOURCES),-fno-strict-aliasing))

$(eval $(call library,judy,$(LIBJUDY_SOURCES),$(LIBJUDY_LINK)))
