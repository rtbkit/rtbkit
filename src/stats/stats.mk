LIBSTATS_SOURCES := \
        distribution.cc

$(eval $(call add_sources,$(LIBSTATS_SOURCES)))

LIBSTATS_LINK :=	utils

$(eval $(call library,stats,$(LIBSTATS_SOURCES),$(LIBSTATS_LINK)))
