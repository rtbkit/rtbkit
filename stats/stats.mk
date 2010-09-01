LIBSTATS_SOURCES := \
        distribution.cc \
	auc.cc

$(eval $(call add_sources,$(LIBSTATS_SOURCES)))

LIBSTATS_LINK :=	utils

$(eval $(call library,stats,$(LIBSTATS_SOURCES),$(LIBSTATS_LINK)))

$(eval $(call include_sub_make,stats_testing,testing))
