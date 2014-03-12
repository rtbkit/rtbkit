# libredis_migration.so
LIBREDISMIGRATION_SOURCES := \
	redis_migration.cc \
	redis_old_types.cc \
	redis_rollback.cc \
	redis_utils.cc

LIBREDISMIGRATION_LINK := \
	banker \
	redis

$(eval $(call library,redis_migration, \
	$(LIBREDISMIGRATION_SOURCES), \
	$(LIBREDISMIGRATION_LINK)))

$(eval $(call program,redis_migrate,redis_migration boost_program_options))
