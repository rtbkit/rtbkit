# Make rules to build the leveldb files

INCLUDE_LEVELDB := $(INCLUDE)/leveldb

LIBRECOSET_LEVELDB_SOURCES := \
	./db/builder.cc \
	./db/c.cc \
	./db/db_impl.cc \
	./db/db_iter.cc \
	./db/filename.cc \
	./db/dbformat.cc \
	./db/log_reader.cc \
	./db/log_writer.cc \
	./db/memtable.cc \
	./db/repair.cc \
	./db/table_cache.cc \
	./db/version_edit.cc \
	./db/version_set.cc \
	./db/write_batch.cc \
	./port/port_posix.cc \
	./table/block.cc \
	./table/block_builder.cc \
	./table/format.cc \
	./table/iterator.cc \
	./table/merger.cc \
	./table/table.cc \
	./table/table_builder.cc \
	./table/two_level_iterator.cc \
	./util/arena.cc \
	./util/cache.cc \
	./util/coding.cc \
	./util/comparator.cc \
	./util/crc32c.cc \
	./util/env.cc \
	./util/env_posix.cc \
	./util/hash.cc \
	./util/histogram.cc \
	./util/logging.cc \
	./util/options.cc \
	./util/status.cc

$(eval $(call set_compile_option,$(LIBRECOSET_LEVELDB_SOURCES),$(INCLUDE_LEVELDB) -DOS_LINUX -DPLATFORM=OS_LINUX -DSNAPPY=1 -DLEVELDB_PLATFORM_POSIX))

LIBRECOSET_LEVELDB_LINK := snappy

$(eval $(call library,leveldb,$(LIBRECOSET_LEVELDB_SOURCES),$(LIBRECOSET_LEVELDB_LINK)))

