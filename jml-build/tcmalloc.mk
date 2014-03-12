ifeq ($(TCMALLOC_ENABLED),1)

MALLOC_LIBRARY?=-ltcmalloc
CXXFLAGS += -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
CFLAGS += -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free


endif
