ifeq ($(TCMALLOC_ENABLED),1)

MEMORY_ALLOC_LIBRARY?=tcmalloc
CXXFLAGS += -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
CFLAGS += -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free


endif
