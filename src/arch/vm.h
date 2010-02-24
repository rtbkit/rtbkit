/* vm.h                                                            -*- C++ -*-
   Jeremy Barnes, 22 February 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Virtual memory functions.
*/

#ifndef __jml__arch__vm_h__
#define __jml__arch__vm_h__

#include <stdint.h>
#include <iostream>
#include <vector>


namespace ML {


enum { page_size = 4096 };

struct Pagemap_Entry {
    Pagemap_Entry(uint64_t mapping = 0)
        : mapping(mapping)
    {
    }

    // Note that this just looks at the pfn.  The other flags might change
    // even if it's at the same place.
    bool operator == (const Pagemap_Entry & other) const
    {
        return pfn == other.pfn;
    }

    bool operator != (const Pagemap_Entry & other) const
    {
        return ! operator == (other);
    }

    union {
        struct {
            uint64_t pfn:55;
            uint64_t shift:6;
            uint64_t reserved:1;
            uint64_t swapped:1;
            uint64_t present:1;
        };
        struct {
            uint64_t swap_type:5;
            uint64_t swap_offset:50;
            uint64_t shift_:6;
            uint64_t reserved_:1;
            uint64_t swapped_:1;
            uint64_t present_:1;
        };
        uint64_t mapping;
    };

    std::string print() const;
};


struct Page_Info : Pagemap_Entry {
    Page_Info()
        : count(0), flags(0)
    {
    }

    // Note that this just looks at the pfn.  The other flags might change
    // even if it's at the same place.
    bool operator == (const Page_Info & other) const
    {
        return pfn == other.pfn;
    }

    bool operator != (const Page_Info & other) const
    {
        return ! operator == (other);
    }

    inline std::string print_mapping() const;

    uint64_t count;

    union {
        struct {
            uint64_t locked:1;
            uint64_t error:1;
            uint64_t referenced:1;
            uint64_t uptodate:1;
            uint64_t dirty:1;
            uint64_t lru:1;
            uint64_t active:1;
            uint64_t slab:1;
            uint64_t writeback:1;
            uint64_t reclaim:1;
            uint64_t buddy:1;
            uint64_t mmap:1;
            uint64_t anon:1;
            uint64_t swapbacked:1;
            uint64_t compound_head:1;
            uint64_t compound_tail:1;
            uint64_t huge:1;
            uint64_t unevictable:1;
            uint64_t hwpoison:1;
            uint64_t nopage:1;
            uint64_t ksm:1;
            uint64_t unused:42;
        };
        uint64_t flags;
    };

    std::string print_flags() const;
    std::string print() const;
};

inline std::ostream &
operator << (std::ostream & stream, const Page_Info & info)
{
    return stream << info.print();
}

std::vector<Page_Info> page_info(const void * addr, int npages);

std::vector<unsigned char> page_flags(const void * addr, int npages);

// Dump the page info for all of the pages in the range
void dump_page_info(const void * start, const void * end,
                    std::ostream & stream = std::cerr);

inline void dump_page_info(const void * start, size_t sz,
                           std::ostream & stream = std::cerr)
{
    const char * end = (const char *)start + sz;
    return dump_page_info(start, end, stream);
}

template<typename X>
X * page_start(X * value)
{
    size_t v = (size_t)value;
    v = v & ~((size_t)(page_size - 1));
    return (X *)v;
}

void dump_maps(std::ostream & stream = std::cerr);


} // namespace ML

#endif /* __jml__arch__vm_h__ */
