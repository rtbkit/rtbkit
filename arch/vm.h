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
#include "jml/arch/exception.h"

namespace ML {


enum {
    page_shift       = 12,
    page_size        = 1 << page_shift,
    page_offset_mask = page_size - 1
};

static const size_t page_num_mask = ~(size_t)page_offset_mask;

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

inline std::ostream &
operator << (std::ostream & stream, const Pagemap_Entry & entry)
{
    return stream << entry.print();
}

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
    size_t v = (size_t)value & page_num_mask;
    return (X *)v;
}

template<typename X>
bool is_page_aligned(const X * p)
{
    return ((size_t)p & page_offset_mask) == 0;
}

template<typename X>
size_t to_page_num(const X * ptr)
{
    return (size_t)ptr >> page_shift;
}

inline ssize_t to_page_num(ssize_t val)
{
    return val >> page_shift;
}

void dump_maps(std::ostream & stream = std::cerr);


/*****************************************************************************/
/* PAGEMAP_READER                                                            */
/*****************************************************************************/

/** An object that can maintain information about virtual memory mappings
    for a given virtual memory range.

    Requires Linux with /proc/self/pagemaps support.  This was added in
    Linux 2.6.25.

    See http://www.mjmwired.net/kernel/Documentation/vm/pagemap.txt
*/

struct Pagemap_Reader {
    // Set up and read the pagemap file for the given memory region.  If
    // entries is passed in, that will be used as the temporary buffer.
    // If fd is passed in, then that must be an open file pointing to the
    // /proc/<pid>/pagemap file of the process that we are interested in.
    Pagemap_Reader(const char * mem, size_t bytes,
                   Pagemap_Entry * entries = 0, int fd = -1);
    
    ~Pagemap_Reader();
    
    // Re-read the entries for the given address range in case they have
    // changed.  Returns the number of pages that have changed.
    template<typename X>
    size_t update(const X * addr, size_t bytes_to_update)
    {
        return update((const char *)addr, (const char *)addr + bytes_to_update);
    }
    
    template<typename X>
    size_t update(const X * addr, const X * end)
    {
        if (addr == end) return 0;
        size_t base = to_page_num(this->mem);
        size_t first_page = to_page_num(addr) - base;
        size_t last_page  = to_page_num(end)  - base;
        if (end != page_start(end)) ++last_page;
        return update(first_page, last_page);
    }

    size_t update(ssize_t first_page = 0, ssize_t last_page = -1);

    // Return the entry for the given address
    template<typename X>
    const Pagemap_Entry & operator [] (const X * addr) const
    {
        size_t base = to_page_num(this->mem);
        size_t page = to_page_num(addr) - base;

        return operator [] (page);
    }

    // Return the entry given an index from zero to npages
    const Pagemap_Entry & operator [] (size_t page_index) const
    {
        if (page_index >= npages)
            throw Exception("Pagemap_Reader::operator [](): bad page index");
        return entries[page_index];
    }

    size_t num_pages() const { return npages; }

    const void * mem_start() const { return mem; }
    const void * mem_end() const { return mem + npages * page_size; }

    const Pagemap_Entry * begin() const { return entries; }
    const Pagemap_Entry * end() const { return entries + npages; }

    void dump(std::ostream & stream) const;

private:
    int fd;
    const char * mem;
    size_t npages;
    Pagemap_Entry * entries;
    bool delete_entries;
    bool close_fd;
};

inline std::ostream &
operator << (std::ostream & stream, const Pagemap_Reader & reader)
{
    reader.dump(stream);
    return stream;
}

} // namespace ML

#endif /* __jml__arch__vm_h__ */
