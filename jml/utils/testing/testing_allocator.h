/* testing_allocator.h                                             -*- C++ -*-
   Jeremy Barnes, 22 March 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Allocator for testing purposes.
*/

#ifndef __jml__utils__testing_allocator_h__
#define __jml__utils__testing_allocator_h__

#include "jml/utils/hash_map.h"
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include <string.h>

namespace ML {

using namespace std;

struct Testing_Allocator_Data {
    
    Testing_Allocator_Data()
        : objects_allocated(0), bytes_allocated(0),
          objects_outstanding(0), bytes_outstanding(0)
    {
    }

    struct Alloc_Info {
        Alloc_Info(size_t size = 0, size_t index = 0)
            : size(size), index(index), freed(false)
        {
        }

        size_t size;
        size_t index;
        bool freed;
    };

    typedef hash_map<void *, Alloc_Info> Info;
    Info info;

    size_t objects_allocated;
    size_t bytes_allocated;
    size_t objects_outstanding;
    size_t bytes_outstanding;

    ~Testing_Allocator_Data()
    {
        if (objects_outstanding != 0 || bytes_outstanding != 0) {
            dump();
            throw Exception("destroyed allocated with outstanding objects");
        }

        for (Info::iterator it = info.begin(), end = info.end();
             it != end;  ++it) {
            if (!it->second.freed)
                throw Exception("memory not freed");
            free(it->first);
        }
    }

    void * allocate(size_t bytes)
    {
        void * mem = malloc(bytes);
        if (!mem)
            throw Exception("couldn't allocate memory");
        if (info.count(mem))
            throw Exception("memory was allocated twice");

        info[mem] = Alloc_Info(bytes, info.size() - 1);

        ++objects_allocated;
        ++objects_outstanding;
        bytes_allocated += bytes;
        bytes_outstanding += bytes;

        memset(mem, -1, bytes);

        return mem;
    }

    void deallocate(void * ptr, size_t bytes)
    {
        if (!info.count(ptr))
            throw Exception("free of unknown memory");

        Alloc_Info & ainfo = info[ptr];

        if (ainfo.freed)
            throw Exception("double-free");

        if (ainfo.size != bytes) {
            cerr << "bytes = " << bytes << endl;
            cerr << "ainfo.size = " << ainfo.size << endl;
            throw Exception("free of wrong size");
        }

        ainfo.freed = true;
        --objects_outstanding;
        bytes_outstanding -= ainfo.size;

        memset(ptr, -1, ainfo.size);
    }

    void dump() const
    {
        cerr << "objects: allocated " << objects_allocated
             << " outstanding: " << objects_outstanding << endl;
        cerr << "bytes: allocated " << bytes_allocated
             << " outstanding: " << bytes_outstanding << endl;

        for (Info::const_iterator it = info.begin(), end = info.end();
             it != end;  ++it) {
            cerr << format("  %012p %8d %8zd %s",
                           it->first,
                           it->second.index,
                           it->second.size,
                           (it->second.freed ? "" : "LIVE"))
                 << endl;
        }
    }
};

struct Testing_Allocator {
    Testing_Allocator(Testing_Allocator_Data & data)
        : data(data)
    {
    }

    Testing_Allocator_Data & data;
    
    void * allocate(size_t bytes)
    {
        return data.allocate(bytes);
    }
    
    void deallocate(void * ptr, size_t bytes)
    {
        data.deallocate(ptr, bytes);
    }
};

} // namespace ML

#endif /* __jml__utils__testing_allocator_h__ */
