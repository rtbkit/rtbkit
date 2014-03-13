/* hex_dump.h                                                      -*- C++ -*-
   Jeremy Barnes, 6 October 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Routine to dump memory in hex format.
*/

#include "hex_dump.h"
#include <iostream>
#include "string_functions.h"

using namespace std;

namespace ML {

void hex_dump(const void * mem, size_t total_memory, size_t max_size)
{
    const char * buffer = (const char *)mem;

    for (unsigned i = 0;  i < total_memory && i < max_size;  i += 16) {
        cerr << format("%04x | ", i);
        for (unsigned j = i;  j < i + 16;  ++j) {
            if (j < total_memory)
                cerr << format("%02x ", (int)*(unsigned char *)(buffer + j));
            else cerr << "   ";
        }
        
        cerr << "| ";
        
        for (unsigned j = i;  j < i + 16;  ++j) {
            if (j < total_memory) {
                if (buffer[j] >= ' ' && buffer[j] < 127)
                    cerr << buffer[j];
                else cerr << '.';
            }
            else cerr << " ";
        }
        cerr << endl;
    }
}

} // namespace ML
