/* rtti_utils.cc
   Jeremy Barnes, 18 October 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.
   Copyright (c) 2010 Recoset.  All rights reserved.

*/

#include "rtti_utils.h"
#include <cxxabi.h>
#include <iostream>

using namespace abi;
using namespace std;

namespace ML {

bool is_convertible(const std::type_info & from_type,
                    const std::type_info & to_type,
                    const void * obj)
{
    const char * adj_ptr = (const char *)obj;
    bool result = to_type.__do_catch(&from_type, (void **)&adj_ptr, 0);
    //if (result)
    //    cerr << "offset = " << (adj_ptr - (const char *)obj) << endl;
    return result;
}

} // namespace ML
   
