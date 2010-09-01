/* multi_array_utils.h                                             -*- C++ -*-
   Jeremy Barnes, 24 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Utilities to deal with multi arrays.
*/

#ifndef __boosting__multi_array_utils_h__
#define __boosting__multi_array_utils_h__

#include <boost/multi_array.hpp>

namespace ML {

template<typename Val, std::size_t Dims, class Allocator>
void swap_multi_arrays(boost::multi_array<Val, Dims, Allocator> & a1,
                       boost::multi_array<Val, Dims, Allocator> & a2)
{
    /* Since we know there is nothing self-referential, we do a bit by
       bit copy.  Note that this might not work for some allocator
       types.
 
       This is a hack needed to get around the lack of a swap function
       in the boost multi array types.
   */

    volatile char * p1 = (volatile char *)(&a1);
    volatile char * p2 = (volatile char *)(&a2);

    for (unsigned i = 0;  i < sizeof(a1);  ++i)
        std::swap(p1[i], p2[i]);
}

} // namespace ML

#endif /* __boosting__multi_array_utils_h__ */
