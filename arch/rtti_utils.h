/* rtti_utils.h                                                    -*- C++ -*-
   Jeremy Barnes, 18 October 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.
   Copyright (c) 2010 Recoset.  All rights reserved.

   Utilities for runtime type info.
*/

#ifndef __jml__arch__rtti_utils_h__
#define __jml__arch__rtti_utils_h__

#include <typeinfo>

namespace ML {

bool is_convertible(const std::type_info & from_type,
                    const std::type_info & to_type,
                    const void * obj);

template<typename FromT>
bool is_convertible(const FromT & from_obj,
                    const std::type_info & to_type)
{
    return is_convertible(typeid(from_obj), to_type, &from_obj);
}

template<typename ToT, typename FromT>
bool is_convertible(const FromT & from_obj)
{
    return is_convertible(typeid(from_obj), typeid(ToT), &from_obj);
}

} // namespace ML

#endif /* __jml__arch__rtti_utils_h__ */
