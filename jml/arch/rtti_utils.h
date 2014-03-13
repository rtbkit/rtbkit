/* rtti_utils.h                                                    -*- C++ -*-
   Jeremy Barnes, 18 October 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Utilities for runtime type info.
*/

#ifndef __jml__arch__rtti_utils_h__
#define __jml__arch__rtti_utils_h__

#include <typeinfo>

namespace ML {

/** Convert the given object pointed to by obj (with a type info node of
    from_type) to the given type (to_type).  If it's convertible, then the
    adjusted pointer is returned, otherwise a null pointer.
*/
const void * is_convertible(const std::type_info & from_type,
                            const std::type_info & to_type,
                            const void * obj);

inline void * is_convertible(const std::type_info & from_type,
                             const std::type_info & to_type,
                             void * obj)
{
    return const_cast<void *>(is_convertible(from_type, to_type,
                                             (const void *)obj));
}

template<typename FromT>
const void * is_convertible(const FromT & from_obj,
                            const std::type_info & to_type)
{
    return is_convertible(typeid(FromT), to_type, &from_obj);
    //return is_convertible(typeid(from_obj), to_type, &from_obj);
}

template<typename ToT, typename FromT>
const void * is_convertible(const FromT & from_obj)
{
    return is_convertible(typeid(FromT), typeid(ToT), &from_obj);
    //return is_convertible(typeid(from_obj), typeid(ToT), &from_obj);
}

} // namespace ML

#endif /* __jml__arch__rtti_utils_h__ */
