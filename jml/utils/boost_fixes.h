/* boost_fixes.h                                                   -*- C++ -*-
   Jeremy Barnes, 16 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Fixes for the boost library.
*/

#ifndef __utils__boost_fixes_h__
#define __utils__boost_fixes_h__

namespace boost {
namespace tuples {
struct null_type;
} // namespace tuples
namespace detail {
namespace tuple_impl_specific {
inline bool tuple_equal(tuples::null_type, tuples::null_type);
} // namespace tuple_impl_specific
} // namespace detail
} // namespace boost

#endif /* __utils__boost_fixes_h__ */
