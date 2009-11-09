/* check_not_nan.h                                                 -*- C++ -*-
   Jeremy Barnes, 8 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Check that a range of values doesn't contain NaN values.
*/

#ifndef __jml__utils__check_not_nan_h__
#define __jml__utils__check_not_nan_h__

#ifndef NDEBUG
#  define CHECK_NOT_NAN(x) \
    { for (unsigned __i = 0;  __i < x.size();  ++__i) { if (isnan(x[__i])) throw Exception(format("element %d of %s is Nan in %s %s:%d", __i, #x, __PRETTY_FUNCTION__, __FILE__, __LINE__)); } }

#  define CHECK_NOT_NAN_RANGE(begin, end)           \
    { for (typeof(begin) it = begin;  it != end;  ++it) { if (isnan(*it)) throw Exception(format("element %d of range %s-%s is Nan in %s %s:%d", std::distance(begin, it), #begin, #end, __PRETTY_FUNCTION__, __FILE__, __LINE__)); } }
#else
#  define CHECK_NOT_NAN(x)
#  define CHECK_NOT_NAN_RANGE(begin, end)
#endif




#endif /* __jml__utils__check_not_nan_h__ */
