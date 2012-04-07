/* check_not_nan.h                                                 -*- C++ -*-
   Jeremy Barnes, 8 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Check that a range of values doesn't contain NaN values.
*/

#ifndef __jml__utils__check_not_nan_h__
#define __jml__utils__check_not_nan_h__

#ifndef NDEBUG
#  define CHECK_NOT_NAN(__x) \
    { for (unsigned __i = 0;  __i < __x.size();  ++__i) { if (std::isnan(__x[__i])) throw Exception(format("element %d of %s is Nan in %s %s:%d", __i, #__x, __PRETTY_FUNCTION__, __FILE__, __LINE__)); } }

#  define CHECK_NOT_NAN_RANGE(__begin, __end)           \
    { for (auto __it = __begin;  __it != __end;  ++__it) { if (std::isnan(*__it)) throw Exception(format("element %zd of range %s-%s is Nan in %s %s:%d", (size_t)std::distance(__begin, __it), #__begin, #__end, __PRETTY_FUNCTION__, __FILE__, __LINE__)); } }
#  define CHECK_NOT_NAN_N(__begin, __n) \
    { for (auto __it = __begin;  __it != __begin + __n;  ++__it) { if (std::isnan(*__it)) throw Exception(format("element %zd of range %s-%s is Nan in %s %s:%d", (size_t)std::distance(__begin, __it), #__begin, #__n, __PRETTY_FUNCTION__, __FILE__, __LINE__)); } }
#else
#  define CHECK_NOT_NAN(x)
#  define CHECK_NOT_NAN_RANGE(begin, end)
#  define CHECK_NOT_NAN_N(begin, n)
#endif




#endif /* __jml__utils__check_not_nan_h__ */
