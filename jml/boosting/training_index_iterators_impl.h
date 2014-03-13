/* training_index_impl.h                                           -*- C++ -*-
   Jeremy Barnes, 23 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of inline methods of the training_index.
*/

#ifndef __boosting__training_index_iterators_impl_h__
#define __boosting__training_index_iterators_impl_h__

#include "training_index_iterators.h"

namespace ML {


inline Index_Iterator Joint_Index::begin() const
{
    return Index_Iterator(this, 0);
}

inline Index_Iterator Joint_Index::end() const
{
    return Index_Iterator(this, size_);
}

inline Index_Iterator Joint_Index::front() const
{
    return Index_Iterator(this, 0);
}

inline Index_Iterator Joint_Index::back() const
{
    return Index_Iterator(this, size_ - 1);
}

inline Index_Iterator Joint_Index::operator [] (int idx) const
{
    return Index_Iterator(this, idx);
}


} // namespace ML

#endif /* __boosting__training_index_iterators_impl_h__ */
