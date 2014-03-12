// Copied from boost; under the Boost license

#ifndef __jml__utils__move_h__
#define __jml__utils__move_h__

#include <iterator>
#include <algorithm>

namespace ML {

template<typename I, typename F>
F uninitialized_move(I first, I last, F result)
{
   for (; first != last; ++result, ++first)
       new (static_cast<void*>(&*result))
           typename std::iterator_traits<F>::value_type
               (std::move(*first));

   return first;
}

template<typename T>
void destroy(T & t)
{
    t.~T();
}

template<typename I, typename F>
F uninitialized_move_and_destroy(I first, I last, F result)
{
    for (; first != last; ++result, ++first) {
        auto && v = std::move(*first);
        new (static_cast<void*>(&*result))
            typename std::iterator_traits<F>::value_type(v);
        destroy(v);
    }

   return first;
}

} // namespace ML


#endif
