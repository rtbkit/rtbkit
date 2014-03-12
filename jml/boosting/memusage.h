/* memusage.h                                                      -*- C++ -*-
   Jeremy Barnes, 20 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Functions to calculate the memory usage of various data structures.
*/

#ifndef __boosting__memusage_h__
#define __boosting__memusage_h__


namespace std {

template<class Key, class Data, class Compare, class Alloc> class map;
template<class First, class Second> class pair;
template<class T, class Alloc> class vector;
template<class Char, class Traits, class Alloc> class basic_string;

};

namespace __gnu_cxx {

template<class Key, class Data, class Hash, class Equal, class Alloc>
class hash_map;

}

namespace ML {

template<class X>
size_t memusage(const X &)
{
    return sizeof(X);
}

template<class Type>
struct mem_traits {
    enum {
        is_fixed = 0
    };
};

#define MU_FUNDAMENTAL(type) \
inline size_t memusage(type) \
{ \
    return sizeof(type); \
} \
template<> \
struct mem_traits<type> { \
    enum { is_fixed = 1 }; \
}; \

MU_FUNDAMENTAL(unsigned char);
MU_FUNDAMENTAL(signed char);
MU_FUNDAMENTAL(unsigned short);
MU_FUNDAMENTAL(signed short);
MU_FUNDAMENTAL(unsigned int);
MU_FUNDAMENTAL(signed int);
MU_FUNDAMENTAL(unsigned long);
MU_FUNDAMENTAL(signed long);
MU_FUNDAMENTAL(unsigned long long);
MU_FUNDAMENTAL(signed long long);
MU_FUNDAMENTAL(bool);

template<class First, class Second>
struct mem_traits<std::pair<First, Second> > {
    enum {
        is_fixed = mem_traits<First>::is_fixed && mem_traits<Second>::is_fixed
    };
};

template<class First, class Second>
size_t memusage(const std::pair<First, Second> & p)
{
    return memusage(p.first) + memusage(p.second);
}

template<class Iterator>
size_t memusage_range(Iterator first, const Iterator & last)
{
    typedef typename std::iterator_traits<Iterator>::value_type value_type;
    size_t result = 0;
    if (mem_traits<value_type>::is_fixed)
        return std::distance(first, last) * sizeof(value_type);
    else while (first != last) result += memusage(*first++);
    return result;
}

template<class Key, class Data, class Compare, class Alloc>
size_t memusage(const std::map<Key, Data, Compare, Alloc> & m)
{
    typedef std::map<Key, Data, Compare, Alloc> array_type;
    
    size_t nodes = m.size();
    size_t node_size = 3 * sizeof(void *) + sizeof(int);
    
    size_t result = sizeof(array_type) + nodes * node_size;
    result += memusage_range(m.begin(), m.end());
    return result;
}

template<class Key, class Data, class Hash, class Equal, class Alloc>
size_t memusage(const __gnu_cxx::hash_map<Key, Data, Hash, Equal, Alloc> & m)
{
    typedef __gnu_cxx::hash_map<Key, Data, Hash, Equal, Alloc> array_type;
    
    size_t num_nodes = m.size();
    size_t num_buckets = m.bucket_count();
    
    size_t result = sizeof(array_type)
        + num_nodes * (sizeof(void *))
        + num_buckets * (sizeof(void *));
    result += memusage_range(m.begin(), m.end());
    return result;
}

template<class T, class Alloc>
size_t memusage(const std::vector<T, Alloc> & vec)
{
    return sizeof(vec) + memusage_range(vec.begin(), vec.end())
        + (vec.capacity() - vec.size()) * sizeof(T);
}

template<class Char, class Traits, class Alloc>
size_t memusage(const std::basic_string<Char, Traits, Alloc> & str)
{
    return str.capacity() * sizeof(Char) + sizeof(str);
}

} // namespace ML

#endif /* __boosting__memusage_h__ */
