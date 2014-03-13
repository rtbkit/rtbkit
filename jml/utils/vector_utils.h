/* vector_utils.h                                                  -*- C++ -*-
   Jeremy Barnes, 1 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
     
   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Helpful functions for dealing with vectors.
*/

#ifndef __utils__vector_utils_h__
#define __utils__vector_utils_h__

#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>


namespace ML {

struct sort_second_asc {
    template<class P>
    bool operator () (const P & p1, const P & p2) const
    {
        return p1.second < p2.second;
    }
};

template<typename T, class Alloc>
void sort_on_second_ascending(std::vector<T, Alloc> & vec)
{
    std::sort(vec.begin(), vec.end(), sort_second_asc());
}

struct stable_sort_second_asc {
    template<class P>
    bool operator () (const P & p1, const P & p2) const
    {
        return p1.second < p2.second
            || (p1.second == p2.second && p1.first < p2.first);
    }
};

template<typename T, class Alloc>
void stable_sort_on_second_ascending(std::vector<T, Alloc> & vec)
{
    std::sort(vec.begin(), vec.end(), stable_sort_second_asc());
}

struct sort_second_desc {
    template<class P>
    bool operator () (const P & p1, const P & p2) const
    {
        return p1.second > p2.second;
    }
};

template<typename T, class Alloc>
void sort_on_second_descending(std::vector<T, Alloc> & vec)
{
    std::sort(vec.begin(), vec.end(), sort_second_desc());
}

struct stable_sort_second_desc {
    template<class P>
    bool operator () (const P & p1, const P & p2) const
    {
        return p1.second > p2.second
            || (p1.second == p2.second && p1.first < p2.first);
    }
};

template<typename T, class Alloc>
void stable_sort_on_second_descending(std::vector<T, Alloc> & vec)
{
    std::sort(vec.begin(), vec.end(), stable_sort_second_desc());
}

struct sort_first_asc {
    template<class P>
    bool operator () (const P & p1, const P & p2) const
    {
        return p1.first < p2.first;
    }
};

template<typename T, class Alloc>
void sort_on_first_ascending(std::vector<T, Alloc> & vec)
{
    std::sort(vec.begin(), vec.end(), sort_first_asc());
}

struct sort_first_desc {
    template<class P>
    bool operator () (const P & p1, const P & p2) const
    {
        return p2.first < p1.first;
    }
};

template<typename T, class Alloc>
void sort_on_first_descending(std::vector<T, Alloc> & vec)
{
    std::sort(vec.begin(), vec.end(), sort_first_desc());
}

template<typename T>
void make_vector_set(std::vector<T> & vec)
{
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
}

} // namespace ML

namespace std {

template<class T, class A>
std::ostream &
operator << (std::ostream & stream, const vector<T, A> & vec)
{
    stream << "[";
    for (unsigned i = 0;  i < vec.size();  ++i)
        stream << " " << vec[i];
    return stream << " ]";
}

template<class T, class A>
std::string
to_string(const vector<T, A> & vec)
{
    ostringstream stream;

    stream << vec;

    return stream.str();
}


} // namespace std

#endif /* __utils__vector_utils_h__ */
