/* pair_utils.h                                                  -*- C++ -*-
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
  
   Helpful functions in dealing with pairs.
*/

#ifndef __utils__pair_utils_h__
#define __utils__pair_utils_h__

#include "jml/utils/boost_fixes.h"
#include <boost/type_traits.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include "jml/utils/sgi_functional.h"
#include <utility>

namespace ML {


/*****************************************************************************/
/* FIRST_EXTRACT_ITERATOR                                                    */
/*****************************************************************************/

template<class Iterator>
boost::transform_iterator<std::select1st<typename Iterator::value_type>,
                          Iterator>
first_extractor(const Iterator & it)
{
    return boost::transform_iterator
        <std::select1st<typename Iterator::value_type>, Iterator>(it);
}


/*****************************************************************************/
/* SECOND_EXTRACT_ITERATOR                                                   */
/*****************************************************************************/

template<class Iterator>
boost::transform_iterator<std::select2nd<typename Iterator::value_type>,
                          Iterator>
second_extractor(const Iterator & it)
{
    return boost::transform_iterator
        <std::select2nd<typename Iterator::value_type>, Iterator>(it);
}


/*****************************************************************************/
/* PAIR_MERGER                                                               */
/*****************************************************************************/

#if 0

template<class X, class Y>
struct tuple_to_pair {

    typedef std::pair<X, Y> result_type;

    result_type operator () (const boost::tuple<X, Y> & t) const
    {
        return std::make_pair(t.template get<0>(), t.template get<1>());
    }
};

template<class Iterator1, class Iterator2>
boost::transform_iterator<tuple_to_pair<typename Iterator1::value_type,
                                        typename Iterator2::value_type>,
                          boost::zip_iterator<boost::tuple<Iterator1,
                                                           Iterator2> > >
pair_merger(const Iterator1 & it1, const Iterator2 & it2)
{
    return boost::make_transform_iterator
        <tuple_to_pair<typename Iterator1::value_type,
                       typename Iterator2::value_type> >
            (boost::make_zip_iterator(boost::make_tuple(it1, it2)));
}

#else

template<typename It1, typename It2>
struct Pair_Merger
    : public boost::iterator_facade<Pair_Merger<It1, It2>,
                                    std::pair<typename std::iterator_traits<It1>::value_type,
                                              typename std::iterator_traits<It2>::value_type>,
                                    boost::forward_traversal_tag> {
    Pair_Merger()
    {
    }
    
    Pair_Merger(const It1 & it1, const It2 & it2)
        : it1(it1), it2(it2)
    {
    }

    It1 it1;
    It2 it2;

    std::pair<typename std::iterator_traits<It1>::value_type,
              typename std::iterator_traits<It2>::value_type>
    operator * () const
    {
        return std::make_pair(*it1, *it2);
    }

    bool equal(const Pair_Merger & other) const
    {
        return it1 == other.it1 && it2 == other.it2;
    }

    void increment()
    {
        ++it1;
        ++it2;
    }
};

template<typename It1, typename It2>
Pair_Merger<It1, It2>
pair_merger(const It1 & it1, const It2 & it2)
{
    return Pair_Merger<It1, It2>(it1, it2);
}

#endif

} // namespace ML

namespace std {

template<typename T1, typename T2>
std::ostream & operator << (std::ostream & stream,
                            const std::pair<T1, T2> & p)
{
    return stream << "(" << p.first << "," << p.second << ")";
}

} // namespace std

#endif /* __utils__pair_utils_h__ */
