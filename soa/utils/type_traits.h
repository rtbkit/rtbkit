/** type_traits.h                                 -*- C++ -*-
    Rémi Attab, 13 Mar 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Missing type traits from gcc 4.6 and boost 1.52.

*/

#pragma once

#include <type_traits>
#include <utility>

namespace Datacratic {


/******************************************************************************/
/* IS COPY CONSTRUCTIBLE                                                      */
/******************************************************************************/

template<typename T>
struct is_copy_constructible :
        public std::is_constructible<
            T, const typename std::add_lvalue_reference<T>::type>
{};


/******************************************************************************/
/* IS DEFAULT CONSTRUCTIBLE                                                   */
/******************************************************************************/

template<typename T>
struct is_default_constructible:
        public std::is_constructible<T>
{};


/******************************************************************************/
/* IS COPY ASSIGNABLE                                                         */
/******************************************************************************/

template<typename T>
struct is_copy_assignable
{
    template<typename U>
    static std::true_type test(
            typename std::remove_reference<
                decltype(*((U*)0) = *((const U*)0))>::type* = 0);

    template<typename>
    static std::false_type test(...);

    typedef decltype(test<T>(0)) type;
    static constexpr bool value = type::value;
};


/******************************************************************************/
/* IS MOVE ASSIGNABLE                                                         */
/******************************************************************************/

template<typename T>
struct is_move_assignable
{
    template<typename U>
    static std::true_type test(
            typename std::remove_reference<
                decltype(*((U*)0) = std::move(*((U*)0)))>::type* = 0);

    template<typename>
    static std::false_type test(...);

    typedef decltype(test<T>(0)) type;
    static constexpr bool value = type::value;
};

} // namespace Datacratic
