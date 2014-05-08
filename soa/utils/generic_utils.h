/** generic_utils.h                                 -*- C++ -*-
    Mathieu Stefani, 30 April 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Various generic utilities

*/

#pragma once

#include <algorithm>
#include <type_traits>

namespace Datacratic {

/** Given a particular container, returns the index of a partircular element
    based on member variable.

    Example:

    struct Foo {
        int id;
    };
    std::vector<Foo> foos { Foo(0), Foo(1), Foo(3), Foo(6) };

    int i1 = indexOf(foos, &Foo::id, 3);
    assert(i1 == 2);

    int i2 = indexOf(foos, &Foo::id, 9);
    assert(i2 == -1);
*/
template<typename Container, typename Value, typename Class, typename Member>
int indexOf(const Container &container, Class Member::*ptr, const Value &value)
{
    static_assert(std::is_same<typename Container::value_type, Member>::value,
                  "Member does not match container value type");
    auto it = 
    std::find_if(std::begin(container), std::end(container),
        [&](const Member &member) { return member.*ptr == value; });

    if (it == std::end(container)) {
        return -1;
    }

    return std::distance(std::begin(container), it);
}

} // namespace Datacratic
