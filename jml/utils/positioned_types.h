/* positioned_types.h                                              -*- C++ -*-
   Jeremy Barnes, 13 November 2012
   Copyright (c) 2012 Datacratic Inc.

   Public domain.
*/

#ifndef __ml__positioned_types_h__
#define __ml__positioned_types_h__

namespace ML {

// Container metaclass that contains simply a list of types that can be passed
// around as a single type
template<typename... Args> struct TypeList;

// Factory metaclass that creates a TypeList by adding a new type onto the front
// of an existing TypeList
template<typename Head, typename Rest>
struct PushFront {
};

// Implementation of the factory class
template<typename Head, typename... Rest>
struct PushFront<Head, TypeList<Rest...> > {
    typedef TypeList<Head, Rest...> List;
};

// Simple type that records that an argument list has argument Arg at position
// Index
template<typename Arg, int Index>
struct InPosition {
};

// Factory metaclass that takes a variable length list of arguments and turns
// it into a TypeList containing InPosition arguments
template<int Index, typename... Args>
struct MakeInPositionList {
};

// Implementation for non-recursive case
template<int Index, typename Head, typename... Rest>
struct MakeInPositionList<Index, Head, Rest...> {
    typedef MakeInPositionList<Index + 1, Rest...> Base;
    typedef typename PushFront<InPosition<Head, Index>,
                               typename Base::List>::List List;
};

// End of recursion
template<int Index>
struct MakeInPositionList<Index> {
    typedef TypeList<> List;
};


/*****************************************************************************/
/* EXTRACT ARG AT POSITION                                                   */
/*****************************************************************************/

/** Given an integer index and a list of types, this will extract both the
    type and the value of the argument at the given index.
*/
template<int Current, int Index,
         typename... Params>
struct ExtractArgAtPosition {
};

/** End of recursion... we have made it to where we need to be */
template<int Current, typename Head, typename... Rest>
struct ExtractArgAtPosition<Current, Current, Head, Rest...> {
    // This is the one we need
    typedef Head type;
    static Head extract(Head arg, Rest... rest)
    {
        return arg;
    }
};

/** Recursive version */
template<int Current, int Index,
         typename Head, typename... Rest>
struct ExtractArgAtPosition<Current, Index, Head, Rest...> {
    typedef typename ExtractArgAtPosition<Current + 1, Index, Rest...>::type type;

    // Recurse to get a later argument
    template<typename... TRest>
    static type
    extract(Head arg, TRest&&... rest)
    {
        return ExtractArgAtPosition<Current + 1, Index, Rest...>
            ::extract(std::forward<Rest>(rest)...);
    }
};


/*****************************************************************************/
/* POSITIONED DUAL TYPE                                                      */
/*****************************************************************************/

/** A container metaclass that holds a position (index) and two types. */

template<int Index, typename T1, typename T2>
struct PositionedDualType {
};

/** A template that constructs a TypeList of PositionedDualType values
    from two TypeLists.  It's kind of like a zip function for two lists
    of types:

    <t0, t1, t2, t3>, <u0, u1, u2, u3>, <0, 1, 2, 3> -->
        < <t0, u0, 0>, <t1, u1, 1>, <t2, u2, 2>, <t3, u3, 3> >
*/

template<int Index, typename TypeList1, typename TypeList2>
struct PositionedDualTypeList {
};

template<int Index,
         typename Head1, typename... Tail1,
         typename Head2, typename... Tail2>
struct PositionedDualTypeList<Index,
                              ML::TypeList<Head1, Tail1...>,
                              ML::TypeList<Head2, Tail2...> > {
    typedef PositionedDualTypeList<Index + 1,
                                   ML::TypeList<Tail1...>,
                                   ML::TypeList<Tail2...> > Base;
typedef typename ML::PushFront<PositionedDualType<Index, Head1, Head2>,
                               typename Base::List>::List List;
};

template<int Index>
struct PositionedDualTypeList<Index,
                              ML::TypeList<>,
                              ML::TypeList<> > {
    typedef ML::TypeList<> List;
};



} // namespace ML

#endif /* __ml__positioned_types_h__ */
