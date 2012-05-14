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

} // namespace ML

#endif /* __ml__positioned_types_h__ */
