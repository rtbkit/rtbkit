/** type_traits.cc                                 -*- C++ -*-
    Rémi Attab, 13 Mar 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Type traits compile tests

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "soa/utils/type_traits.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace std;
using namespace Datacratic;

struct Movable
{
    const Movable& operator=(const Movable&) = delete;
    Movable& operator=(Movable&&) { return *this; }
};

struct Copiable
{
    Copiable& operator=(Copiable&&) = delete;
    const Copiable& operator=(const Copiable&) { return *this; }
};

struct Neither
{
    const Neither& operator=(const Neither&) = delete;
    Neither& operator=(Neither&&) = delete;
};

static_assert(Datacratic::is_copy_assignable<size_t>::value, "size_t copy");
static_assert(Datacratic::is_move_assignable<size_t>::value, "size_t move");

static_assert(!Datacratic::is_copy_assignable<Movable>::value, "Movable copy");
static_assert( Datacratic::is_move_assignable<Movable>::value, "Movable move");

static_assert( Datacratic::is_copy_assignable<Copiable>::value, "Copiable copy");
static_assert(!Datacratic::is_move_assignable<Copiable>::value, "Copiable move");

static_assert(!Datacratic::is_copy_assignable<Neither>::value, "Neither copy");
static_assert(!Datacratic::is_move_assignable<Neither>::value, "Neither move");


BOOST_AUTO_TEST_CASE( test_something )
{
    cerr << "Nothing to see here, move along..." << endl;
}
