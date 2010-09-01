/* fixed_array_test.cc
   Jeremy Barnes, 8 February 2005
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

   Test of the fixed array class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#define BOOST_AUTO_TEST_MAIN
#include <boost/test/auto_unit_test.hpp>

#include "jml/utils/fixed_array.h"

using namespace ML;

BOOST_AUTO_UNIT_TEST(default_construct_1)
{
    fixed_array<float, 1> arr;
}

BOOST_AUTO_UNIT_TEST(default_construct_2)
{
    fixed_array<float, 2> arr;
}

BOOST_AUTO_UNIT_TEST(default_construct_3)
{
    fixed_array<float, 3> arr;
}

BOOST_AUTO_UNIT_TEST(default_const_construct_1)
{
    fixed_array<const float, 1> arr;
}

BOOST_AUTO_UNIT_TEST(default_const_construct_2)
{
    fixed_array<const float, 2> arr;
}

BOOST_AUTO_UNIT_TEST(default_const_construct_3)
{
    fixed_array<const float, 3> arr;
}

BOOST_AUTO_UNIT_TEST(sized_construct_1)
{
    fixed_array<float, 1> arr(10);
    //arr[1] = 10;
    //BOOST_CHECK_EQUAL(arr[1], 10);
}

BOOST_AUTO_UNIT_TEST(sized_construct_2)
{
    fixed_array<float, 2> arr(10, 20);
}

BOOST_AUTO_UNIT_TEST(sized_construct_3)
{
    fixed_array<float, 3> arr(10, 20, 30);
}

