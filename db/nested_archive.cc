/* nested_archive.cc
   Jeremy Barnes, 20 March 2005
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

   Implementation of the nested archive classes.
*/

#include "nested_archive.h"


namespace ML {
namespace DB {


/*****************************************************************************/
/* NESTED_READER                                                             */
/*****************************************************************************/

Nested_Reader::Nested_Reader()
{
    open(stream);
}


/*****************************************************************************/
/* NESTED_WRITER                                                             */
/*****************************************************************************/

Nested_Writer::Nested_Writer()
{
    open(stream);
}

} // namespace DB
} // namespace ML
