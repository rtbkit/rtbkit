/* portable_oarchive.cc
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

   Implementation of the portable oarchive.
*/


#include "portable_oarchive.h"
#include "nested_archive.h"
#include "jml/utils/filter_streams.h"

using namespace std;


namespace ML {
namespace DB {


/*****************************************************************************/
/* PORTABLE_BIN_OARCHIVE                                                     */
/*****************************************************************************/

portable_bin_oarchive::portable_bin_oarchive()
    : stream(0), offset_(0)
{
}

portable_bin_oarchive::portable_bin_oarchive(const std::string & filename)
    : stream(new filter_ostream(filename)), owned_stream(stream),
      offset_(0)
{
}

portable_bin_oarchive::portable_bin_oarchive(std::ostream & stream)
    : stream(&stream), offset_(0)
{
}

void portable_bin_oarchive::open(const std::string & filename)
{
    stream = new filter_ostream(filename.c_str());
    owned_stream.reset(stream);
    offset_ = 0;
}

void portable_bin_oarchive::open(std::ostream & stream)
{
    this->stream = &stream;
    owned_stream.reset();
    offset_ = 0;
}

void
portable_bin_oarchive::
save(const Nested_Writer & writer)
{
    writer.serialize(*this);
}

} // namespace DB
} // namespace ML
