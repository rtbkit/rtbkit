/* filter_streams.h                                                -*- C++ -*-
   Jeremy Barnes, 12 March 2005
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
   
   Streams that understand "-" syntax.
*/

#ifndef __utils__filter_streams_h__
#define __utils__filter_streams_h__

#include <iostream>
#include <fstream>
#include "boost/scoped_ptr.hpp"


namespace ML {

class filter_ostream : public std::ostream {
public:
    filter_ostream();
    filter_ostream(const std::string & file, std::ios_base::openmode mode
                   = std::ios_base::out);

    void open(const std::string & file,
              std::ios_base::openmode mode = std::ios_base::out);

private:
    boost::scoped_ptr<std::ostream> stream;
};

class filter_istream : public std::istream {
public:
    filter_istream();
    filter_istream(const std::string & file, std::ios_base::openmode mode
                   = std::ios_base::in);

    void open(const std::string & file,
              std::ios_base::openmode mode = std::ios_base::in);

private:
    boost::scoped_ptr<std::istream> stream;
};

} // namespace ML

#endif /* __utils__filter_streams_h__ */

