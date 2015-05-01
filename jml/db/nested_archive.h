/* nested_archive.h                                                -*- C++ -*-
   Jeremy Barnes, 30 January 2005
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

   A nested archive, that allows one to be embedded within another.
*/

#ifndef __db__nested_archive_h__
#define __db__nested_archive_h__

#include "persistent.h"
#include <sstream>

namespace ML {
namespace DB {


/*****************************************************************************/
/* NESTED_READER                                                             */
/*****************************************************************************/

/** A store reader that can be used to read an archive nested within another
    archive.
*/

class Nested_Reader : public Store_Reader {
public:
    Nested_Reader();

    template<class Archive>
    void serialize(Archive & archive)
    {
        // reconstitute
        std::string data;
        archive >> data;
        stream.str(data);
    }

private:
    std::istringstream stream;
};


/*****************************************************************************/
/* NESTED_WRITER                                                             */
/*****************************************************************************/

/** A store writer that can be used to write an archive and nest it within
    another archive.
*/

class Nested_Writer : public Store_Writer {
public:
    Nested_Writer();

    template<class Archive>
    void serialize(Archive & archive) const
    {
        archive << stream.str();
    }
    
private:
    std::ostringstream stream;
};

} // namespace DB
} // namespace ML


#endif /* __db__nested_archive_h__ */

