/* file_functions.h                                                -*- C++ -*-
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

   Functions to deal with files.
*/

#ifndef __utils__file_functions_h__
#define __utils__file_functions_h__

#include "boost/shared_ptr.hpp"
#include <string>
#include <stdint.h>

namespace ML {


size_t get_file_size(int fd);

size_t get_file_size(const std::string & filename);

std::string get_link_target(const std::string & link);

std::string get_name_from_fd(int fd);

typedef std::pair<uint32_t, uint32_t> inode_type;

inode_type get_inode(const std::string & filename);

inode_type get_inode(int fd);

void delete_file(const std::string & filename);


/*****************************************************************************/
/* FILE_READ_BUFFER                                                          */
/*****************************************************************************/

/** A class that memory maps a file in order to allow it to be read as a
    buffer.  Usually much faster than streams, as the virtual memory hardware
    can do all the paging for us.
*/

class File_Read_Buffer {
public:
    File_Read_Buffer();
    File_Read_Buffer(const std::string & filename);
    File_Read_Buffer(int fd);
    File_Read_Buffer(const File_Read_Buffer & other);

    void open(const std::string & filename);
    void open(int fd);
    
    void close();

    const char * start() const { return region->start; }
    const char * end() const { return region->start + region->size; }
    size_t size() const { return region->size; }

    std::string filename() const { return filename_; }

    /* Only access this if you know what you are doing... */
    class Region {
    public:
        virtual ~Region();
        const char * start;
        size_t size;
    };

    std::string filename_;
    boost::shared_ptr<Region> region;

    class MMap_Region;
    class Mem_Region;
};

} // namespace ML


#endif /* __utils__file_functions_h__ */
