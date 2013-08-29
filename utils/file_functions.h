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

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <string>
#include <stdint.h>
#include <ftw.h>

namespace ML {


size_t get_file_size(int fd);

size_t get_file_size(const std::string & filename);

std::string get_link_target(const std::string & link);

std::string get_name_from_fd(int fd);

typedef std::pair<uint32_t, uint32_t> inode_type;

inode_type get_inode(const std::string & filename);

inode_type get_inode(int fd);

void delete_file(const std::string & filename);

/* wrappers around fcntl with F_GETFL/F_SETFL */
void set_file_flag(int fd, int newFlag);
void unset_file_flag(int fd, int oldFlag);
bool is_file_flag_set(int fd, int flag);

void set_permissions(std::string filename,
                     const std::string & perms,
                     const std::string & group);

/** Call fdatasync on the file. */
void syncFile(const std::string & filename);

/** Does the file exist? */
bool fileExists(const std::string & filename);

enum FileAction {
    FA_CONTINUE = FTW_CONTINUE,
    FA_SKIP_SIBLINGS = FTW_SKIP_SIBLINGS,
    FA_SKIP_SUBTREE = FTW_SKIP_SUBTREE,
    FA_STOP = FTW_STOP
};

enum FileType {
    FT_FILE = FTW_F,
    FT_DIR  = FTW_D,
    FT_DIR_INACCESSIBLE = FTW_DNR,
    FT_FILE_INACCESSIBLE = FTW_NS
};

std::string print(FileType type);

std::ostream & operator << (std::ostream & stream, const FileType & type);

typedef boost::function<FileAction (std::string dir,
                                    std::string basename,
                                    const struct stat & stats,
                                    FileType type,
                                    int depth)>
    OnFileFound;

void scanFiles(const std::string & path,
               OnFileFound onFileFound,
               int maxDepth = -1);


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
    File_Read_Buffer(const char * start, size_t length,
                     const std::string & fileName = "anonymous memory",
                     boost::function<void ()> onDone = boost::function<void ()>());

    void open(const std::string & filename);
    void open(int fd);

    void open(const char * start, size_t length,
              const std::string & filename = "anonymous memory",
              boost::function<void ()> onDone = boost::function<void ()>());
    
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
    std::shared_ptr<Region> region;

    class MMap_Region;
    class Mem_Region;
};

} // namespace ML


#endif /* __utils__file_functions_h__ */
