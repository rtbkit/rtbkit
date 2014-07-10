/* file_functions.cc
   Jeremy Barnes, 14 March 2005
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

#include "file_functions.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include "string_functions.h"
#include "jml/arch/exception.h"
#include "jml/arch/spinlock.h"
#include <iostream>
#include <boost/weak_ptr.hpp>
#include <map>
#include <grp.h>
#include <exception>
#include "guard.h"
#include <mutex>


using namespace std;


namespace ML {

size_t get_file_size(int fd)
{
    struct stat stats;
    int result = fstat(fd, &stats);
    if (result != 0)
        throw Exception("fstat: " + string(strerror(errno)));
    return stats.st_size;
}

size_t get_file_size(const std::string & filename)
{
    struct stat stats;
    int result = stat(filename.c_str(), &stats);
    if (result != 0)
        throw Exception("stat: " + string(strerror(errno)));
    return stats.st_size;
}

std::string get_link_target(const std::string & link)
{
    /* Interface to the readlink call */
    size_t bufsize = 1024;
    
    /* Loop over, making the buffer successively larger if it is too small. */
    while (true) {  // break in loop
        char buf[bufsize];
        int res = readlink(link.c_str(), buf, bufsize);
        if (res == -1)
            throw Exception(errno, "readlink", "get_link_name()");
        if (res == bufsize) {
            bufsize *= 2;
            continue;
        }
        buf[res] = 0;
        return buf;
    }
}

std::string get_name_from_fd(int fd)
{
    string fname = format("/proc/self/fd/%d", fd);
    return get_link_target(fname);
}

inode_type get_inode(const std::string & filename)
{
    struct stat st;

    int res = stat(filename.c_str(), &st);

    if (res != 0)
        throw Exception(errno, "fstat", "get_inode()");

    return make_pair(st.st_dev, st.st_ino);
}

inode_type get_inode(int fd)
{
    struct stat st;

    int res = fstat(fd, &st);

    if (res != 0)
        throw Exception(errno, "fstat", "get_inode()");

    return make_pair(st.st_dev, st.st_ino);
}

void delete_file(const std::string & filename)
{
    int res = unlink(filename.c_str());
    if (res != 0)
        throw Exception(errno, "couldn't delete file " + filename, "unlink");
}

void set_file_flag(int fd, int newFlag)
{
    int oldFlags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, oldFlags | newFlag);
}

void unset_file_flag(int fd, int oldFlag)
{
    int oldFlags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, oldFlags & ~oldFlag);
}

bool is_file_flag_set(int fd, int flag)
{
    int oldFlags = fcntl(fd, F_GETFL, 0);
    return ((oldFlags & flag) == flag);
}

/** Does the file exist? */
bool fileExists(const std::string & filename)
{
    struct stat stats;
    int res = stat(filename.c_str(), &stats);
    if (res == -1)
        return false;  // no chunks
    return true;
}

void set_permissions(std::string filename,
                     const std::string & perms,
                     const std::string & group)
{
    if (filename.find("ipc://") == 0)
        filename = string(filename, 6);

    char * endptr;
    if (perms != "") {
        int mode = strtol(perms.c_str(), &endptr, 8);
        if (*endptr != '\0')
            throw Exception(errno, "parsing permission modes %s",
                            perms.c_str());
        
        cerr << "setting permissions of " << filename << " to " << perms
             << "(" << mode << ")" << endl;

        int res = chmod(filename.c_str(), mode);
        if (res == -1)
            throw Exception(errno, "chmod on %s in setPermissions");
    }

    if (group != "") {
        int groupNum = strtol(group.c_str(), &endptr, 10);
        if (*endptr != '\0') {
            struct group * gr = getgrnam(group.c_str());
            if (gr == 0)
                throw Exception(errno, "finding group \"%s\"", group.c_str());
            groupNum = gr->gr_gid;
        }
        
        cerr << "setting group of " << filename << " to " << groupNum
             << " for group " << group << endl;
        
        int res = chown(filename.c_str(), -1, groupNum);

        if (res == -1)
            throw Exception(errno, "chown in setPermissions");
    }
}

std::string print(FileType type)
{
    switch (type) {
    case FT_FILE: return "FILE";
    case FT_DIR:  return "DIR";
    case FT_DIR_INACCESSIBLE: return "DIR_INACCESSIBLE";
    case FT_FILE_INACCESSIBLE: return "FILE_INACCESSIBLE";
    default:
        return ML::format("FileType(%d)", type);
    }
}

std::ostream &
operator << (std::ostream & stream, const FileType & type)
{
    return stream << print(type);
}

struct ScanFilesData;

static __thread ScanFilesData * scanFilesThreadData = 0;

struct ScanFilesData {
    ScanFilesData(const OnFileFound & onFileFound,
                  int maxDepth)
        : onFileFound(onFileFound),
          maxDepth(maxDepth),
          isThrown(false)
    {
    }
    
    OnFileFound onFileFound;
    int maxDepth;
    std::exception_ptr thrown;
    bool isThrown;

    static int onFile (const char *fpath, const struct stat *sb,
                       int typeflag, struct FTW *ftwbuf)
    {
        ScanFilesData * d = scanFilesThreadData;
        try {
            if (d->maxDepth != -1 && ftwbuf->level > d->maxDepth)
                return FTW_SKIP_SIBLINGS;
            string dir(fpath, fpath + ftwbuf->base);
            string basename(fpath + ftwbuf->base);

            FileAction action = d->onFileFound(dir, basename, *sb,
                                               (FileType)typeflag,
                                               ftwbuf->level);
            return action;
        } catch (...) {
            d->thrown = std::current_exception();
            d->isThrown = true;
            return FTW_STOP;
        }
    }
}; 

void scanFiles(const std::string & path,
               OnFileFound onFileFound,
               int maxDepth)
{
    ScanFilesData data(onFileFound, maxDepth);

    scanFilesThreadData = &data;

    int res = nftw(path.c_str(),
                   &ScanFilesData::onFile, 
                   maxDepth == -1 ? 100 : maxDepth + 1,
                   FTW_ACTIONRETVAL);

    if (scanFilesThreadData->isThrown) {
        auto exc = scanFilesThreadData->thrown;
        scanFilesThreadData = 0;
        rethrow_exception(exc);
    }

    scanFilesThreadData = 0;

    if (res == -1) {
        if (errno == ENOENT)
            return;
        throw ML::Exception(errno, "ftw");
    }
}

/** Call fdatasync on the file. */
void syncFile(const std::string & filename)
{
    int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd == -1)
        throw ML::Exception(errno, "syncFile for " + filename);
    ML::Call_Guard guard([=] () { close(fd); });
    int res = fdatasync(fd);
    if (res == -1)
        throw ML::Exception(errno, "fdatasync for " + filename);
}


/*****************************************************************************/
/* FILE_READ_BUFFER                                                          */
/*****************************************************************************/

/** Helper class to specify and clean up a memory mapped region. */
class File_Read_Buffer::MMap_Region
    : public File_Read_Buffer::Region {

    inode_type inode;

    MMap_Region(int fd, inode_type inode)
        : inode(inode)
    {
        size = get_file_size(fd);

        if (size == 0) {
            start = 0;
        }
        else {
            start = (const char *)mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
            if (start == MAP_FAILED)
                throw Exception(errno, "mmap", "MMap_Region()");
        }

    }

    MMap_Region(const MMap_Region & other);

    MMap_Region & operator = (const MMap_Region & other);

    typedef std::map<inode_type, std::weak_ptr<MMap_Region> > cache_type;

    static cache_type & cache()
    {
        static cache_type result;
        return result;
    }

    static ML::Spinlock & lock()
    {
        static ML::Spinlock result;
        return result;
    }

public:
    static std::shared_ptr<MMap_Region> get(int fd)
    {
        std::lock_guard<ML::Spinlock> guard(lock());

        inode_type inode = get_inode(fd);
        std::weak_ptr<MMap_Region> & ptr = cache()[inode];

        std::shared_ptr<MMap_Region> result;
        result = ptr.lock();
        
        if (!result) {
            result.reset(new MMap_Region(fd, inode));
            ptr = result;
        }

        return result;
    }

    virtual ~MMap_Region()
    {
        if  (size) munmap((void *)start, size);

        std::lock_guard<ML::Spinlock> guard(lock());
        if (cache()[inode].expired())
            cache().erase(inode);
    }
};

/** Region that comes from a memory block. */
class File_Read_Buffer::Mem_Region : public File_Read_Buffer::Region {
public:
    Mem_Region(const char * start, size_t size,
               boost::function<void ()> onDone)
    {
        this->start = start;
        this->size = size;
        this->onDone = onDone;
    }

    boost::function<void ()> onDone;

    virtual ~Mem_Region()
    {
        if (onDone)
            onDone();
    }
};

File_Read_Buffer::Region::~Region()
{
}


/*****************************************************************************/
/* FILE_READ_BUFFER                                                          */
/*****************************************************************************/

File_Read_Buffer::File_Read_Buffer()
{
}

File_Read_Buffer::File_Read_Buffer(const std::string & filename)
{
    open(filename);
}

File_Read_Buffer::File_Read_Buffer(int fd)
{
    open(fd);
}

File_Read_Buffer::File_Read_Buffer(const char * start, size_t length,
                                   const std::string & filename,
                                   boost::function<void ()> onDone)
{
    open(start, length, filename, onDone);
}

File_Read_Buffer::File_Read_Buffer(const File_Read_Buffer & other)
    : filename_(other.filename_), region(other.region)
{
}

void File_Read_Buffer::open(const std::string & filename)
{
    int fd = ::open(filename.c_str(), O_RDONLY);

    if (fd == -1)
        throw Exception(errno, filename, "File_Read_Buffer::open()");

    try {
        region = MMap_Region::get(fd);
        filename_ = filename;
    }
    catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
}

void File_Read_Buffer::open(int fd)
{
    region = MMap_Region::get(fd);
    filename_ = get_name_from_fd(fd);
}

void File_Read_Buffer::open(const char * start, size_t length,
                            const std::string & filename,
                            boost::function<void ()> onDone)
{
    region.reset(new Mem_Region(start, length, onDone));
    filename_ = filename;
}

void File_Read_Buffer::close()
{
    region.reset();
    filename_ = "";
}

} // namespace ML
