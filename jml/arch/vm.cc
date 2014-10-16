/* vm.cc
   Jeremy Barnes, 22 February 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Virtual memory functions.
*/

#include "vm.h"
#include "jml/arch/format.h"
#include "jml/arch/exception.h"
#include "jml/utils/guard.h"
#include "jml/utils/info.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include <boost/bind.hpp>
#include <boost/crc.hpp>
#include <fstream>


using namespace std;


namespace ML {

std::string
Pagemap_Entry::
print() const
{
    if (present)
        return format("P%c %09lx s 2^%-5d", (swapped ? 'S' : '.'),
                      pfn, (int)shift);
    else if (swapped)
        return format(".S %02d/%06x s 2^%-5d", (int)swap_type,
                      (int)swap_offset, (int)shift);
    else return "..                 ";
}

std::string
Page_Info::
print_mapping() const
{
    return Pagemap_Entry::print();
}

std::string
Page_Info::
print_flags() const
{
    const char * letters = "lerudLaswRbmAShtHUPnk";
    string result;
    uint64_t val = 1;
    for (unsigned i = 0;  i < 22;  ++i) {
        bool f = flags & val;
        val <<= 1;
        if (f) result += letters[i];
        else result += '.';
    }
    
    result += format(" %06llx", (long long)flags);
    
    return result;
}

std::string
Page_Info::
print() const
{
    string result = print_mapping();
    if (pfn != 0 && (count != 0 || flags != 0))
        result = result + format("  c:%6lld ", (long long)count)
            + print_flags();
    return result;
}

std::vector<Page_Info> page_info(const void * addr, int npages)
{
    int pm_fd = open("/proc/self/pagemap", O_RDONLY);
    if (pm_fd == -1)
        throw Exception("open pagemap; " + string(strerror(errno)));
    Call_Guard close_pm_fd(boost::bind(::close, pm_fd));

    // Both of these might fail as these two files require special priviledges
    // to read
    int pf_fd = open("/proc/kpageflags", O_RDONLY);
    int pc_fd = open("/proc/kpagecount", O_RDONLY);

    // These will call guard with an fd of -1 if the files weren't open, which
    // won't hurt us
    Call_Guard close_pf_fd(boost::bind(::close, pf_fd), pf_fd != -1);
    Call_Guard close_pc_fd(boost::bind(::close, pc_fd), pc_fd != -1);

    size_t page_num = (size_t)addr / 4096;

    int res = lseek(pm_fd, page_num * 8, SEEK_SET);
    if (res == -1)
        throw Exception("page_info(): lseek: " + string(strerror(errno)));

    vector<Page_Info> result;
    for (unsigned i = 0;  i < npages;  ++i) {
        Page_Info info;

        int res = read(pm_fd, &info.mapping, 8);
        if (res != 8)
            throw Exception("read pm_fd");
     
        if (info.present) {
            if (pf_fd != -1) {
                int res = lseek(pf_fd, info.pfn * 8, SEEK_SET);
                if (res == -1)
                    throw Exception("lseek pf_fd");
                res = read(pf_fd, &info.flags, 8);
                if (res != 8)
                    throw Exception("read flags");
            }

            if (pc_fd != -1) {
                res = lseek(pc_fd, info.pfn * 8, SEEK_SET);
                if (res == -1)
                    throw Exception("lseek pc_fd");
                res = read(pc_fd, &info.count, 8);
                if (res != 8)
                    throw Exception("read count");
            }
        }

        result.push_back(info);
    }

    return result;
}

void dump_page_info(const void * start, const void * end,
                    std::ostream & stream)
{
    const char * p1 = (const char *)start;
    const char * p2 = (const char *)end;
    size_t dist = p2 - p1;
    size_t pages = dist / page_size;
    size_t rem   = dist % page_size;
    if (rem > 0) ++pages;

    if (pages > 10000)
        throw Exception("dump_page_info: too many pages");

    vector<Page_Info> info = page_info(start, pages);

    if (info.size() != pages)
        throw Exception("no pages");

    const char * p = page_start(p1);
    for (unsigned i = 0;  i < pages;  ++i, p += page_size) {
        

        stream << format("%04x %12p ", i, p)
               << info[i] << endl;
    }
}

std::vector<unsigned char> page_flags(const void * addr, int npages)
{
    int pm_fd = open("/proc/self/pagemap", O_RDONLY);
    if (pm_fd == -1)
        throw Exception("open pagemap; " + string(strerror(errno)));
    Call_Guard close_pm_fd(boost::bind(::close, pm_fd));

    size_t page_num = (size_t)addr / 4096;

    int res = lseek(pm_fd, page_num * 8, SEEK_SET);
    if (res == -1)
        throw Exception("page_info(): lseek: " + string(strerror(errno)));

    vector<unsigned char> result;
    result.reserve(npages);

    Page_Info info;

    uint64_t buf[1024];

    for (unsigned i = 0;  i < npages;  i += 1024) {
        int n = min<size_t>(1024, npages - i);

        int res = read(pm_fd, buf, 8 * n);

        if (res != 8 * n)
            throw Exception("read pm_fd");
     
        for (unsigned j = 0;  j < n;  ++j) {
            info.mapping = buf[j];
            result.push_back(info.present);
        }
    }

    return result;
    
}

void dump_maps(std::ostream & out)
{
    std::ifstream stream("/proc/self/maps");

    out << string(60, '=') << endl;
    out << "maps" << endl;

    while (stream) {
        string s;
        std::getline(stream, s);
        out << s << endl;
    }

    out << string(60, '=') << endl;
    out << endl << endl;
}

/*****************************************************************************/
/* PAGEMAP_READER                                                            */
/*****************************************************************************/

Pagemap_Reader::
Pagemap_Reader(const char * mem, size_t bytes,
               Pagemap_Entry * entries, int fd)
    : fd(fd), mem(mem), entries(entries),
      delete_entries(entries == 0), close_fd(fd == -1)
{
    npages = to_page_num(mem + bytes) - to_page_num(mem);

#if 0
    cerr << "mem = " << (const void *)mem << endl;
    cerr << "mem + bytes = " << (const void *)(mem + bytes) << endl;
    cerr << "page1 = " << to_page_num(mem + bytes) << endl;
    cerr << "page2 = " << to_page_num(mem) << endl;
    cerr << "bytes = " << bytes << endl;
    cerr << "npages = " << npages << endl;
    cerr << "pagemap_reader: fd = " << fd << endl;
#endif

    if (close_fd)
        this->fd = open("/proc/self/pagemap", O_RDONLY);
    if (this->fd == -1)
        throw Exception(errno, "Pagemap_Reader()",
                        "open(\"proc/self/pagemap\", O_RDONLY)");
    Call_Guard do_close_fd(boost::bind(close, this->fd), close_fd);

    if (delete_entries)
        this->entries = new Pagemap_Entry[npages];

    try {
        update();
    } catch (...) {
        if (delete_entries) delete[] entries;
        throw;
    }

    //cerr << "pagemap_reader init " << this 
    //     << ": entries = " << entries << " this->entries = "
    //     << this->entries << " delete_entries = " << delete_entries
    //     << " fd = " << this->fd << endl;

    do_close_fd.clear();
}

Pagemap_Reader::
~Pagemap_Reader()
{
    //cerr << "pagemap_reader exit " << this 
    //     << ": entries = " << entries << " this->entries = "
    //     << this->entries << " delete_entries = " << delete_entries << endl;

    if (delete_entries)
        delete[] entries;

    if (close_fd && fd != -1) {
        int res = close(fd);
        if (res == -1)
            cerr << "~Pagemap_Reader(): close on fd: " << strerror(errno)
                 << endl;
    }

    entries = 0;
}

size_t
Pagemap_Reader::
update(ssize_t first_page, ssize_t last_page)
{
    if (last_page == -1)
        last_page = npages;

    if (first_page < 0 || last_page > npages || last_page < first_page)
        throw Exception("Pagemap_Reader::update(): pages out of range");

    // Where do we seek to in the pagemap file?
        
    // Counts the number of modified page map entries
    size_t result = 0;

    size_t CHUNK = 1024;  // pages at a time

    // Buffer to read them into
    Pagemap_Entry buf[CHUNK];

    size_t base_page_num = to_page_num(mem);

    //cerr << "update: first_page = " << first_page
    //     << " last_page = " << last_page << endl;

    // Update a chunk at a time
    for (size_t page = first_page;  page < last_page;  /* no inc */) {
        size_t limit = std::min<size_t>(page + CHUNK, last_page);
        size_t todo = limit - page;

        //cerr << "page = " << page << " last_page = " << last_page
        //     << " limit = " << limit << " todo " << todo << endl;

        // Where to seek in the pagemap file?
        off_t seek_pos = (base_page_num + page) * sizeof(Pagemap_Entry);
        
        //cerr << "seek_pos = " << seek_pos << " base_page_num = "
        //     << base_page_num << endl;

        ssize_t res = pread(fd, buf,
                            todo * sizeof(Pagemap_Entry),
                            seek_pos);

        if (res <= 0) {
            cerr << "fd: " << fd << " filename: " << fd_to_filename(fd)
                 << endl;
        }

        if (res == -1)
            throw Exception(errno, "Pagemap_Reader::update()", "pread");
        if (res <= 0)
            throw Exception("Pagemap_Reader::update(): nothing read "
                            "from pagemap file");
        
        //cerr << "res = " << res << endl;

        res /= sizeof(Pagemap_Entry);  // convert bytes to objects

        //cerr << "read " << res << " objects" << endl;

        for (unsigned i = 0;  i < res;  ++i) {
            result += this->entries[page + i] != buf[i];
            this->entries[page + i] = buf[i];
        }
        
        page += res;
    }
    
    return result;
}

void
Pagemap_Reader::
dump(std::ostream & stream) const
{
    const char * p = mem;
    for (unsigned i = 0;  i < npages;  ++i, p += page_size) {
        boost::crc_32_type calc_crc;

        uint32_t crc = 0;
        if (entries[i].present && !entries[i].swapped) {
            calc_crc.process_bytes(p, page_size);
            crc = calc_crc.checksum();
        }

        stream << format("%04x %12p ", i, p)
               << entries[i];
        if (entries[i].present && !entries[i].swapped)
            stream << format("%08x", crc);
        stream << endl;
    }
}


} // namespace ML


