/* vm.cc
   Jeremy Barnes, 22 February 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Virtual memory functions.
*/

#include "vm.h"
#include "jml/arch/format.h"
#include "jml/arch/exception.h"
#include "jml/utils/guard.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include <boost/bind.hpp>
#include <fstream>


using namespace std;


namespace ML {

std::string
Page_Info::
print_mapping() const
{
    if (present)
        return format("P%c %09x s%5d", (swapped ? 'S' : '.'),
                      pfn, (int)shift);
    else if (swapped)
        return format(".S %02d/%06x s%5d", (int)swap_type, (int)swap_offset);
    else return "..                 ";
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
    
    result += format(" %06x", flags);
    
    return result;
}

std::string
Page_Info::
print() const
{
    string result = print_mapping();
    if (pfn != 0 && (count != 0 || flags != 0))
        result = result + format("  c:%6d ", count)
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
    Call_Guard close_pf_fd(boost::bind(::close, pf_fd));
    Call_Guard close_pc_fd(boost::bind(::close, pc_fd));

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
        stream << format("%04x %012p ", i, p)
               << info[i] << endl;
    }
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

} // namespace ML


