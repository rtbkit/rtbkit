/* simd.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Implementation of SIMD code.
*/

#include "simd.h"
#include <boost/tuple/tuple.hpp>
#include "jml/arch/exception.h"
#include <iostream>
#include <iomanip>

using namespace std;


namespace ML {

#if defined __i686__ || defined __amd64__

namespace {

enum {
    CPUID_VENDOR_ID = 0,
    CPUID_LEVEL = 0,
    CPUID_FEATURES = 1,
    CPUID_CACHE_INFO = 2,
    CPUID_SERIAL_NUMBER = 3,
    CPUID_EXT_CACHE_INFO = 4,
    CPUID_MONITOR_MWAIT = 5,
    CPUID_THERMAL_POWER = 6,
    CPUID_DCA_ACCESS = 7,
    CPUID_EXT_LEVEL =      0x80000000,
    CPUID_EXT_FEATURES =   0x80000001,
    CPUID_EXT_BRAND1 =     0x80000002,
    CPUID_EXT_BRAND2 =     0x80000003,
    CPUID_EXT_BRAND3 =     0x80000004,
    CPUID_EXT_L1CACHE =    0x80000005,
    CPUID_EXT_L2CACHE =    0x80000006,
    CPUID_EXT_APM_INFO =   0x80000007,
    CPUID_EXT_ADDR_SIZES = 0x80000008,
    CPUID_EXT_SVM_INFO =   0x8000000A
};

struct Regs {
    uint32_t eax, ebx, ecx, edx;
};

Regs
cpuid(uint32_t request, uint32_t ecx = 0)
{
    Regs result = {0, 0, 0, 0};
    asm volatile
        (
#if defined(__i686__)         
         "sub  $0x40,  %%esp\n\t"
         "push %%ebx\n\t"
         "push %%edx\n\t"
#else
         "sub  $0x40,  %%rsp\n\t"
         "push %%rbx\n\t"
         "push %%rdx\n\t"
#endif
         "cpuid\n\t"

         "mov  %%eax,  0(%[addr])\n\t"
         "mov  %%ebx,  4(%[addr])\n\t"
         "mov  %%ecx,  8(%[addr])\n\t"
         "mov  %%edx, 12(%[addr])\n\t"
#if defined(__i686__)         
         "pop  %%edx\n\t"
         "pop  %%ebx\n\t"
         "add  $0x40,   %%esp\n\t"
#else
         "pop  %%rdx\n\t"
         "pop  %%rbx\n\t"
         "add  $0x40,   %%rsp\n\t"
#endif
         : "+a" (request), "+c" (ecx)
         : [addr] "S" (&result)
         : "cc", "memory"
         );
    return result;
}

} // file scope

uint32_t cpuid_flags()
{
    return cpuid(1).edx;
}

namespace {

std::string to_ascii(uint32_t x)
{
    std::string result(4, ' ');
    for (unsigned i = 0;  i < 4;  ++i)
        result[i] = x >> (i * 8);
    return result;
}

} // file scope

std::string vendor_id()
{
    Regs r = cpuid(CPUID_VENDOR_ID);
    return to_ascii(r.ebx) + to_ascii(r.edx) + to_ascii(r.ecx);
}

std::string model_id()
{
    uint32_t cpuid_extlevel = cpuid(CPUID_EXT_LEVEL).eax;

    cerr << "cpuid_extlevel = " << setw(16) << cpuid_extlevel << endl;

    if (cpuid_extlevel < 0x80000000 || cpuid_extlevel > 0x8000ffff)
        return "";  // no model if no extended CPUID

    std::string result;
    for (unsigned i = 0;  i < 3;  ++i) {
        Regs r = cpuid(CPUID_EXT_BRAND1 + i);
        result
            +=  to_ascii(r.eax) + to_ascii(r.ebx)
              + to_ascii(r.ecx) + to_ascii(r.edx);
    }

    //cerr << "model_id = " << result << endl;

    if (result.size() != 48)
        throw Exception("model_id(): invalid model ID");
    
    if (result[47] != 0)
        throw Exception("model_id(): invalid model ID terminator");

    return result.c_str();  // truncate to null terminator
}

CPU_Info::CPU_Info()
{
    cpuid_level = cpuid_extlevel = standard1 = standard2 = extended = amd = 0;

    cpuid_level = cpuid(CPUID_LEVEL).eax;
    cpuid_extlevel = cpuid(CPUID_EXT_LEVEL).eax;

    //cerr << "cpuid level = " << cpuid_level << endl;
    //cerr << "cpuid ext level = " << cpuid_extlevel - 0x80000000 << endl;

    if (cpuid_extlevel < 0x80000000 || cpuid_extlevel > 0x8000ffff)
        cpuid_extlevel = 0;

    vendor = vendor_id();
    model = model_id();
    
    Regs r = cpuid(CPUID_FEATURES);
    standard1 = r.edx;
    standard2 = r.ecx;

    if (cpuid_extlevel >= CPUID_EXT_FEATURES) {
        r = cpuid(CPUID_EXT_FEATURES);
        extended = r.edx;
        amd = r.ecx;
    }

#if 0
    if (fpu) cerr << "fpu ";

    if (vme) cerr << "vme ";
    if (de) cerr << "de ";
    if (pse) cerr << "pse ";
    if (tsc) cerr << "tsc ";
    if (msr) cerr << "msr ";
    if (pae) cerr << "pae ";
    if (mce) cerr << "mce ";
    if (cx8) cerr << "cx8 ";
    if (apic) cerr << "apic ";
    if (sep) cerr << "sep ";
    if (mtrr) cerr << "mtrr ";
    if (pge) cerr << "pge ";
    if (mca) cerr << "mca ";
    if (cmov) cerr << "cmov ";
    if (pat) cerr << "pat ";
    if (pse36) cerr << "pse36 ";
    if (psn) cerr << "psn ";
    if (clflush) cerr << "clflush ";
    if (dts) cerr << "dts ";
    if (acpi) cerr << "acpi ";
    if (mmx) cerr << "mmx ";
    if (fxsr) cerr << "fxsr ";
    if (sse) cerr << "sse ";
    if (sse2) cerr << "sse2 ";
    if (ss) cerr << "ss ";
    if (ht) cerr << "ht ";
    if (tm) cerr << "tm ";
    if (ia64) cerr << "ia64 ";
    if (pbe) cerr << "pbe ";

    if (sse3) cerr << "sse3 ";
    if (monitor) cerr << "monitor ";
    if (ds_cpl) cerr << "ds_cpl ";
    if (vmx) cerr << "vmx ";
    if (est) cerr << "est ";
    if (tm2) cerr << "tm2 ";
    if (pni) cerr << "pni ";
    if (cid) cerr << "cid ";
    if (cx16) cerr << "cx16 ";
    if (xtpr) cerr << "xtpr ";
    if (dca) cerr << "dca ";

    if (syscall) cerr << "syscall ";
    if (nx) cerr << "nx ";       // 20
    if (mmxext) cerr << "mmxext ";
    if (ffxsr) cerr << "ffxsr ";
    if (rdtscp) cerr << "rdtscp ";   
    if (lm) cerr << "lm ";
    if (threednowext) cerr << "3nowext ";
    if (threednow) cerr << "3dnow "; 

    if (lahfsahf) cerr << "lahfsahf ";
    if (svm) cerr << "svm ";
    if (cmplegacy) cerr << "cmplegacy ";
    if (altmovcr8) cerr << "altmovcr8 ";

    cerr << endl;
#endif
}

#if 0
std::string
CPU_Info::
print_flags()
{
    //static const char * STANDARD1_NAMES = { };
    return "";
}
#endif

CPU_Info * static_cpu_info = 0;

#endif // __i686__

} // namespace ML
