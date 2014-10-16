/* proc_stats.h                                                   -*- C++ -*-
   Rémi Attab, 19 January 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Gathers process and system stats from the proc files.
*/


#include "soa/service/process_stats.h"
#include "jml/arch/exception.h"

#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include <vector>
#include <array>
#include <functional>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <sys/resource.h>


using namespace Datacratic;
using namespace std;
using namespace boost;


namespace {

static const string ProcStatFile = "/proc/self/stat";
enum ProcStatFields {
    STAT_MINFLT = 9,
    STAT_MAJFLT = 11,
    STAT_UTIME = 13,
    STAT_STIME = 14,
    STAT_VSIZE = 22,
    STAT_RSS = 23
};

static const string ProcStatmFile = "/proc/self/statm";
enum ProcStatmFields {
    STATM_SIZE = 0,
    STATM_RESIDENT = 1,
    STATM_SHARED = 2,
};


static const string ProcLoadAvgFile = "/proc/loadavg";
enum ProcLoadAvgFields {
    LOADAVG_1 = 0, 
    LOADAVG_5 = 1,
    LOADAVG_15 = 2
};


vector<string> readProcFile(const string& procFile) {
    ifstream ifs(procFile);
    if (ifs.fail()) {
        throw ML::Exception ("Unable to open proc file " + procFile);
    }

    std::array<char, 1024> buffer;
    ifs.getline(&buffer[0], buffer.max_size());
    if (ifs.fail() || ifs.eof()) {
        throw ML::Exception ("Unable to read proc file " + procFile);
    }

    std::string rawStats = &buffer[0];

    vector<string> stats;
    split(stats, rawStats, [](char rhs)->bool {return rhs == ' ';});

    return stats;
}

};

void ProcessStats::sampleLoadAverage () {
    if (!doLoadAverage) {
        return;
    }

    vector<string> stats = readProcFile(ProcLoadAvgFile);

    loadAverage1 = lexical_cast<float>(stats[LOADAVG_1]);
    loadAverage5 = lexical_cast<float>(stats[LOADAVG_5]);
    loadAverage15 = lexical_cast<float>(stats[LOADAVG_15]);
}

//! \deperecated by sampleRUsage and sampleStatm
void ProcessStats::sampleStat () {
    vector<string> stats = readProcFile(ProcStatFile);

    long ticks = sysconf(_SC_CLK_TCK);

    minorFaults = lexical_cast<uint64_t>(stats[STAT_MINFLT]);
    majorFaults = lexical_cast<uint64_t>(stats[STAT_MAJFLT]);

    userTime = lexical_cast<uint64_t>(stats[STAT_UTIME]) / ((double) ticks);
    systemTime = lexical_cast<uint64_t>(stats[STAT_STIME]) / ((double) ticks);

    virtualMem = lexical_cast<uint64_t>(stats[STAT_VSIZE]);
    residentMem = lexical_cast<uint64_t>(stats[STAT_RSS]);
}

void ProcessStats::sampleStatm () {
    vector<string> stats = readProcFile(ProcStatmFile);

    int pageSize = getpagesize();
    virtualMem = lexical_cast<uint64_t>(stats[STATM_SIZE]) * pageSize;
    residentMem = lexical_cast<uint64_t>(stats[STATM_RESIDENT]) * pageSize; 
    sharedMem = lexical_cast<uint64_t>(stats[STATM_SHARED]) * pageSize;
}

void ProcessStats::sampleRUsage () {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);

    userTime = ru.ru_utime.tv_sec + (ru.ru_utime.tv_usec / 1000000.0);
    systemTime = ru.ru_stime.tv_sec + (ru.ru_stime.tv_usec / 1000000.0);

    minorFaults = ru.ru_minflt;
    majorFaults = ru.ru_majflt;

    voluntaryContextSwitches = ru.ru_nvcsw;
    involuntaryContextSwitches = ru.ru_nivcsw;
}


void ProcessStats::logToCallback (
        LogCallback cb, 
        const ProcessStats& last,
        const ProcessStats& cur,
        const string& prefix)
{
    std::string p = !prefix.empty() ? prefix + "." : "";

    cb(p + "timeUser", cur.userTime - last.userTime);
    cb(p + "timeSystem", cur.systemTime - last.systemTime);

    cb(p + "faultsMinor", cur.minorFaults - last.minorFaults);
    cb(p + "faultsMajor", cur.majorFaults - last.majorFaults);

    cb(p + "memVirtual", cur.virtualMem);
    cb(p + "memResident", cur.residentMem);
    cb(p + "memShared", cur.sharedMem);

    cb(p + "contextSwitchesVoluntary",
            cur.voluntaryContextSwitches - last.voluntaryContextSwitches);
    cb(p + "contextSwitchesInvoluntary",
            cur.involuntaryContextSwitches - last.involuntaryContextSwitches);
    
    if (cur.doLoadAverage) {
        cb(p + "loadAverage01", cur.loadAverage1);
        cb(p + "loadAverage05", cur.loadAverage5);
        cb(p + "loadAverage15", cur.loadAverage15);
    }
}


Json::Value ProcessStats::toJson (
        const ProcessStats& last, 
        const ProcessStats& cur, 
        const string& prefix) 
{
    Json::Value result;

    auto cb = [&](string name, double val) {
        result[name] = (long long)val;
    };
    logToCallback(cb, last, cur, prefix);

    return result;
}
