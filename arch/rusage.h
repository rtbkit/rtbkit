/* rusage.h                                                        -*- C++ -*-
   Jeremy Barnes, 26 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Small structure to track resource usage.
*/

#pragma once

namespace ML {

/** Resource usage structure. */
struct RUsage {

    RUsage()
    {
        clear();
    }

    RUsage(const struct rusage & res)
    {
        userTime = res.ru_utime.tv_sec + (res.ru_utime.tv_usec / 1000000.0);
        systemTime = res.ru_stime.tv_sec + (res.ru_stime.tv_usec / 1000000.0);
        minorFaults = res.ru_minflt;
        majorFaults = res.ru_majflt;
        voluntaryContextSwitches = res.ru_nvcsw;
        involuntaryContextSwitches = res.ru_nivcsw;
    }

    void clear()
    {
        userTime = systemTime = minorFaults = majorFaults
            = voluntaryContextSwitches = involuntaryContextSwitches
            = 0;
    }

    double userTime;
    double systemTime;
    long minorFaults;
    long majorFaults;
    long voluntaryContextSwitches;
    long involuntaryContextSwitches;
    
    void getForCurrentProcess()
    {
        struct rusage rusage;
        getrusage(RUSAGE_SELF, &rusage);  // TODO: check return code...
        *this = RUsage(rusage);
    }

    RUsage operator - (const RUsage & other) const
    {
        RUsage result;
        result.userTime = userTime - other.userTime;
        result.systemTime = systemTime - other.systemTime;
        result.minorFaults = minorFaults - other.minorFaults;
        result.majorFaults = majorFaults - other.majorFaults;
        result.voluntaryContextSwitches
            = voluntaryContextSwitches - other.voluntaryContextSwitches;
        result.involuntaryContextSwitches
            = involuntaryContextSwitches - other.involuntaryContextSwitches;

        return result;
    }

    Json::Value toJson() const
    {
        Json::Value result;

        result["userTime"] = userTime;
        result["systemTime"] = systemTime;
        result["totalTime"] = userTime + systemTime;
        result["minorFaults"] = (int)minorFaults;
        result["majorFaults"] = (int)majorFaults;
        result["totalFaults"] = int(minorFaults + majorFaults);
        result["voluntaryContextSwitches"] = (int)voluntaryContextSwitches;
        result["involuntaryContextSwitches"] = (int)involuntaryContextSwitches;
        result["contextSwitches"]
            = int(voluntaryContextSwitches + involuntaryContextSwitches);

        return result;
    }
};

} // namespace ML
