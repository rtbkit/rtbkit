/* cpu_info.cc
   Jeremy Barnes, 22 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

*/

#include "cpu_info.h"

#include <fstream>
#include <iostream>


using namespace std;


namespace ML {

int num_cpus_result = 0;

void init_num_cpus()
{
    ifstream stream("/proc/cpuinfo");

    int ncpus = 0;

    while (stream) {
        string line;
        getline(stream, line);

        //cerr << "line = " << line << endl;

        string::size_type found = line.find("processor");
        //cerr << "found = " << found << endl;
        if (found == 0) ++ncpus;
    }

    //cerr << "ncpus = " << ncpus << endl;

    if (ncpus == 0)
        ncpus = 1;
    
    num_cpus_result = ncpus;

    //cerr << "num_cpus_result = " << num_cpus_result << endl;
}

} // namespace ML

