/** s3tee.cc
    Jeremy Barnes, 4 September 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Implementation of the "tee" command that can write its data to an
    s3 bucket.
*/

#include "soa/service/s3.h"
#include "jml/utils/filter_streams.h"
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp> 
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <poll.h>

namespace po = boost::program_options;

using namespace std;
using namespace Datacratic;
using namespace ML;

int main(int argc, char* argv[])
{
    ios::sync_with_stdio(true);

    vector<string> outputFiles;
    string localFile;
    string s3KeyId;
    string s3Key;
    
    po::options_description desc("Main options");
    desc.add_options()
        ("output-uri,o", po::value(&outputFiles), "Output files/uris (can have multiple file/s3://bucket/object)")
        ("s3-key-id,I", po::value(&s3KeyId), "S3 key id")
        ("s3-key,K", po::value(&s3Key), "S3 key")
        ("help,h", "Produce help message");
    
    po::positional_options_description pos;
    pos.add("output-uri", -1);
    po::variables_map vm;
    bool showHelp = false;

    try{
        po::parsed_options parsed = po::command_line_parser(argc, argv)
            .options(desc)
            .positional(pos)
            .run();
        po::store(parsed, vm);
        po::notify(vm);
    }catch(const std::exception & exc){
        //invalid command line param
        cerr << "command line parsing error: " << exc.what() << endl;
        showHelp = true;
    }

    //If one of the options is set to 'help'...
    if (showHelp || vm.count("help")){
        //Display the options_description
        cout << desc << "\n";
        return 1;
    }

    for (auto f: outputFiles){
        if(f.substr(0, 5) == "s3://"){
            size_t pos = f.substr(5).find("/");
            if (s3KeyId != "")
                registerS3Bucket(f.substr(5, pos), s3KeyId, s3Key);
        }
    }

    std::vector<filter_ostream> streams;
    streams.reserve(outputFiles.size() + 1);

    streams.emplace_back("-");

    for (auto f: outputFiles)
        streams.emplace_back(f);

    size_t bufSize = 4096 * 16;
    char buf[bufSize];

    for (;;) {
        ssize_t res = read(0, buf, bufSize);
        if (res == 0)
            break;
        if (res == -1 && errno == EINTR)
            continue;
        if (res == -1)
            throw ML::Exception(errno, "read");
        for (unsigned s = 0;  s < streams.size();  ++s)
            streams[s].write(buf, res);
        streams[0] << std::flush;
    }

    return 0;
}

