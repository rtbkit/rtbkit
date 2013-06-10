/* s3_copy.cc
   Jeremy Barnes, 9 June 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Copy a large object from on S3 location to another.
*/

#include "soa/service/s3.h"
#include "jml/utils/filter_streams.h"
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp> 
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <iostream>

namespace po = boost::program_options;

using namespace std;
using namespace Datacratic;
using namespace ML;

int main(int argc, char* argv[])
{
    ios::sync_with_stdio(true);

    string inputUri;
    vector<string> outputFiles;
    string localFile;
    string s3KeyId;
    string s3Key;
    
    string compression = "none";
    
    po::options_description desc("Main options");
    desc.add_options()
        ("input-uri,source,s", po::value(&inputUri)->required(), "Input S3 URI s3://bucket/object)")
        ("output-uri,dest,o,d", po::value(&outputFiles), "Output files/uris (can have multiple file/s3://bucket/object)")
        ("id,i", po::value<string>(&s3KeyId), "S3 access id")
        ("key,k", po::value<string>(&s3Key), "S3 access id key")
        ("s3-key-id,I", po::value<string>(&s3KeyId), "S3 access id")
        ("s3-key,K", po::value<string>(&s3Key), "S3 access id key")
        ("compression,c", po::value<string>(&compression), "Compression to apply (default: none, valid: auto,gz,bz2,xz")
        ("help,h", "Produce help message");
    
    po::positional_options_description pos;
    pos.add("input-uri", 1);
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

    if (compression == "auto")
        compression = "";

    //If one of the options is set to 'help'...
    if (showHelp || vm.count("help")){
        //Display the options_description
        cout << "usage : s3cp [options] [source] [dest]+" << endl;
        cout << desc << endl;
        return showHelp ? 1 : 0;
    }
    
    if (s3KeyId != "")
        registerS3Buckets(s3KeyId, s3Key);

    ML::filter_istream in(inputUri, ios::in, compression);

    std::vector<filter_ostream> streams;
    streams.reserve(outputFiles.size() + 1);

    for (auto f: outputFiles)
        streams.emplace_back(f, ios::out, compression);

    Date start = Date::now();
    size_t bytesDone = 0;
    size_t bufSize = 4096 * 1024;
    char buf[bufSize];

    while (in) {
        in.read(buf, bufSize);
        ssize_t res = in.gcount();
        for (unsigned s = 0;  s < streams.size();  ++s)
            streams[s].write(buf, res);
        bytesDone += res;

        double elapsed = Date::now().secondsSince(start);
        cerr << "done " << bytesDone / 1000000 << "MB in "
             << elapsed << "s at "
             << bytesDone / 1000000 / elapsed
             << " Mbytes/second" << endl;
    }

    return 0;
}
