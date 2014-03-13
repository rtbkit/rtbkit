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

    vector<string> inputFiles;
    vector<string> outputFiles;
    string s3KeyId;
    string s3Key;
    
    po::options_description desc("Main options");
    desc.add_options()
        ("input-uri,i", po::value(&inputFiles), "Input files/uris (can have multiple file/s3://bucket/object)")
        ("output-uri,o", po::value(&outputFiles), "Output files/uris (can have multiple file/s3://bucket/object)")
        ("s3-key-id,I", po::value<string>(&s3KeyId), "S3 key id")
        ("s3-key,K", po::value<string>(&s3Key), "S3 key")
        ("help,h", "Produce help message");
    
    po::positional_options_description pos;
    pos.add("input-uri", -1);
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
        return showHelp ? 0 : 1;
    }

    if (outputFiles.empty())
        outputFiles.push_back("-");

    if (!s3KeyId.empty()) {
        for (auto f: outputFiles){
            if(f.substr(0, 5) == "s3://"){
                size_t pos = f.substr(5).find("/");
                registerS3Bucket(f.substr(5, pos), s3KeyId, s3Key);
            }
        }
        for (auto f: inputFiles){
            if(f.substr(0, 5) == "s3://"){
                size_t pos = f.substr(5).find("/");
                registerS3Bucket(f.substr(5, pos), s3KeyId, s3Key);
            }
        }
    }

    std::vector<filter_ostream> outStreams;

    outStreams.reserve(outputFiles.size());
    for (auto f: outputFiles)
        outStreams.emplace_back(f);

    size_t bufSize = 4096 * 16;
    char buf[bufSize];

    for (auto f: inputFiles) {
        if (f == "-") {

            for (;;) {
                ssize_t res = read(0, buf, bufSize);
                if (res == 0)
                    break;
                if (res == -1 && errno == EINTR)
                    continue;
                if (res == -1)
                    throw ML::Exception(errno, "read");
                for (unsigned s = 0;  s < outStreams.size();  ++s) {
                    if (outputFiles[s] == "-") {
                        //res = write(0, buf, bufSize);
                        outStreams[s].write(buf, res);
                        outStreams[s] << std::flush;
                    }
                    else {
                        outStreams[s].write(buf, res);
                    }
                }
            }
        }
        else {
            filter_istream stream(f);
            while (stream) {
                stream.read(buf, bufSize);
                ssize_t read = stream.gcount();

                for (unsigned s = 0;  s < outStreams.size();  ++s) {
                    if (outputFiles[s] == "-") {
                        //res = write(0, buf, bufSize);
                        outStreams[s].write(buf, read);
                        outStreams[s] << std::flush;
                    }
                    else {
                        outStreams[s].write(buf, read);
                    }
                }
            }
        }
    }
    return 0;
}

