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
using namespace ML;
using namespace Datacratic;

int main(int argc, char* argv[]) {

    po::options_description desc("Allowed options");

    string id = "";
    string key = "";
    string source = "";
    string dest = "";
    
    desc.add_options()
        ("source,s", po::value<string>(&source)->required(), "Source file.")
        ("dest,d", po::value<string>(&dest)->required(), "Destination file.")
        ("id,i", po::value<string>(&id), "S3 access id")
        ("key,k", po::value<string>(&key), "S3 access id key")
        ("help,h", "Produce help message");

    po::positional_options_description pos;
    pos.add("source", 1);
    pos.add("dest", 1);
    po::variables_map vm;
    bool showHelp = false;
    try {
        po::parsed_options parsed = po::command_line_parser(argc, argv)
            .options(desc)
            .positional(pos)
            .run();
        po::store(parsed, vm);
    } catch (...) {
        // invalid command line param
        showHelp = true;
    }

    //If one of the options is set to 'help'...
    if (showHelp || vm.count("help")) {
        //Display the options_description
        cout << "usage : s3cp [options] [source] [dest]" << endl;
        cout << desc << endl;
        return showHelp ? 1 : 0;
    }    

    po::notify(vm);

    if (id != "")
        registerS3Buckets(id, key);

    filter_istream in(source);
    filter_ostream out(dest);

    int buf_size = 1024 * 1024;
    char buf[buf_size];

    while (in) {
        in.read(buf, buf_size);
        out.write(buf, in.gcount());
    }

    out.flush();

    return 0;
}
