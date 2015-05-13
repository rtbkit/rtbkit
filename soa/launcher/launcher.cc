/* launcher.cc                                        -*- C++ -*-
   Eric Robert, 29 February 2013
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Tool to launch the stack.
*/


#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp>
#include <fstream>

#include "soa/launcher/launcher.h"

int main(int argc, char ** argv)
{
    using namespace boost::program_options;
    options_description configuration_options("Configuration options");

    std::string filename;
    std::string node;
    std::string script;
    std::string bin;
    bool launch = false;
    bool master = false;

    configuration_options.add_options()
        ("file,F", value(&filename), "filename of the launch sequence")
        ("launch,L", value(&launch)->zero_tokens(), "run the launch monitoring process?")
        ("master,M", value(&master)->zero_tokens(), "specify that this will be the master node?")
        ("node,N", value(&node), "name of the current node")
        ("script,S", value(&script), "filename of the launch script sequence to generate and use")
        ("bin,P", value(&bin), "location of the binaries");

    options_description all_opt;
    all_opt.add(configuration_options);
    all_opt.add_options()
        ("help,h", "print this message");

    positional_options_description p;
    p.add("file", -1);

    variables_map vm;
    store(command_line_parser(argc, argv).options(all_opt).positional(p).run(), vm);
    notify(vm);

    if(vm.count("help")) {
        std::cerr << all_opt << std::endl;
        exit(1);
    }

    if(filename.empty()) {
        std::cerr << "configuration file is required" << std::endl;
        exit(1);
    }

    if(node.empty()) {
        std::cerr << "current node is required" << std::endl;
        exit(1);
    }

    std::ifstream file(filename);
    if(!file) {
        std::cerr << "failed to open file " << filename << std::endl;
    }

    Json::Reader reader;
    Json::Value root;
    if(!reader.parse(file, root)) {
        std::cerr << "cannot read file '" << filename << "'" << std::endl;
        std::cerr << reader.getFormattedErrorMessages();
        exit(1);
    }

    if(bin.empty()) {
        auto env = getenv("BIN");
        if(env) {
            bin = env;
        }

        if(bin.empty()) {
            bin = "./build/x86_64/bin";
        }
    }

    Datacratic::Launcher::Service::get().run(root, node, filename, script, launch, master, bin);

    if(launch) {
        int res = system(script.c_str());
        if(res == -1) {
            std::cerr << "cannot launch script" << std::endl;
            exit(1);
        } 
    }
}

