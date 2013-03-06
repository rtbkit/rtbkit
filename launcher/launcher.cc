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

    configuration_options.add_options()
        ("file,F", value(&filename), "Filename of the launch sequence")
        ("node,N", value(&node), "Name of the current node")
        ("script,S", value(&script), "Filename of the launch script sequence to generate");

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

    auto & service = Datacratic::Launcher::Service::get();
    service.run(root, node, script);
}

