/** sns_send.cc
    Jeremy Barnes, 24 December 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Send an SNS message.
*/

#include "soa/service/sns.h"
#include "jml/utils/filter_streams.h"
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp> 
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

namespace po = boost::program_options;

using namespace std;
using namespace Datacratic;
using namespace ML;

int main(int argc, char* argv[])
{
    ios::sync_with_stdio(true);

    string accessKeyId;
    string accessKey;
    string topicArn;
    string subject;
    string message = "@-";
    int timeout = 60;
    
    po::options_description desc("Main options");
    desc.add_options()
        ("topic-arn,t", po::value(&topicArn), "topic to read from ")
        ("message,m", po::value(&message)->default_value(message),
         "Message (@ = read from file, @- = read from stdin)")
        ("subject,s", po::value(&subject)->default_value(subject),
         "subject of message")
        ("access-key-id,I", po::value<string>(&accessKeyId), "Access key id")
        ("access-key,K", po::value<string>(&accessKey), "Access key")
        ("timeout,T", po::value(&timeout)->default_value(timeout),
         "timeout in seconds to send message")
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

    if (message.empty())
        throw ML::Exception("must specify a message");
    if (message[0] == '@') {
        ML::filter_istream stream(string(message, 1));
        string msg;
        while (stream) {
            char buf[4096];
            stream.read(buf, 4096);
            int c = stream.gcount();
            msg += string(buf, buf + c);
        }

        message = msg;
    }

    SnsApi api(accessKeyId, accessKey);
    string id = api.publish(topicArn, message, timeout, subject);
    cout << id << endl;
    return 0;
}

