#include "soa/service/s3.h"
#include "jml/utils/file_functions.h"
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp> 
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace std;

int main(int argc, char* argv[]){
    po::options_description desc("Allowed options");

    string localFile = "";
    string id = "";
    string key = "";
    string bucket = "";
    string s3File = "";
    string direction = "";
    int maxSizeKB = -1;
    
    desc.add_options()
        ("help,h", "Produce help message")
        ("localfile,l", po::value<string>(&localFile), "Local file")
        ("id,i", po::value<string>(&id), "S3 id")
        ("key,k", po::value<string>(&key), "S3 key")
        ("bucket,b", po::value<string>(&bucket), "S3 bucket")
        ("s3file,s", po::value<string>(&s3File), "S3 file key")
        ("direction,d", po::value<string>(&direction), 
            "Direction (u for upload, d for download)")
        ("maxsizekb,m", po::value<int>(&maxSizeKB), 
            "If specified, at most {maxsizebk} BK will be transferred. (Download only)");


    po::positional_options_description pos;
    pos.add("output", -1);
    po::variables_map vm;
    bool showHelp = false;
    try{
        po::parsed_options parsed = po::command_line_parser(argc, argv)
            .options(desc)
            .positional(pos)
            .run();
        po::store(parsed, vm);
        po::notify(vm);

        if(direction != "u" && direction != "d"){
            cout << "Invalid direction\n";
            showHelp = true;
        }
    }catch(...){
        //invalid command line param
        showHelp = true;
    }

    //If one of the options is set to 'help'...
    if (showHelp || vm.count("help")){
        //Display the options_description
        cout << desc << "\n";
        return showHelp ? 1 : 0;
    }else if(localFile.length() == 0 || 
            id.length() == 0 || 
            key.length() == 0 ||
            bucket.length() == 0 ||
            s3File.length() == 0
    ){
        cout << "You need to specify all parameters except help and maxsizekb. " 
                << "Run with \"-h\" to list them.\n";
        return 1;
    }
    
    try{
        if(direction == "u"){
            //upload
            cout << "File: " << localFile 
                    << " - Size: " << fs::file_size(localFile) <<  "\n";
            ML::File_Read_Buffer frb(localFile);
            Datacratic::S3Api s3(id, key);
            string result = s3.upload(
                    frb.start(), 
                    fs::file_size(localFile), 
                    bucket, "/" + s3File); 
            cout << result << "\n";
        }else{
            //download
            Datacratic::S3Api s3(id, key);
            s3.downloadToFile(
                    "s3://" + bucket + "/" + s3File, 
                    localFile, 
                    maxSizeKB > 0 ? 1024 * maxSizeKB : -1);
        }
    }catch(const fs::filesystem_error& ex){
        cout << "File does not exist.\n";
        return 1;
    }
    return 0;
}

