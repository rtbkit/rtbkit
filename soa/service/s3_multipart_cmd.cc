/** s3_multipart_cmd.cc
    Sunil Rottoo, 29th May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.
    
    
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
    string outputFile;
    string localFile;
    string s3KeyId;
    string s3Key;
    bool cancel = false; //whether the multipart upload should be cancelled
    
    po::options_description desc("Main options");
    desc.add_options()
        ("output-uri,o", po::value<string>(&outputFile), "Output files/uris (can have multiple file/s3://bucket/object)")
        ("s3-key-id,I", po::value<string>(&s3KeyId), "S3 key id")
        ("s3-key,K", po::value<string>(&s3Key), "S3 key")
        ("cancel,c", po::value<bool>(&cancel), "whether or not to cancel the upload")
        ("help,h", "Produce help message");
    
    po::variables_map vm;

    po::store(po::command_line_parser(argc, argv)
          .options(desc)
          .run(),
          vm);
    notify(vm);

    cerr << "The output uri is " << outputFile << endl;
    cerr << "s3KeyId " << s3KeyId << endl;
    cerr<<  "s3Key " << s3Key << endl;
    Datacratic::S3Api s3(s3KeyId, s3Key);
    string bucket,object;
    std::tie(bucket,object) = S3Api::parseUri(outputFile);
    cerr << "Bucket : " << bucket << " object " << object << endl;
    if (s3KeyId != "")
        registerS3Bucket(bucket, s3KeyId, s3Key);

    // now see if there is a multipart upload in process
    bool inProgress;
    string uploadId;
    std::tie(inProgress,uploadId) = s3.isMultiPartUploadInProgress(bucket , "/" + object);
    cerr << outputFile << ":multipart upload in progress? " << inProgress << endl;
    if(cancel && inProgress)
    {
        cerr << "Cancelling multipart upload " << uploadId << endl;
        s3.erase(bucket, "/" + object, "uploadId=" + uploadId);
        std::tie(inProgress,uploadId) = s3.isMultiPartUploadInProgress(bucket , "/" + object);        
        cerr << outputFile << ":after cancel:multipart upload in progress? " << 
            inProgress << endl;
    }
    return 0;
}

