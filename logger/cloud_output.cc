/* cloud_output.cc
   Jeremy Barnes, 18 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#include "cloud_output.h"
#include <memory>
#include <boost/filesystem.hpp>
#include "jml/utils/file_functions.h"

namespace Datacratic {
using namespace std;
using namespace ML;
namespace fs = boost::filesystem;

CloudSink::
CloudSink(const std::string & uri, bool append, bool disambiguate,
          std::string backupDir, std::string bucket, string accessKeyId, 
          string accessKey):
    backupDir_(backupDir),bucket_(bucket), 
    accessKeyId_(accessKeyId), accessKey_(accessKey)
{
    if (uri != "")
    {
        currentUri = uri;
        open(uri, append, disambiguate);
    }
}

CloudSink::
~CloudSink()
{
    close();
}

std::string
CloudSink::disambiguateUri(std::string uri) const
{
    Datacratic::S3Api s3(accessKeyId_, accessKey_);
    bool disamb = false;
    string result = uri;
    unsigned int n = 0;
    auto orig = S3Api::parseUri(result);
    while(!disamb)
    {
        //cerr << "Disambiguating s3 uri " << result << endl;
        auto bucketObject = S3Api::parseUri(result);
        // make sure it matches the bucket that it was created with
        if(bucketObject.first != bucket_)
            throw ML::Exception("trying to open a url %s which has a different bucket %s from the specified bucket",
                                bucketObject.first.c_str(), bucket_.c_str());

        S3Api::ObjectInfo info = s3.tryGetObjectInfo(bucket_, bucketObject.second);
        if(!info.exists)
        {
            cerr << "file " << result << " does not exist..checking for multipart upload" << endl;

            bool inProgress = s3.isMultiPartUploadInProgress(bucket_,"/" + bucketObject.second);
            if(!inProgress)
            {
                cerr << "no multipart upload in progress..disambiguation complete " << endl;
                disamb = true;
                break;
            }
            else
            {
                cerr << "multipart upload in progress..disambiguation required" << endl;
            }
        }
        // get a new uri
        fs::path resourcePath(orig.second);

        string ext = resourcePath.extension().string();
        fs::path theStem = resourcePath.stem();
        while (!theStem.extension().empty())
        {
            ext = theStem.extension().string()  + ext;
            theStem = theStem.stem();
        }

        string num = ML::format(".%d",++n); 
        fs::path disambName = resourcePath.parent_path().string() /
                              fs::path((theStem.string() + num + ext));
        result = "s3://" + orig.first + "/" + disambName.string();

    }
    //cerr << "disambiguated uri: " <<  result << endl;
    return result ;
}

void
CloudSink::
open(const std::string & uri, bool append, bool disambiguate)
{
    cloudStream.close();
    string disambUri(uri);
    if(disambiguate)
    {
        disambUri = disambiguateUri(uri);
    }

    currentUri = disambUri;

    cloudStream.open(disambUri, std::ios_base::out |
                          (append ? std::ios_base::app : std::ios::openmode()));

    // Get the file name from the s3 uri. We want to preserve the path since
    // if we only get the filename we could overwrite files with the same name
    // but in a different directory. uri format is s3://
    fs::path backupPath(backupDir_);
    fs::path filePath(backupPath / disambUri.substr(5));
    // Get the path and create the directories
    fs::create_directories(filePath.parent_path());
    // create the local file and directory
    fileStream.open(filePath.string(), std::ios_base::out |
            (append ? std::ios_base::app : std::ios::openmode()));
}

void
CloudSink::
close()
{
    cloudStream.close();
    fileStream.close();
    fs::path filePath(backupDir_ + currentUri.substr(5));
//    cerr << "Erasing local backup file " << filePath.string() << endl;
    fs::remove(filePath);
}

size_t
CloudSink::
write(const char * data, size_t size)
{
    fileStream.write(data, size);
    cloudStream.write(data, size);
    return size ;
}

size_t
CloudSink::
flush(FileFlushLevel flushLevel)
{
    return 0;
}

std::vector<fs::path>
CloudOutput::getFilesToUpload()
{
    vector<fs::path> allDirs;
    vector<fs::path> filesToUpload;
    allDirs.push_back(fs::path(backupDir_));
    while(!allDirs.empty())
    {
        fs::path curDir = allDirs.back();
        allDirs.pop_back();
        for (fs::directory_iterator it = fs::directory_iterator(curDir);
             it != fs::directory_iterator(); ++it)
        {
            if(fs::is_directory(*it))
            {
                allDirs.push_back(it->path());
            }
            else
            {
                filesToUpload.push_back(it->path());
            }
        }
    }
    return filesToUpload;
}
void
CloudOutput::uploadLocalFiles()
{
    fs::path backupPath(backupDir_);
    cerr << "Uploading local files " << endl;
    std::vector<boost::filesystem::path> filesToUpload = getFilesToUpload() ;
    auto doUpload = [=]()
    {
        Datacratic::S3Api s3(accessKeyId_, accessKey_);
        for( auto file : filesToUpload)
        {

            fs::path::iterator path1 = backupPath.begin();
            fs::path::iterator path2 = file.begin();
            // find out where the backupdir stops
            while( *path1 == *path2 && path2 != file.end() && path1 != backupPath.end())
            {
                ++path1;
                ++path2;
            }
            // if we reached the end there is nothing to do
            if(path2 == backupPath.end())
            {
                cerr << "Empty backup directory - no files to upload " << endl;
                continue;
            }
            else
            {
                string bucket((*path2).string());
                if(bucket != bucket_)
                {
                    cerr << "Found a bucket that does not belong to us...skipping"
                         << bucket << endl;
                    continue;
                }

                ++path2;
                fs::path s3Path("/");
                while(path2 != file.end())
                {
                    s3Path = s3Path/ *path2;
                    ++path2;
                }
                ML::File_Read_Buffer frb(file.string());

                string result = s3.upload(frb.start(), fs::file_size(file), 
                                          bucket_, s3Path.string());

                // Check that the upload succeeded
                std::string s3File = s3Path.string();
                // now remove the first slash
                s3File = s3File.substr(1, s3File.size());
           
                S3Api::ObjectInfo info = s3.tryGetObjectInfo(bucket_, s3File);
                if(info.exists && info.size == fs::file_size(file))
                {
                    cerr << "File " << file << " was successfully transferred. Deleting local copy..." << endl;
                    fs::remove(fs::path(file));
                }
                else
                {
                    cerr << "Failed to upload file " << file << " to s3..." << endl;
                }
            }
        }
    };
   std::thread uploadThread(doUpload);
   uploadThread.detach();
}

std::shared_ptr<CompressingOutput::Sink>
CloudOutput::createSink(const string & uri, bool append)
{
    return make_shared<CloudSink>(uri, append, true, backupDir_, bucket_, 
                                  accessKeyId_, accessKey_);
}

 RotatingCloudOutput::RotatingCloudOutput(std::string backupDir, 
                                          string bucket,
                                          string accessKeyId,
                                          string accessKey)
:RotatingOutputAdaptor(std::bind(&RotatingCloudOutput::createFile,
                                      this,
                                  std::placeholders::_1)),
        backupDir_(backupDir),bucket_(bucket), accessKeyId_(accessKeyId), 
        accessKey_(accessKey)
{
}


void
RotatingCloudOutput::
open(const std::string & filenamePattern,
     const std::string & periodPattern,
     const std::string & compression,
     int level)
{
    this->compression = compression;
    this->level = level;

    
    RotatingOutputAdaptor::open(filenamePattern, periodPattern);
}

RotatingCloudOutput::
~RotatingCloudOutput()
{
    close();
}

CloudOutput *
RotatingCloudOutput::
createFile(const string & filename)
{

    std::unique_ptr<CloudOutput> result(new CloudOutput(backupDir_, bucket_,
                                                      accessKeyId_,accessKey_));

    result->onPreFileOpen = [=] (const string & fn)
    {
        if (this->onPreFileOpen)
        {
            this->onPreFileOpen(fn);
        }
    };
    result->onPostFileOpen = [=] (const string & fn)
        { if (this->onPostFileOpen) this->onPostFileOpen(fn); };
    result->onPreFileClose = [=] (const string & fn)
        { if (this->onPreFileClose) this->onPreFileClose(fn); };
    result->onPostFileClose = [=] (const string & fn)
        { if (this->onPostFileClose) this->onPostFileClose(fn); };
    result->onFileWrite = [=] (const string& channel, const std::size_t bytes)
    { if (this->onFileWrite) this->onFileWrite(channel, bytes); };


    result->open(filename, compression, level);

    return result.release();
}


/*****************************************************************************/
/* CLOUD OUTPUT                                                              */
/*****************************************************************************/
namespace
{
    std::once_flag flag;
}
CloudOutput::
    CloudOutput(std::string backupDir, std::string bucket, 
                std::string accessKeyId, std::string accessKey, size_t ringBufferSize)
    : NamedOutput(ringBufferSize),backupDir_(backupDir),bucket_(bucket),
      accessKeyId_(accessKeyId),accessKey_(accessKey)
{

    if( !fs::exists(backupDir))
        fs::create_directory(backupDir);

    registerS3Bucket(bucket, accessKeyId, accessKey);
    // get the list of files we need to upload and pass it to another thread
    std::call_once(flag, [this]{uploadLocalFiles();}) ;
}

CloudOutput::
~CloudOutput()
{
}

} // namespace Datacratic
