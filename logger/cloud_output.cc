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
          std::string backupDir):
    currentUri_(uri),backupDir_(backupDir)
{
    if (uri != "")
        open(uri, append, disambiguate);
}

CloudSink::
~CloudSink()
{
    //cerr << "CloudSink::~CloudSink::was called with uri " << currentUri_ << endl;
    close();
}

void
CloudSink::
open(const std::string & uri, bool append, bool disambiguate)
{

    cerr <<"CloudSink::open: with uri " <<uri << endl;
    cloudStream.close();
    cloudStream.open(uri, std::ios_base::out |
                          (append ? std::ios_base::app : std::ios::openmode()));

    // Get the file name from the s3 uri. We want to preserve the path since
    // if we only get the filename we could overwrite files with the same name
    // but in a different directory. uri format is s3://
    fs::path backupPath(backupDir_);
    fs::path filePath(backupPath / uri.substr(5));
    cerr << "The file path uri is " << filePath.string() << endl;
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
   fs::path filePath(backupDir_ + currentUri_.substr(5));
   cerr << "Erasing local file " << filePath.string() << endl;
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
        cerr <<"current directory = " << curDir.string() << endl;
        for (fs::directory_iterator it = fs::directory_iterator(curDir);
             it != fs::directory_iterator(); ++it)
        {
            if(fs::is_directory(*it))
            {
                cerr << "Found directory " <<  it->path().string() << endl;
                allDirs.push_back(it->path());
            }
            else
            {
                cerr << "found a file " << it->path().string() << endl;
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
           
                S3Api::ObjectInfo info = s3.getObjectInfo(bucket_, s3File);
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
//    cerr << "CloudOutput::createSink was called with uri " << uri << endl;
    return make_shared<CloudSink>(uri, append, true, backupDir_);
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
    //cerr << "RotatingCloudOutput::createFile. Entering..." << endl;
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
    cerr <<"opening cloud output with compression " << compression << endl;
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
