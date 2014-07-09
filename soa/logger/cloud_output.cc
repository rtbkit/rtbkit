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
std::vector<boost::filesystem::path> CloudOutput::filesToUpload_;
    std::map<std::string,unsigned> CloudOutput::pendingDisamb_;

CloudSink::
CloudSink(const std::string & uri, bool append, bool disambiguate,
          std::string backupDir, std::string bucket, string accessKeyId, 
          string accessKey, unsigned int numThreads):
    backupDir_(backupDir),bucket_(bucket), 
    accessKeyId_(accessKeyId), accessKey_(accessKey),numThreads_(numThreads)
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
    CloudUtils::ObjectPart thePart = CloudUtils::parseObject(orig.second);
    // getting the local disamb number if there is one
    auto foundLocal = CloudOutput::pendingDisamb_.find(orig.second);
    if(foundLocal != CloudOutput::pendingDisamb_.end())
    {
        cerr << "! Found a local disamb of " << foundLocal->second << " for uri "
             << uri << endl;
        n = foundLocal->second;
        fs::path resourcePath(orig.second);
        string num = ML::format(".%d",++n);
        fs::path disambName = resourcePath.parent_path().string() /
                              fs::path((thePart.stem_ + num + thePart.ext_));
        result = "s3://" + orig.first + "/" + disambName.string();
        cerr << " We start looking on s3 for the uri " << result << endl;
    }
    while(!disamb)
    {
        cerr << "Disambiguating s3 uri " << result << endl;
        auto bucketObject = S3Api::parseUri(result);
        // make sure it matches the bucket that it was created with
        if(bucketObject.first != bucket_)
            throw ML::Exception("trying to open a url %s which has a different bucket %s from the specified bucket",
                                bucketObject.first.c_str(), bucket_.c_str());

        // Before going to see if it exists on s3 we check if we have it locally

        S3Api::ObjectInfo info = s3.tryGetObjectInfo(bucket_, bucketObject.second);
        if(!info.exists)
        {
            cerr << "file " << result << " does not exist..checking for multipart upload" << endl;
            bool inProgress;
            string uploadId;
            std::tie(inProgress,uploadId) = s3.isMultiPartUploadInProgress(bucket_,"/" + bucketObject.second);
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
    cerr << "disambiguated uri: " <<  result << endl;
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

    string compression = "";
    int level = -1;
    cloudStream.open(disambUri, std::ios_base::out |
                     (append ? std::ios_base::app : std::ios::openmode()),
                     compression, level, numThreads_);

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
    cerr << "Erasing local backup file " << filePath.string() << endl;
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

void
CloudOutput::getFilesToUpload()
{
    vector<fs::path> allDirs;
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
                filesToUpload_.push_back(it->path());
                string bucket,object;
                std::tie(bucket,object) = CloudUtils::parseCloudBackupFilePath(backupDir_,
                                                                               it->path().string());
                CloudUtils::ObjectPart thePart  = CloudUtils::parseObject(object);
                // now look for it in the map. If not there we insert it with a
                // count of 0
                auto found = pendingDisamb_.find(object);
                if( found == pendingDisamb_.end())
                {
                    pendingDisamb_[object] = thePart.disamb_;
                }
                else
                {
                    // Do not assume that the files are in order of disamb number
                    if(found->second < thePart.disamb_)
                        found->second = thePart.disamb_;
                }
            }
        }
    }
    // for each file we must upload print the disamb number
    for( auto file: filesToUpload_)
    {
        string bucket,object;
        std::tie(bucket,object) = CloudUtils::parseCloudBackupFilePath(backupDir_, file.string());
        auto found = pendingDisamb_.find(object);
        cerr << "Object: " << object << " disamb " << found->second << endl;
    }
    return ;
}

namespace CloudUtils
{
    std::pair<string,string>  parseCloudBackupFilePath(std::string backupDir,
                                                  boost::filesystem::path file)
    {
        std::pair<string,string> result;
        fs::path backupPath(backupDir);
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
            return result;
        }
        else
        {
            string bucket((*path2).string());
            result.first = bucket;
            ++path2;
            fs::path s3Path;
            while(path2 != file.end())
            {
                s3Path = s3Path/ *path2;
                ++path2;
            }
            result.second = s3Path.string();
        }

        return result;
    }

    ObjectPart parseObject(string object)
    {

        fs::path resourcePath(object);
        unsigned disamb = 0;
        string ext = resourcePath.extension().string();
        fs::path theStem = resourcePath.stem();
        ObjectPart thePart(theStem.string(), 0, ext);
        while (!theStem.extension().empty())
        {
            string curExt = theStem.extension().string();
            string curExtNoDot = theStem.extension().string().substr(1);
//            cerr << "cur ext no dot<" << curExtNoDot <<">" << endl;
            theStem = theStem.stem();
            thePart.stem_ = theStem.string();
            // check if the extension consists of only numbers
            if( std::all_of(curExtNoDot.begin(), curExtNoDot.end(),
                            [](char c){return isdigit(c);}))
            {
                disamb = stoul(curExtNoDot);
                thePart.disamb_ = disamb;
//                cerr << "ext is all digits " << endl;
                break;
            }
            ext = curExt + ext;
            thePart.ext_ = ext;
        }
        cerr << "Object: " << object << endl;
        cerr << "\t" << "Stem: " << thePart.stem_ << " disamb : " << thePart.disamb_
             << " extension: " << thePart.ext_ << endl;
        return thePart;
    }
};

void
CloudOutput::uploadLocalFiles()
{
    fs::path backupPath(backupDir_);

    getFilesToUpload();

    cerr << "Uploading local files " << endl;
    auto doUpload = [=]()
    {
        Datacratic::S3Api s3(accessKeyId_, accessKey_);
        for( auto file : filesToUpload_)
        {
            string bucket,object;
            std::tie(bucket,object) = CloudUtils::parseCloudBackupFilePath(backupDir_,file);
            if(bucket != bucket_)
            {
                cerr << "bucket " << bucket << " did not match " << bucket_ << endl;
                continue;
            }
            ML::File_Read_Buffer frb(file.string());

            string result = s3.upload(frb.start(), fs::file_size(file), 
                                      bucket_, "/" + object);

            // Check that the upload succeeded
            std::string s3File = object;
           
            S3Api::ObjectInfo info = s3.tryGetObjectInfo(bucket, s3File);
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
    };
   std::thread uploadThread(doUpload);
   uploadThread.detach();
}

std::shared_ptr<CompressingOutput::Sink>
CloudOutput::createSink(const string & uri, bool append)
{
    return make_shared<CloudSink>(uri, append, true, backupDir_, bucket_, 
                                  accessKeyId_, accessKey_, numThreads_);
}

RotatingCloudOutput::RotatingCloudOutput(std::string backupDir, 
                                         string bucket,
                                         string accessKeyId,
                                         string accessKey,
                                         unsigned int numThreads)
:RotatingOutputAdaptor(std::bind(&RotatingCloudOutput::createFile,
                                      this,
                                  std::placeholders::_1)),
        backupDir_(backupDir),bucket_(bucket), accessKeyId_(accessKeyId), 
        accessKey_(accessKey),numThreads_(numThreads)
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
                                                        accessKeyId_,accessKey_,
                                                        numThreads_));

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
                std::string accessKeyId, std::string accessKey, 
                unsigned int numThreads, size_t ringBufferSize)
    : NamedOutput(ringBufferSize),backupDir_(backupDir),bucket_(bucket),
      accessKeyId_(accessKeyId),accessKey_(accessKey),numThreads_(numThreads)
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
