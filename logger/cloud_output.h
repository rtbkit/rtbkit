/* cloud_output.h                                                  -*- C++ -*-
   Jeremy Barnes, 18 September 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Output that puts files into... the CLOUD!
*/

#include "file_output.h"


#include "logger.h"
#include "jml/arch/timers.h"
#include "soa/types/date.h"
#include <boost/thread/recursive_mutex.hpp>
#include <boost/filesystem.hpp>
#include "soa/types/periodic_utils.h"
#include "compressor.h"


namespace Datacratic {

    namespace CloudUtils {
        // give the full path of a cloud back up file this function will
        // return the name of the bucket and the actual file name
        // eg. given
        //./logger_cloud_backup/tests.datacratic.com/logger/2013-05-29/router-2013-05-29-20:00:00.log.gz
        // returns a pair where the first element is the bucket and the second is the
        // resource
        std::pair<std::string,std::string>
                        parseCloudBackupFilePath(std::string backupDir, 
                                                 boost::filesystem::path file);

        // Given an object such as logger/2013-05-29/router-2013-05-29-20:00:00.2.log.gz
        // return the stem, the disamb number and the extension. for the example above
        // stem: router-2013-05-29-20:00:00
        // disamb:2
        // ext: .log.gz
        struct ObjectPart
        {
            ObjectPart(std::string stem,unsigned disamb,std::string ext)
                :stem_(stem),disamb_(disamb),ext_(ext)
                {
                }
            std::string stem_;
            unsigned disamb_;
            std::string ext_;
        };
        ObjectPart parseObject(std::string object);

    };

/*****************************************************************************/
/* CLOUD SINK                                                                 */
/*****************************************************************************/

/** Class that writes to a cloud. */

struct CloudSink : public CompressingOutput::Sink {
    CloudSink(const std::string & uri ,
              bool append, bool disambiguate, std::string backupDir,
              std::string bucket, std::string accessKeyId, std::string accessKey,
              unsigned int numThreads);

    virtual ~CloudSink();

    void open(const std::string & uri,
              bool append,
              bool disambiguate);

    std::string disambiguateUri(std::string uri) const;

    virtual void close();

    virtual size_t write(const char * data, size_t size);

    virtual size_t flush(FileFlushLevel flushLevel);

    std::string backupDir_;
    std::string bucket_;
    std::string accessKeyId_;
    std::string accessKey_;
    unsigned int numThreads_;
    /// Current stream to the cloud (TM)
    ML::filter_ostream cloudStream;
    // we write to a temporary file on local disk which we delete when
    // the corresponding cloud stream is closed.
    ML::filter_ostream fileStream;

    /// File descriptor we're writing to
    //int fd;
};


/*****************************************************************************/
/* CLOUD OUTPUT                                                              */
/*****************************************************************************/

/** Logger output that records its clouds into the cloud.

    This works in the following manner:
    1.  Data is compressed as a stream.
    2.  Once data comes out the other end of the compression, it is both
        streamed into the cloud and written to a temporary file on disk.
    3.  When it is time for a log rotation, the two clouds are closed and
        the stream to the cloud closed.  We then check that the cloud made
        it to the cloud; at which point we delete the log file.
    4.  When we restart, we look in the cache directory for any temporary
        files.  If any are found, we complete the upload to the cloud and
        then delete that file from disk.
*/

struct CloudOutput : public NamedOutput {

    CloudOutput(std::string backupDir, std::string bucket, 
                std::string accessKeyId, std::string accessKey,
                unsigned int numThreads,
                size_t ringBufferSize = 65536);

    virtual ~CloudOutput();

    virtual std::shared_ptr<Sink>
    createSink(const std::string & uri, bool append);

    void getFilesToUpload() ;
    void uploadLocalFiles() ;

    std::string backupDir_;
    std::string bucket_;
    std::string accessKeyId_;
    std::string accessKey_;
    unsigned numThreads_;
    // note that this structure is only filled in a function that is guaranteed
    // to be called once
    static std::vector<boost::filesystem::path> filesToUpload_;
    // for each object that we need to upload store the highest disambiguation
    // number seen
    static std::map<std::string,unsigned> pendingDisamb_;
};


/*****************************************************************************/
/* ROTATING CLOUD OUTPUT                                                      */
/*****************************************************************************/

/** Logger that rotates clouds. */

struct RotatingCloudOutput : public RotatingOutputAdaptor {

    RotatingCloudOutput(std::string backupDir, std::string bucket, 
                        std::string accessKeyId, std::string accessKey,
                        unsigned int numThreads);

    virtual ~RotatingCloudOutput();

    /* Open the cloud for rotation. */
    /* uriPattern: The URI of the s3 bucket
     * periodPattern: The frequency of rotation i.e "2s" means "rotate
     * every 2 seconds." The accepted symbols are
     * x: milliseconds
     * s: seconds
     * m: minutes
     * h: hours
     * d: days
     * w: weeks
     * M: months
     * y: years
     */
    void open(const std::string & uriPattern,
              const std::string & periodPattern,
              const std::string & compression = "",
              int level = -1);

private:
    CloudOutput * createFile(const std::string & filename);

    std::string compression;
    int level;
    std::string backupDir_;
    std::string bucket_;
    std::string accessKeyId_;
    std::string accessKey_;
    unsigned int numThreads_;//number of threads to use for s3 upload per file
};

} // namespace Datacratic


