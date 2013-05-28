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


/*****************************************************************************/
/* CLOUD SINK                                                                 */
/*****************************************************************************/

/** Class that writes to a cloud. */

struct CloudSink : public CompressingOutput::Sink {
    CloudSink(const std::string & uri ,
              bool append, bool disambiguate, std::string backupDir,
              std::string bucket, std::string accessKeyId, std::string accessKey
        );

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
                size_t ringBufferSize = 65536);

    virtual ~CloudOutput();

    virtual std::shared_ptr<Sink>
    createSink(const std::string & uri, bool append);

    std::vector<boost::filesystem::path> getFilesToUpload() ;
    void uploadLocalFiles() ;

    std::string backupDir_;
    std::string bucket_;
    std::string accessKeyId_;
    std::string accessKey_;

};


/*****************************************************************************/
/* ROTATING CLOUD OUTPUT                                                      */
/*****************************************************************************/

/** Logger that rotates clouds. */

struct RotatingCloudOutput : public RotatingOutputAdaptor {

    RotatingCloudOutput(std::string backupDir, std::string bucket, 
                        std::string accessKeyId, std::string accessKey);

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
};

} // namespace Datacratic


