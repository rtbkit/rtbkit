/* file_output.h                                                   -*- C++ -*-
   Jeremy Barnes, 29 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Get the output from the given file.
*/

#ifndef __logger__file_output_h__
#define __logger__file_output_h__

#include "logger.h"
#include "rotating_output.h"
#include "compressing_output.h"
#include "soa/service/s3.h"


namespace Datacratic {


/*****************************************************************************/
/* FILE OUTPUT                                                               */
/*****************************************************************************/

/** Class that logs messages into a file, optionally gzipped or otherwise
    compressed.

    The file writing is provided in another thread to avoid the entire
    logger getting bogged down if a disk can't keep up.
*/

struct NamedOutput : public CompressingOutput {

    NamedOutput(size_t ringBufferSize = 65536);

    virtual ~NamedOutput();

    virtual void open(const std::string & filename,
                      const std::string & compression = "",
                      int level = -1);

    /** Rotate the log file, finishing off writing the old one and
        then moving everything into the new one.

        This is an asynchronous operation.  It will return straight
        away.
    */
    void rotate(const std::string & newFilename,
                const std::string & newCompression = "",
                int newLevel = -1);

    virtual void close();

    boost::function<void (std::string)> onPreFileOpen;
    boost::function<void (std::string)> onPostFileOpen;
    boost::function<void (std::string)> onPreFileClose;
    boost::function<void (std::string)> onPostFileClose;

protected:

    virtual std::shared_ptr<Sink>
    createSink(const std::string & filename, bool append) = 0;

    void switchFile(const std::string & filename,
                    const std::string & compression,
                    int level);

    void closeFile();
};


/*****************************************************************************/
/* FILE SINK                                                                 */
/*****************************************************************************/

/** Class that writes to a file. */

struct FileSink : public CompressingOutput::Sink {
    FileSink(const std::string & filename = "",
             bool append = true,
             bool disambiguate = true);

    virtual ~FileSink();

    void open(const std::string & filename,
              bool append,
              bool disambiguate);

    virtual void close();
        
    virtual size_t write(const char * data, size_t size);

    virtual size_t flush(FileFlushLevel flushLevel);

    /// File descriptor we're writing to
    int fd;
};


/*****************************************************************************/
/* FILE OUTPUT                                                               */
/*****************************************************************************/

/** Class that logs messages into a file, optionally gzipped or otherwise
    compressed.

    The file writing is provided in another thread to avoid the entire
    logger getting bogged down if a disk can't keep up.
*/

struct FileOutput : public NamedOutput {

    FileOutput(const std::string & filename = "",
               size_t ringBufferSize = 65536);

    virtual ~FileOutput();

    virtual std::shared_ptr<Sink>
    createSink(const std::string & filename, bool append);
};


/*****************************************************************************/
/* ROTATING FILE OUTPUT                                                      */
/*****************************************************************************/

/** Logger that rotates files. */

struct RotatingFileOutput : public RotatingOutputAdaptor {

    RotatingFileOutput();

    virtual ~RotatingFileOutput();
    
    /** Open the file for rotation. */
    void open(const std::string & filenamePattern,
              const std::string & periodPattern,
              const std::string & compression = "",
              int level = -1);
    
private:
    FileOutput * createFile(const std::string & filename);

    std::string compression;
    int level;
};

} // namespace Datacratic


#endif /* __logger__file_output_h__ */
