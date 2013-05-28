/* file_output.cc
   Jeremy Barnes, 29 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "file_output.h"
#include "jml/utils/parse_context.h"
#include <boost/tuple/tuple.hpp>

#define BOOST_SYSTEM_NO_DEPRECATED

#include <boost/filesystem.hpp>


using namespace std;
using namespace boost::filesystem;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* FILE OUTPUT                                                               */
/*****************************************************************************/

NamedOutput::
NamedOutput(size_t ringBufferSize)
    : CompressingOutput(ringBufferSize)
{
}

NamedOutput::
~NamedOutput()
{
    close();
}

void
NamedOutput::
open(const std::string & filename,
     const std::string & compression,
     int level)
{
    close();
    switchFile(filename, compression, level);
    startWorkerThread();
}

void
NamedOutput::
closeFile()
{
    if (!sink) return;

    //if (onPreFileClose)
    //    onPreFileClose(currentFilename);

    closeCompressor();
    sink->close();

    //if (onPostFileClose)
    //    onPostFileClose(currentFilename);
}

void
NamedOutput::
rotate(const std::string & newFilename,
       const std::string & newCompression,
       int newLevel)
{
    auto op = [=] ()
        {
            this->switchFile(newFilename, newCompression, newLevel);
        };

    pushOperation(op);
}

void
NamedOutput::
close()
{
    WorkerThreadOutput::stopWorkerThread();
    closeFile();
}

void
NamedOutput::
switchFile(const std::string & filename,
           const std::string & compression_,
           int level)
{

    closeCompressor();
    closeFile();

    string compression = compression_;
    if (compression == "")
        compression = Compressor::filenameToCompression(filename);

    string fn = filename;

    bool append = false;
    if (filename.length() > 0 && filename[filename.length() - 1] == '+') {
        fn = string(filename, 0, filename.length() - 2);
        append = true;
    }

    if (onPreFileOpen)
        onPreFileOpen(fn);

    // todo - fix this by saving the output of createSink and getting 
    // the new uri if there is one
    std::shared_ptr<CompressingOutput::Sink> theSink = createSink(fn, append);
    CompressingOutput::open(theSink, compression, level);

    if (onPostFileOpen)
    {
//        cerr << "NamedOutput::switchfile: calling on postfile open with " << 
//            theSink->currentUri << endl;
        onPostFileOpen(theSink->currentUri);
    }
}


/*****************************************************************************/
/* FILE SINK                                                                 */
/*****************************************************************************/

FileSink::
FileSink(const std::string & filename, bool append, bool disambiguate)
    : fd(-1)
{
    if (filename != "")
        open(filename, append, disambiguate);
}

FileSink::
~FileSink()
{
    close();
}

void
FileSink::
open(const std::string & filename, bool append, bool disambiguate)
{
    close();

    string fn = filename;

    path p = filename;
    path d = p;  d.remove_filename();

    string e = p.extension().string();
    path s = p.stem();
    while (s.extension() != "") {
        e = s.extension().string() + e;
        s = s.stem();
    }

    //cerr << "dir " << d << " ext " << e << " stem " << s << endl;

    if (!exists(d)) {
        //cerr << "creating directory " << d << endl;
        create_directories(d);
    }

    if (disambiguate) {
        string base = (d/s).string();
        string disamb = "";
        int n = 0;

        while (exists((fn = base + disamb + e))) {
            cerr << "file " + (base + disamb + e) + " exists; disambiguating"
                 << endl;
            disamb = ML::format(".%d", ++n);
        }

    }    

    cerr << "opening file " << fn << " with append " << append << endl;

    fd = ::open(fn.c_str(),
                O_WRONLY | O_CREAT | O_EXCL | (append ? O_APPEND : 0),
                00664);
    
    if (fd == -1)
        throw ML::Exception(errno, "open of " + filename);

    currentUri = fn;

}

void
FileSink::
close()
{
    //cerr << "closing file " << currentUri << endl;

    if (fd != -1) {
        int r = fdatasync(fd);
        if (r == -1)
            throw ML::Exception(errno, "fdatasync " + currentUri);

        int res = ::close(fd);
        if (res == -1)
            throw ML::Exception(errno, "close " + currentUri);
        fd = -1;

        currentUri = "";
    }
}

size_t
FileSink::
write(const char * data, size_t size)
{
    size_t done = 0;
    
    while (done < size) {
        ssize_t res = ::write(fd, data + done, size - done);
        if (res == -1)
            throw ML::Exception(errno, "write to FileSink for "
                                + currentUri);
        done += res;
    }

    return done;
}

size_t
FileSink::
flush(FileFlushLevel flushLevel)
{
    switch (flushLevel) {

    case FLUSH_NONE:
        return 0;

    case FLUSH_TO_OS:
        return 0;   // we call write() straight away, so ther is no internal
                    // buffering and nothing to do here

    case FLUSH_TO_DISK: {
        int r = fdatasync(fd);
        if (r == -1)
            throw ML::Exception(errno, "fdatasync for " + currentUri);
        return 0;
    }

    default:
        throw ML::Exception("FileSink::flush(): unknown flush level");
    }
}


/*****************************************************************************/
/* FILE OUTPUT                                                               */
/*****************************************************************************/

FileOutput::
FileOutput(const std::string & filename, size_t ringBufferSize)
    : NamedOutput(ringBufferSize)
{
    if (filename != "")
        open(filename);
}

FileOutput::
~FileOutput()
{
    close();
}

std::shared_ptr<CompressingOutput::Sink>
FileOutput::
createSink(const std::string & filename, bool append)
{
    return std::make_shared<FileSink>(filename, append);
}


/*****************************************************************************/
/* ROTATING FILE OUTPUT                                                      */
/*****************************************************************************/

RotatingFileOutput::
RotatingFileOutput()
    : RotatingOutputAdaptor(std::bind(&RotatingFileOutput::createFile,
                                      this,
                                      std::placeholders::_1))
{
}

RotatingFileOutput::
~RotatingFileOutput()
{
    close();
}
    
void
RotatingFileOutput::
open(const std::string & filenamePattern,
     const std::string & periodPattern,
     const std::string & compression,
     int level)
{
    this->compression = compression;
    this->level = level;

    RotatingOutputAdaptor::open(filenamePattern, periodPattern);
}

FileOutput *
RotatingFileOutput::
createFile(const std::string & filename)
{
    std::unique_ptr<FileOutput> result(new FileOutput());

    result->onPreFileOpen = [=] (const string & fn)
        { if (this->onPreFileOpen) this->onPreFileOpen(fn); };
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


} // namespace Datacratic
