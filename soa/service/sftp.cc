/* sftp.cc
   Jeremy Barnes, 21 June 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   sftp connection.
*/

#include <mutex>
#include <boost/iostreams/stream_buffer.hpp>
#include "soa/service/sftp.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include "soa/types/date.h"
#include <fstream>
#include "jml/utils/filter_streams.h"
#include "jml/utils/exc_assert.h"
#include <thread>
#include <unordered_map>


using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* SOCKET CONNECTION                                                         */
/*****************************************************************************/

SocketConnection::
SocketConnection()
    : sock(-1)
{
}

SocketConnection::
~SocketConnection()
{
    close();
}

void
SocketConnection::
connect(const std::string & hostname,
        const std::string & port)
{
    struct addrinfo hints;
    struct addrinfo *result, *rp;

    /* Obtain address(es) matching host/port */
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;     /* Allow IPv4 or IPv6 */
    hints.ai_socktype = SOCK_STREAM; /* Datagram socket */
    hints.ai_flags = AI_CANONNAME;
    hints.ai_protocol = 0;           /* Any protocol */

    int res = getaddrinfo(hostname.c_str(), port.c_str(), &hints, &result);
    if (res != 0)
        throw ML::Exception("getaddrinfo: %s", gai_strerror(res));

    cerr << "res = " << res << endl;
    cerr << "result = " << result << endl;

    /* getaddrinfo() returns a list of address structures.
       Try each address until we successfully connect(2).
       If socket(2) (or connect(2)) fails, we (close the socket
       and) try the next address. */

    for (rp = result; rp; rp = rp->ai_next) {
        if (rp->ai_canonname)
            cerr << "trying " << rp->ai_canonname << endl;
        else cerr << "trying null" << endl;

        sock = socket(rp->ai_family, rp->ai_socktype,
                      rp->ai_protocol);
        if (sock == -1) {
            cerr << "couldn't create connection socket: "
                 << strerror(errno) << endl;
            continue;
        }
            
        if (::connect(sock, rp->ai_addr, rp->ai_addrlen) != -1) {
            cerr << "connected" << endl;
            break;                  /* Success */
        }            

        cerr << "couldn't connect: " << strerror(errno) << endl;

        ::close(sock);
    }
        
    if (!rp)
        throw ML::Exception("couldn't connect anywhere");
        
    freeaddrinfo(result);           /* No longer needed */
}

void
SocketConnection::
close()
{
    ::close(sock);
}


/*****************************************************************************/
/* SSH CONNECTION                                                            */
/*****************************************************************************/

SshConnection::
SshConnection()
    : session(0)
{
}

SshConnection::
~SshConnection()
{
    close();
}

void
SshConnection::
connect(const std::string & hostname,
        const std::string & port)
{
    SocketConnection::connect(hostname, port);

    /* Create a session instance
     */ 
    session = libssh2_session_init();

    if(!session)
        throw ML::Exception("couldn't get libssh2 session");
 
    /* ... start it up. This will trade welcome banners, exchange keys,
     * and setup crypto, compression, and MAC layers
     */ 
    int rc = libssh2_session_handshake(session, sock);

    if(rc) {
        throw ML::Exception("error establishing session");
    }
 
    /* At this point we havn't yet authenticated.  The first thing to do
     * is check the hostkey's fingerprint against our known hosts Your app
     * may have it hard coded, may go to a file, may present it to the
     * user, that's your call
     */ 
    const char * fingerprint
        = libssh2_hostkey_hash(session, LIBSSH2_HOSTKEY_HASH_SHA1);

    printf("Fingerprint: ");
    for(int i = 0; i < 20; i++) {
        printf("%02X ", (unsigned char)fingerprint[i]);
    }
    printf("\n");
}

void
SshConnection::
passwordAuth(const std::string & username,
                  const std::string & password)
{
    /* We could authenticate via password */ 
    if (libssh2_userauth_password(session,
                                  username.c_str(),
                                  password.c_str())) {

        throw ML::Exception("password authentication failed: " + lastError());
    }
}

void
SshConnection::
publicKeyAuth(const std::string & username,
              const std::string & publicKeyFile,
              const std::string & privateKeyFile)
{
/* Or by public key */ 
    if (libssh2_userauth_publickey_fromfile(session, username.c_str(),
                                            publicKeyFile.c_str(),
                                            privateKeyFile.c_str(),
                                            "")) {
        throw ML::Exception("public key authentication failed: " + lastError());
    }
}
 
void
SshConnection::
setBlocking()
{
    /* Since we have not set non-blocking, tell libssh2 we are blocking */ 
    libssh2_session_set_blocking(session, 1);
}

std::string
SshConnection::
lastError() const
{
    char * errmsg = 0;
    int res = libssh2_session_last_error(session, &errmsg, 0, 0);
    if (res)
        cerr << "error getting error: " << res << endl;
    return errmsg;
}

void
SshConnection::
close()
{
    if (session) {
        libssh2_session_disconnect(session, "Normal Shutdown");
        libssh2_session_free(session);
    }
    session = 0;

    SocketConnection::close();
}


/*****************************************************************************/
/* ATTRIBUTES                                                                */
/*****************************************************************************/



/*****************************************************************************/
/* DIRECTORY                                                                 */
/*****************************************************************************/

SftpConnection::Directory::
Directory(const std::string & path,
          LIBSSH2_SFTP_HANDLE * handle,
          SftpConnection * owner)
    : path(path), handle(handle), owner(owner)
{
}

SftpConnection::Directory::
~Directory()
{
    libssh2_sftp_close(handle);
}

void
SftpConnection::Directory::
ls() const
{
    do {
        char mem[512];
        char longentry[512];
        LIBSSH2_SFTP_ATTRIBUTES attrs;
 
        /* loop until we fail */ 
        int rc = libssh2_sftp_readdir_ex(handle, mem, sizeof(mem),

                                         longentry, sizeof(longentry),
                                         &attrs);
        if(rc > 0) {
            /* rc is the length of the file name in the mem
               buffer */ 
 
            if (longentry[0] != '\0') {
                printf("%s\n", longentry);
            } else {
                if(attrs.flags & LIBSSH2_SFTP_ATTR_PERMISSIONS) {
                    /* this should check what permissions it
                       is and print the output accordingly */ 
                    printf("--fix----- ");
                }
                else {
                    printf("---------- ");
                }
 
                if(attrs.flags & LIBSSH2_SFTP_ATTR_UIDGID) {
                    printf("%4ld %4ld ", attrs.uid, attrs.gid);
                }
                else {
                    printf("   -    - ");
                }
 
                if(attrs.flags & LIBSSH2_SFTP_ATTR_SIZE) {
                    printf("%8lld ", (unsigned long long)attrs.filesize);
                }
                    
                printf("%s\n", mem);
            }
        }
        else
            break;
 
    } while (1);
}

void
SftpConnection::Directory::
forEachFile(const OnFile & onFile) const
{
    do {
        char mem[512];
        char longentry[512];
        Attributes attrs;
 
        /* loop until we fail */ 
        int rc = libssh2_sftp_readdir_ex(handle,
                                         mem, sizeof(mem),
                                         longentry, sizeof(longentry),
                                         &attrs);

        if(rc > 0) {
            /* rc is the length of the file name in the mem
               buffer */ 
            string filename(mem, mem + rc);
            onFile(filename, attrs);
        }
        else
            break;
 
    } while (1);
}


/*****************************************************************************/
/* FILE                                                                      */
/*****************************************************************************/

SftpConnection::File::
File(const std::string & path,
     LIBSSH2_SFTP_HANDLE * handle,
     SftpConnection * owner)
    : path(path), handle(handle), owner(owner)
{
}

SftpConnection::File::
~File()
{
    libssh2_sftp_close(handle);
}

SftpConnection::Attributes
SftpConnection::File::
getAttr() const
{
    Attributes result;
    int res = libssh2_sftp_fstat_ex(handle, &result, 0);
    if (res == -1)
        throw ML::Exception("getAttr(): " + owner->lastError());
    return result;
}

uint64_t
SftpConnection::File::
size() const
{
    return getAttr().filesize;
}

void
SftpConnection::File::
downloadTo(const std::string & filename) const
{
    uint64_t bytesToRead = size();

    uint64_t done = 0;
    std::ofstream stream(filename.c_str());

    size_t bufSize = 1024 * 1024;

    char * buf = new char[bufSize];
            
    Date start = Date::now();

    for (;;) {
        ssize_t numRead = libssh2_sftp_read(handle, buf, bufSize);
        //cerr << "read " << numRead << " bytes" << endl;
        if (numRead < 0) {
            throw ML::Exception("read(): " + owner->lastError());
        }
        if (numRead == 0) break;

        stream.write(buf, numRead);
        uint64_t doneBefore = done;
        done += numRead;

        if (doneBefore / 10000000 != done / 10000000) {
            double elapsed = Date::now().secondsSince(start);
            double rate = done / elapsed;
            cerr << "done " << done << " of "
                 << bytesToRead << " at "
                 << rate / 1024.0
                 << "k/sec window " << numRead
                 << " time left "
                 << (bytesToRead - done) / rate
                 << "s" << endl;
        }
    }

    delete[] buf;
}

/*****************************************************************************/
/* SFTP CONNECTION                                                           */
/*****************************************************************************/

SftpConnection::
SftpConnection()
    : sftp_session(0)
{
}

SftpConnection::
~SftpConnection()
{
    close();
}

void
SftpConnection::
connectPasswordAuth(const std::string & hostname,
                    const std::string & username,
                    const std::string & password,
                    const std::string & port)
{
    SshConnection::connect(hostname, port);
    SshConnection::passwordAuth(username, password);

    sftp_session = libssh2_sftp_init(session);
 
    if (!sftp_session) {
        throw ML::Exception("can't initialize SFTP session: "
                            + lastError());
    }

}

void
SftpConnection::
connectPublicKeyAuth(const std::string & hostname,
                              const std::string & username,
                              const std::string & publicKeyFile,
                              const std::string & privateKeyFile,
                              const std::string & port)
{
    SshConnection::connect(hostname, port);
    SshConnection::publicKeyAuth(username, publicKeyFile, privateKeyFile);

    sftp_session = libssh2_sftp_init(session);
 
    if (!sftp_session) {
        throw ML::Exception("can't initialize SFTP session: "
                            + lastError());
    }

}

SftpConnection::Directory
SftpConnection::
getDirectory(const std::string & path)
{
    LIBSSH2_SFTP_HANDLE * handle
        = libssh2_sftp_opendir(sftp_session, path.c_str());
        
    if (!handle) {
        throw ML::Exception("couldn't open path: " + lastError());
    }

    return Directory(path, handle, this);
}

SftpConnection::File
SftpConnection::
openFile(const std::string & path)
{
    LIBSSH2_SFTP_HANDLE * handle
        = libssh2_sftp_open_ex(sftp_session, path.c_str(),
                               path.length(), LIBSSH2_FXF_READ, 0,
                               LIBSSH2_SFTP_OPENFILE);
        
    if (!handle) {
        throw ML::Exception("couldn't open path: " + lastError());
    }

    return File(path, handle, this);
}

bool
SftpConnection::
getAttributes(const std::string & path, Attributes & attrs)
    const
{
    int res = libssh2_sftp_stat_ex(sftp_session,
                                   path.c_str(), path.length(), LIBSSH2_SFTP_STAT,
                                   &attrs);
    return (res != -1);
}
    
void
SftpConnection::
close()
{
    if (sftp_session) {
        libssh2_sftp_shutdown(sftp_session);
        sftp_session = 0;
    }

    SshConnection::close();
}

void
SftpConnection::
uploadFile(const char * start,
           size_t size,
           const std::string & path)
{
    /* Request a file via SFTP */ 
    LIBSSH2_SFTP_HANDLE * handle =
        libssh2_sftp_open(sftp_session, path.c_str(),
                          LIBSSH2_FXF_WRITE|LIBSSH2_FXF_CREAT|LIBSSH2_FXF_TRUNC,
                          LIBSSH2_SFTP_S_IRUSR|LIBSSH2_SFTP_S_IWUSR|
                          LIBSSH2_SFTP_S_IRGRP|LIBSSH2_SFTP_S_IROTH);
    
    if (!handle) {
        throw ML::Exception("couldn't open path: " + lastError());
    }

    Date started = Date::now();

    uint64_t offset = 0;
    uint64_t lastPrint = 0;
    Date lastTime = started;

    for (; offset < size; ) {
        /* write data in a loop until we block */ 
        size_t toSend = std::min<size_t>(size - offset,
                                         1024 * 1024);

        ssize_t rc = libssh2_sftp_write(handle,
                                        start + offset,
                                        toSend);
        
        if (rc == -1)
            throw ML::Exception("couldn't upload file: " + lastError());

        offset += rc;
        
        if (offset > lastPrint + 5 * 1024 * 1024 || offset == size) {
            Date now = Date::now();

            double mb = 1024 * 1024;

            double doneMb = offset / mb;
            double totalMb = size / mb;
            double elapsedOverall = now.secondsSince(started);
            double mbSecOverall = doneMb / elapsedOverall;
            double elapsedSince = now.secondsSince(lastTime);
            double mbSecInst = (offset - lastPrint) / mb / elapsedSince;

            cerr << ML::format("done %.2fMB of %.2fMB (%.2f%%) at %.2fMB/sec inst and %.2fMB/sec overall",
                               doneMb, totalMb,
                               100.0 * doneMb / totalMb,
                               mbSecInst,
                               mbSecOverall)
                 << endl;
                               

            lastPrint = offset;
            lastTime = now;
        }
        //cerr << "at " << offset / 1024.0 / 1024.0
        //     << " of " << size << endl;
    }
 
    libssh2_sftp_close(handle);
}


struct SftpStreamingDownloadSource {

    SftpStreamingDownloadSource(SftpConnection * owner,
                                std::string path)
    {
        impl.reset(new Impl());
        impl->owner = owner;
        impl->path = path;
        impl->start();
    }

    typedef char char_type;
    struct category
        : public boost::iostreams::input /*_seekable*/,
          public boost::iostreams::device_tag,
          public boost::iostreams::closable_tag
    { };
    
    struct Impl {
        Impl()
            : owner(0), offset(0), handle(0)
        {
        }

        ~Impl()
        {
            stop();
        }

        const SftpConnection * owner;
        std::string path;
        size_t offset;
        LIBSSH2_SFTP_HANDLE * handle;

        Date startDate;

        void start()
        {
            handle
                = libssh2_sftp_open_ex(owner->sftp_session, path.c_str(),
                                       path.length(), LIBSSH2_FXF_READ, 0,
                                       LIBSSH2_SFTP_OPENFILE);
            
            if (!handle) {
                throw ML::Exception("couldn't open path: "
                                    + owner->lastError());
            }
        }

        void stop()
        {
            if (handle) libssh2_sftp_close(handle);
        }

        std::streamsize read(char_type* s, std::streamsize n)
        {
            BOOST_STATIC_ASSERT(sizeof(char_type) == 1);

            ssize_t numRead = libssh2_sftp_read(handle, s, n);
            if (numRead < 0) {
                throw ML::Exception("read(): " + owner->lastError());
            }
            
            return numRead;
        }
    };

    std::shared_ptr<Impl> impl;

    std::streamsize read(char_type* s, std::streamsize n)
    {
        return impl->read(s, n);
    }

#if 0
    void seek(std::streamsize where, std::ios_base::seekdir dir)
    {
    }
#endif

    bool is_open() const
    {
        return !!impl;
    }

    void close()
    {
        impl.reset();
    }
};



struct SftpStreamingUploadSource {

    SftpStreamingUploadSource(SftpConnection * owner,
                              const std::string & path,
                              const ML::OnUriHandlerException & excCallback)
    {
        impl.reset(new Impl());
        impl->owner = owner;
        impl->path = path;
        impl->onException = excCallback;
        impl->start();
    }

    typedef char char_type;
    struct category
        : public boost::iostreams::output,
          public boost::iostreams::device_tag,
          public boost::iostreams::closable_tag
    {
    };

    struct Impl {
        Impl()
            : owner(0), handle(0), offset(0), lastPrint(0)
        {
        }

        ~Impl()
        {
            stop();
        }

        SftpConnection * owner;
        LIBSSH2_SFTP_HANDLE * handle;
        std::string path;
        ML::OnUriHandlerException onException;
        
        size_t offset;
        size_t lastPrint;
        Date lastTime;

        Date startDate;

        void start()
        {
            /* Request a file via SFTP */ 
            handle =
                libssh2_sftp_open(owner->sftp_session, path.c_str(),
                                  LIBSSH2_FXF_WRITE|LIBSSH2_FXF_CREAT|LIBSSH2_FXF_TRUNC,
                                  LIBSSH2_SFTP_S_IRUSR|LIBSSH2_SFTP_S_IWUSR|
                                  LIBSSH2_SFTP_S_IRGRP|LIBSSH2_SFTP_S_IROTH);
            
            if (!handle) {
                onException();
                throw ML::Exception("couldn't open path: " + owner->lastError());
            }

            startDate = Date::now();
        }
        
        void stop()
        {
            if (handle) libssh2_sftp_close(handle);
        }

        std::streamsize write(const char_type* s, std::streamsize n)
        {
            ssize_t done = 0;

            while (done < n) {

                ssize_t rc = libssh2_sftp_write(handle, s + done, n - done);
            
                if (rc == -1) {
                    onException();
                    throw ML::Exception("couldn't upload file: " + owner->lastError());
                }
            
                offset += rc;
                done += rc;

                if (offset > lastPrint + 5 * 1024 * 1024) {
                    Date now = Date::now();
                
                    double mb = 1024 * 1024;
                
                    double doneMb = offset / mb;
                    double elapsedOverall = now.secondsSince(startDate);
                    double mbSecOverall = doneMb / elapsedOverall;
                    double elapsedSince = now.secondsSince(lastTime);
                    double mbSecInst = (offset - lastPrint) / mb / elapsedSince;
                
                    cerr << ML::format("done %.2fMB at %.2fMB/sec inst and %.2fMB/sec overall",
                                       doneMb, 
                                       mbSecInst,
                                       mbSecOverall)
                         << endl;
                
                
                    lastPrint = offset;
                    lastTime = now;
                }
            }

            return done;
        }

        void flush()
        {
        }

        void finish()
        {
            stop();

            double elapsed = Date::now().secondsSince(startDate);

            cerr << "uploaded " << offset / 1024.0 / 1024.0
                 << "MB in " << elapsed << "s at "
                 << offset / 1024.0 / 1024.0 / elapsed
                 << "MB/s" << endl;
        }
    };

    std::shared_ptr<Impl> impl;

    std::streamsize write(const char_type* s, std::streamsize n)
    {
        return impl->write(s, n);
    }

    bool is_open() const
    {
        return !!impl;
    }

    void close()
    {
        impl->finish();
        impl.reset();
    }
};

ML::filter_ostream
SftpConnection::
streamingUpload(const std::string & path)
{
    ML::filter_ostream result;
    auto onException = [&] { result.notifyException(); };
    auto sb = streamingUploadStreambuf(path, onException);
    result.openFromStreambuf(sb.release(), true, path);
    
    return result;
}

std::unique_ptr<std::streambuf>
SftpConnection::
streamingUploadStreambuf(const std::string & path,
                         const ML::OnUriHandlerException & onException)
{
    std::unique_ptr<std::streambuf> result;
    result.reset(new boost::iostreams::stream_buffer<SftpStreamingUploadSource>
                 (SftpStreamingUploadSource(this, path, onException),
                  131072));
    return result;
}

ML::filter_istream
SftpConnection::
streamingDownload(const std::string & path)
{
    ML::filter_istream result;
    auto sb = streamingDownloadStreambuf(path);
    result.openFromStreambuf(sb.release(), true, path);
    
    return result;
}

std::unique_ptr<std::streambuf>
SftpConnection::
streamingDownloadStreambuf(const std::string & path)
{
    std::unique_ptr<std::streambuf> result;
    result.reset(new boost::iostreams::stream_buffer<SftpStreamingDownloadSource>
                 (SftpStreamingDownloadSource(this, path),
                  131072));
    return result;
}

int
SftpConnection::
unlink(const string & path){
    return libssh2_sftp_unlink(sftp_session, path.c_str());
}

namespace {

struct SftpHostInfo {
    std::string sftpHost;
    std::shared_ptr<SftpConnection> connection;  //< Used to access this uri
};

std::mutex sftpHostsLock;
std::unordered_map<std::string, SftpHostInfo> sftpHosts;

} // file scope

/** Sftp support for filter_ostream opens.  Register the host name here, and
    you can open it directly from sftp.
*/

void registerSftpHostPassword(const std::string & hostname,
                              const std::string & username,
                              const std::string & password,
                              const std::string & port)
{
    std::unique_lock<std::mutex> guard(sftpHostsLock);
    if (sftpHosts.count(hostname)){
        throw HostAlreadyRegistered(hostname);
    }

    SftpHostInfo info;
    info.sftpHost = hostname;
    info.connection = std::make_shared<SftpConnection>();
    info.connection->connectPasswordAuth(hostname, username, password, port);
    
    sftpHosts[hostname] = info;
}

void registerSftpHostPublicKey(const std::string & hostname,
                               const std::string & username,
                               const std::string & publicKeyFile,
                               const std::string & privateKeyFile,
                               const std::string & port)
{
    std::unique_lock<std::mutex> guard(sftpHostsLock);
    if (sftpHosts.count(hostname)){
        throw HostAlreadyRegistered(hostname);
    }

    SftpHostInfo info;
    info.sftpHost = hostname;
    info.connection = std::make_shared<SftpConnection>();
    info.connection->connectPublicKeyAuth(hostname, username,
                                          publicKeyFile,
                                          privateKeyFile,
                                          port);
    sftpHosts[hostname] = info;
}

struct RegisterSftpHandler {
    static std::pair<std::streambuf *, bool>
    getSftpHandler(const std::string & scheme,
                   const std::string & resource,
                   std::ios_base::open_mode mode,
                   const std::map<std::string, std::string> & options,
                   const ML::OnUriHandlerException & onException)
    {
        string::size_type pos = resource.find('/');
        if (pos == string::npos)
            throw ML::Exception("unable to find sftp host name in resource "
                                + resource);
        string host(resource, 0, pos);

        std::shared_ptr<SftpConnection> connection;

        {
            std::unique_lock<std::mutex> guard(sftpHostsLock);
            auto it = sftpHosts.find(host);
            if (it == sftpHosts.end())
                throw ML::Exception("unregistered sftp host " + host);
            connection = it->second.connection;
        }

        ExcAssert(connection);

        if (mode == ios::in) {
            return make_pair(connection->streamingDownloadStreambuf("sftp://" + resource)
                             .release(),
                             true);
        }
        else if (mode == ios::out) {
            return make_pair(connection->streamingUploadStreambuf("sftp://"
                                                                  + resource,
                                                                  onException)
                             .release(),
                             true);
        }
        else throw ML::Exception("no way to create sftp handler for non in/out");
    }

    RegisterSftpHandler()
    {
        ML::registerUriHandler("sftp", getSftpHandler);
    }

} registerSftpHandler;

std::shared_ptr<SftpConnection> getSftpConnectionForHost(const std::string & hostname)
{
    std::unique_lock<std::mutex> guard(sftpHostsLock);
    auto it = sftpHosts.find(hostname);
    if (it == sftpHosts.end())
        throw ML::Exception("unregistered sftp host " + hostname);
    return it->second.connection;
}



} // namespace Datacratic
