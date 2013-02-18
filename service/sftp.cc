/* sftp.cc
   Jeremy Barnes, 21 June 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   sftp connection.
*/

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

} // namespace Datacratic
