/* sftp.h                                                          -*- C++ -*-
   Jeremy Barnes, 21 June 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Sftp functionality.
*/

#pragma once

#include <string>
#include <libssh2.h>
#include <libssh2_sftp.h>
#include <functional>


namespace Datacratic {


/*****************************************************************************/
/* SOCKET CONNECTION                                                         */
/*****************************************************************************/

struct SocketConnection {
    int sock;

    SocketConnection();

    ~SocketConnection();

    void connect(const std::string & hostname,
                 const std::string & port);

    void close();
};


/*****************************************************************************/
/* SSH CONNECTION                                                            */
/*****************************************************************************/

struct SshConnection : public SocketConnection {
    LIBSSH2_SESSION *session;

    SshConnection();

    ~SshConnection();

    void connect(const std::string & hostname,
                 const std::string & port);

    void passwordAuth(const std::string & username,
                      const std::string & password);

    void publicKeyAuth(const std::string & username,
                       const std::string & publicKeyFile,
                       const std::string & privateKeyFile);

    void setBlocking();

    std::string lastError() const;

    void close();
};


/*****************************************************************************/
/* SFTP CONNECTION                                                           */
/*****************************************************************************/

struct SftpConnection : public SshConnection {
    LIBSSH2_SFTP *sftp_session;

    SftpConnection();

    ~SftpConnection();

    void connectPasswordAuth(const std::string & hostname,
                             const std::string & username,
                             const std::string & password,
                             const std::string & port = "ssh");

    void connectPublicKeyAuth(const std::string & hostname,
                              const std::string & username,
                              const std::string & publicKeyFile,
                              const std::string & privateKeyFile,
                              const std::string & port = "ssh");

    struct Attributes : public LIBSSH2_SFTP_ATTRIBUTES {
    };

    struct File {
        std::string path;
        LIBSSH2_SFTP_HANDLE *handle;
        SftpConnection * owner;

        File(const std::string & path,
             LIBSSH2_SFTP_HANDLE * handle,
             SftpConnection * owner);

        ~File();

        Attributes getAttr() const;

        uint64_t size() const;

        void downloadTo(const std::string & filename) const;
    };

    struct Directory {
        std::string path;
        LIBSSH2_SFTP_HANDLE *handle;
        SftpConnection * owner;
        
        Directory(const std::string & path,
                  LIBSSH2_SFTP_HANDLE * handle,
                  SftpConnection * owner);

        ~Directory();

        void ls() const;

        typedef std::function<void (std::string, Attributes)> OnFile;

        void forEachFile(const OnFile & onFile) const;
    };

    Directory getDirectory(const std::string & path);

    File openFile(const std::string & path);

    void uploadFile(const char * start,
                    size_t size,
                    const std::string & path);

    void close();
};




} // namespace Datacratic
