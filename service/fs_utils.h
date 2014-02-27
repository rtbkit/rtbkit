/* fs_utils.h                                                       -*- C++ -*-
   Wolfgang Sourdeau, February 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   A set of file-system abstraction functions intended to support common
   operations among different fs types or alikes.
*/

#pragma once

#define _GNU_SOURCE 1
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

#include "soa/types/date.h"
#include "soa/types/url.h"

namespace Datacratic {

/* URLINFO */

struct UrlInfo {
    UrlInfo()
        : exists(false), size(0)
    {}

    bool exists;
    Date lastModified;
    int64_t size;
    std::string etag;
    std::string storageClass;
    std::string ownerId;
    std::string ownerName;

    JML_IMPLEMENT_OPERATOR_BOOL(exists);
};


/* URLFSHANDLER */

struct UrlFsHandler {
    virtual UrlInfo getInfo(const Url & url) const = 0;

    virtual size_t getSize(const Url & url) const;
    virtual std::string getEtag(const Url & url) const;

    virtual void makeDirectory(const Url & url) const = 0;
    virtual void erase(const Url & url) const = 0;
};

/** Register a new handler for handling URIs of the given scheme. */
void registerUrlFsHandler(const std::string & scheme,
                          UrlFsHandler * handler);


/* FUNCS */

// Return the object info for either a file or an S3 object
UrlInfo getUriObjectInfo(const std::string & filename);

// Return the object info for either a file or an S3 object, or null if
// it doesn't exist
UrlInfo tryGetUriObjectInfo(const std::string & filename);

// Return an URI for either a file or an s3 object
size_t getUriSize(const std::string & filename);

// Return an etag for either a file or an s3 object
std::string getUriEtag(const std::string & filename);

// Create the directories for the given path.  For S3 it does nothing;
// for normal directories it does mkdir -p
void makeUriDirectory(const std::string & uri);

// Erase the object at the given uri
void eraseUriObject(const std::string & uri);

// Erase the object at the given uri
bool tryEraseUriObject(const std::string & uri);

}
