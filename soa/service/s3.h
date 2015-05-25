/* s3.h                                                            -*- C++ -*-
   Jeremy Barnes, 3 July 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Class to deal with doing s3.
   Note: Your access key must have the listallmybuckets permission on the aws side.
*/

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include "tinyxml2/tinyxml2.h"
#include "jml/arch/exception.h"
#include "jml/utils/unnamed_bool.h"
#include "jml/utils/filter_streams.h"
#include "aws.h"
#include "fs_utils.h"
#include "http_endpoint.h"
#include "http_rest_proxy.h"

#include "soa/types/basic_value_descriptions.h"
#include "soa/types/value_description.h"

namespace Datacratic {

struct S3Config {
    std::string accessKeyId;
    std::string accessKey;
};
CREATE_STRUCTURE_DESCRIPTION(S3Config);


/*****************************************************************************/
/* S3 OBJECTINFO TYPES                                                       */
/*****************************************************************************/

/* This enum contains the list of attributes that can be queried via the
 * S3Api::getObjectInfo functions */

enum S3ObjectInfoTypes {
    LASTMODIFIED  = 1 << 0,
    SIZE          = 1 << 1,
    ETAG          = 1 << 2,

    STORAGECLASS  = 1 << 3,
    OWNERID       = 1 << 4,
    OWNERNAME     = 1 << 5,

    SHORT_INFO = LASTMODIFIED | SIZE | ETAG,
    FULL_EXTRAS = STORAGECLASS | OWNERID | OWNERNAME,
    FULL_INFO = SHORT_INFO | FULL_EXTRAS
};


/*****************************************************************************/
/* S3 API                                                                    */
/*****************************************************************************/

/** Interface to Amazon's S3 service. */

struct S3Api : public AwsApi {
    /** Default value for bandwidth to service.  In mega*bytes* per second.
        Default value is 20.0 MBPS for ec2 instances in the same availability
        zone.
    */
    static double defaultBandwidthToServiceMbps;

    S3Api();

    /** Set up the API to called with the given credentials. */
    S3Api(const std::string & accessKeyId,
          const std::string & accessKey,
          double bandwidthToServiceMbps = defaultBandwidthToServiceMbps,
          const std::string & defaultProtocol = "http",
          const std::string & serviceUri = "s3.amazonaws.com");

    /** Set up the API to called with the given credentials. */
    void init();
    void init(const std::string & accessKeyId,
              const std::string & accessKey,
              double bandwidthToServiceMbps = defaultBandwidthToServiceMbps,
              const std::string & defaultProtocol = "http",
              const std::string & serviceUri = "s3.amazonaws.com");

    std::string accessKeyId;
    std::string accessKey;
    std::string defaultProtocol;
    std::string serviceUri;
    double bandwidthToServiceMbps;

    typedef std::vector<std::pair<std::string, std::string> > StrPairVector;

    struct Content {
        Content()
            : data(0), size(0), hasContent(false)
        {
        }

        Content(const char * data, uint64_t size,
                const std::string & contentType = "",
                const std::string & contentMd5 = "")
            : data(data), size(size), hasContent(true),
              contentType(contentType), contentMd5(contentMd5)
        {
        }

        Content(const tinyxml2::XMLDocument & xml);

        const char * data;
        uint64_t size;
        bool hasContent;

        std::string str;
        std::string contentType;
        std::string contentMd5;
    };

    struct Range {
        Range(uint64_t aSize)
            : offset(0), size(aSize)
        {}

        Range(uint64_t aOffset, uint64_t aSize)
            : offset(aOffset), size(aSize)
        {}

        static Range Full;

        uint64_t endPos() const
        { return (offset + size - 1); }

        void adjust(size_t downloaded)
        {
            if (downloaded > size) {
                throw ML::Exception("excessive adjustment size:"
                                    " downloaded %zu size %lu",
                                     downloaded, size);
            }
            offset += downloaded;
            size -= downloaded;
        }

        bool operator == (const Range & other) const
        { return offset == other.offset && size == other.size; }

        bool operator != (const Range & other) const
        { return !(*this == other); }

        uint64_t offset;
        uint64_t size;
    };

    /** A set of parameters that specify a request. */
    struct RequestParams {
        RequestParams()
            : downloadRange(0)
        {
        }

        std::string verb;
        std::string bucket;
        std::string resource;
        std::string subResource;
        std::string date;

        std::string contentType;
        std::string contentMd5;
        Content content;
        Range downloadRange;

        StrPairVector headers;
        StrPairVector queryParams;
    };

    /** The response of a request.  Has a return code and a body. */
    struct Response {
        Response()
            : code_(0)
        {
        }

        std::string body() const
        {
            if (code_ < 200 || code_ >= 300)
                throw ML::Exception("invalid http code returned");
            return body_;
        }

        std::unique_ptr<tinyxml2::XMLDocument> bodyXml() const
        {
            if (code_ != 200)
                throw ML::Exception("invalid http code returned");
            std::unique_ptr<tinyxml2::XMLDocument> result(new tinyxml2::XMLDocument());
            result->Parse(body_.c_str());
            return result;
        }

        operator std::unique_ptr<tinyxml2::XMLDocument>() const
        {
            return bodyXml();
        }

        std::string bodyXmlStr() const
        {
            auto x = bodyXml();
            tinyxml2::XMLPrinter printer;
            x->Print(&printer);
            return printer.CStr();
        }

        std::string getHeader(const std::string & name) const
        {
            auto it = header_.headers.find(name);
            if (it == header_.headers.end())
                throw ML::Exception("required header " + name + " not found");
            return it->second;
        }

        long code_;
        std::string body_;
        HttpHeader header_;
    };

    enum Redundancy {
        REDUNDANCY_DEFAULT,
        REDUNDANCY_STANDARD,
        REDUNDANCY_REDUCED,
        REDUNDANCY_GLACIER
    };

    /** Set the meaning of REDUNDANCY_DEFAULT.  Default is REDUNDANCY_STANDARD.
     */
    static void setDefaultRedundancy(Redundancy redundancy);

    /** Get the meaning of REDUNDANCY_DEFAULT.  */
    static Redundancy getDefaultRedundancy();

    enum ServerSideEncryption {
        SSE_NONE,
        SSE_AES256
    };

    struct ObjectMetadata {
        ObjectMetadata()
            : redundancy(REDUNDANCY_DEFAULT),
              serverSideEncryption(SSE_NONE),
              numThreads(8)
        {
        }

        ObjectMetadata(const Redundancy & redundancy)
            : redundancy(redundancy),
              serverSideEncryption(SSE_NONE),
              numThreads(8)
        {
        }

        std::vector<std::pair<std::string, std::string> >
        getRequestHeaders() const;

        Redundancy redundancy;
        ServerSideEncryption serverSideEncryption;
        std::string contentType;
        std::string contentEncoding;
        std::map<std::string, std::string> metadata;
        std::string acl;
        unsigned int numThreads;
    };

    /** Signed request that can be executed. */
    struct SignedRequest {
        RequestParams params;
        std::string auth;
        std::string uri;
        double bandwidthToServiceMbps;
        S3Api * owner;

        /** Perform the request synchronously and return the result. */
        Response performSync() const;
    };

    /** Calculate the signature for a given request. */
    std::string signature(const RequestParams & request) const;

    /** Prepare a request to be executed. */
    SignedRequest prepare(const RequestParams & request) const;

    /** Escape a resource used by S3; this in particular leaves a slash
        in place. */
    static std::string s3EscapeResource(const std::string & resource);

    /** Perform a HEAD request from end to end. */
    Response head(const std::string & bucket,
                  const std::string & resource,
                  const std::string & subResource = "",
                  const StrPairVector & headers = StrPairVector(),
                  const StrPairVector & queryParams = StrPairVector())
        const
    {
        return headEscaped(bucket, s3EscapeResource(resource), subResource,
                           headers, queryParams);
    }

    /** Perform a GET request from end to end. */
    Response get(const std::string & bucket,
                 const std::string & resource,
                 const Range & downloadRange,
                 const std::string & subResource = "",
                 const StrPairVector & headers = StrPairVector(),
                 const StrPairVector & queryParams = StrPairVector())
        const
    {
        return getEscaped(bucket, s3EscapeResource(resource), downloadRange,
                          subResource, headers, queryParams);
    }


    /** Perform a POST request from end to end. */
    Response post(const std::string & bucket,
                  const std::string & resource,
                  const std::string & subResource = "",
                  const StrPairVector & headers = StrPairVector(),
                  const StrPairVector & queryParams = StrPairVector(),
                  const Content & content = Content())
        const
    {
        return postEscaped(bucket, s3EscapeResource(resource), subResource,
                           headers, queryParams, content);
    }

    /** Perform a PUT request from end to end including data. */
    Response put(const std::string & bucket,
                 const std::string & resource,
                 const std::string & subResource = "",
                 const StrPairVector & headers = StrPairVector(),
                 const StrPairVector & queryParams = StrPairVector(),
                 const Content & content = Content())
        const
    {
        return putEscaped(bucket, s3EscapeResource(resource), subResource,
                          headers, queryParams, content);
    }

    /** Perform a DELETE request from end to end including data. */
    Response erase(const std::string & bucket,
                   const std::string & resource,
                   const std::string & subResource = "",
                   const StrPairVector & headers = StrPairVector(),
                   const StrPairVector & queryParams = StrPairVector(),
                   const Content & content = Content())
        const
    {
        return eraseEscaped(bucket, s3EscapeResource(resource), subResource,
                            headers, queryParams, content);
    }

    enum CheckMethod {
        CM_SIZE,     ///< Check via the size of the content
        CM_MD5_ETAG, ///< Check via the md5 of the content vs the etag
        CM_ASSUME_INVALID  ///< Anything there is assumed invalid
    };

    /** Upload a memory buffer into an s3 bucket.  Uses a multi-part upload
        algorithm that can achieve 200MB/second for data already in memory.

        If the resource already exists, then it will use the given method
        to determine whether it's OK or not.

        Returns the etag field of the uploaded file.
    */
    std::string upload(const char * data,
                       size_t bytes,
                       const std::string & bucket,
                       const std::string & resource,
                       CheckMethod check = CM_SIZE,
                       const ObjectMetadata & md = ObjectMetadata(),
                       int numInParallel = -1);

    std::string upload(const char * data,
                       size_t bytes,
                       const std::string & uri,
                       CheckMethod check = CM_SIZE,
                       const ObjectMetadata & md = ObjectMetadata(),
                       int numInParallel = -1);

    typedef std::function<void (const char * chunk,
                                size_t size,
                                int chunkIndex,
                                uint64_t offset,
                                uint64_t totalSize) >
        OnChunk;

    /** OnChunk function that writes to the given file. */
    static OnChunk writeToFile(const std::string & filename);

    /** Download the contents of a bucket.  This will call the given
        output function for each chunk that is received.  Note that there
        is no guarantee that the chunks will be received in order as the
        download happens in multiple parallel chunks.
    */
    void download(const std::string & bucket,
                  const std::string & object,
                  const OnChunk & onChunk,
                  ssize_t startOffset = 0,
                  ssize_t endOffset = -1) const;

    void download(const std::string & uri,
                  const OnChunk & onChunk,
                  ssize_t startOffset = 0,
                  ssize_t endOffset = -1) const;

    void downloadToFile(const std::string & uri,
                  const std::string & outfile,
                  ssize_t endOffset = -1) const;

    struct ObjectInfo : public FsObjectInfo {
        ObjectInfo()
        {}

        ObjectInfo(tinyxml2::XMLNode * element);
        ObjectInfo(const S3Api::Response & response);

        std::string key;
    };

    typedef std::function<bool (const std::string & prefix,
                                const std::string & objectName,
                                const ObjectInfo & info,
                                int depth)>
        OnObject;

    typedef std::function<bool (const std::string & prefix,
                                const std::string & dirName,
                                int depth)>
        OnSubdir;

    /** For each file matching the given prefix in the given bucket, call
        the callback.
    */
    void forEachObject(const std::string & bucket,
                       const std::string & prefix = "",
                       const OnObject & onObject = OnObject(),
                       const OnSubdir & onSubdir = OnSubdir(),
                       const std::string & delimiter = "/",
                       int depth = 1,
                       const std::string & startAt = "") const;

    typedef std::function<bool (const std::string & uri,
                                const ObjectInfo & info,
                                int depth)>
        OnObjectUri;

    /** For each file matching the given prefix in the given bucket, call
        the callback.
    */
    void forEachObject(const std::string & uriPrefix,
                       const OnObjectUri & onObject,
                       const OnSubdir & onSubdir = OnSubdir(),
                       const std::string & delimiter = "/",
                       int depth = 1,
                       const std::string & startAt = "") const;

    /** Value for the "delimiter" parameter in forEachObject for when we
        don't want any subdirectories.  It is equal to the empty string.
    */
    static const std::string NO_SUBDIRS;

    /** Does the object exist? */
    ObjectInfo tryGetObjectInfo(const std::string & bucket,
                                const std::string & object,
                                S3ObjectInfoTypes infos = SHORT_INFO) const;
    ObjectInfo tryGetObjectInfo(const std::string & uri,
                                S3ObjectInfoTypes infos = SHORT_INFO) const;


    /** Return the ObjectInfo about the object.  Throws an exception if it
        doesn't exist.
    */
    ObjectInfo getObjectInfo(const std::string & bucket,
                             const std::string & object,
                             S3ObjectInfoTypes infos = SHORT_INFO) const;
    ObjectInfo getObjectInfo(const std::string & uri,
                             S3ObjectInfoTypes infos = SHORT_INFO) const;

    /** Erase the given object.  Throws an exception if it fails. */
    void eraseObject(const std::string & bucket,
                     const std::string & object);

    /** Erase the given object.  Throws an exception if it fails. */
    void eraseObject(const std::string & uri);

    /** Erase the given object.  Returns true if an object was erased or false
        otherwise.
    */
    bool tryEraseObject(const std::string & bucket,
                        const std::string & object);
    
    /** Erase the given object.  Returns true if an object was erased or false
        otherwise.
    */
    bool tryEraseObject(const std::string & uri);

    /** Return the public URI that should be used to access a public object. */
    static std::string getPublicUri(const std::string & uri,
                                    const std::string & protocol);

    static std::string getPublicUri(const std::string & bucket,
                                    const std::string & object,
                                    const std::string & protocol);

    typedef std::function<bool (std::string bucket)> OnBucket;

    /** Call the given callback on every bucket in the current
        account.
    */
    bool forEachBucket(const OnBucket & bucket) const;

    /** Turn a s3:// uri string into a bucket name and object. */
    static std::pair<std::string, std::string>
    parseUri(const std::string & uri);

    struct MultiPartUploadPart {
        MultiPartUploadPart()
            : partNumber(0), done(false)
        {
        }

        int partNumber;
        uint64_t startOffset;
        uint64_t size;
        std::string lastModified;
        std::string contentMd5;
        std::string etag;
        bool done;

        void fromXml(tinyxml2::XMLElement * element);
    };

    struct MultiPartUpload {
        std::string id;
        std::vector<MultiPartUploadPart> parts;
    };

    enum UploadRequirements {
        UR_EXISTING,   ///< OK to return an existing one
        UR_FRESH,      ///< Must be a fresh one
        UR_EXCLUSIVE   ///< Must be a fresh one, and no other may exist
    };

    /** Obtain a multipart upload, either in progress or a new one. */
    MultiPartUpload
    obtainMultiPartUpload(const std::string & bucket,
                          const std::string & resource,
                          const ObjectMetadata & metadata,
                          UploadRequirements requirements) const;

    std::pair<bool,std::string>
    isMultiPartUploadInProgress(const std::string & bucket,
                                const std::string & resource) const;

    std::string
    finishMultiPartUpload(const std::string & bucket,
                          const std::string & resource,
                          const std::string & uploadId,
                          const std::vector<std::string> & etags) const;

    void uploadRecursive(std::string dirSrc,
                         std::string bucketDest,
                         bool includeDir);

    /** Pre-escaped versions of the above methods */

    /* head */
    Response headEscaped(const std::string & bucket,
                         const std::string & resource,
                         const std::string & subResource = "",
                         const StrPairVector & headers = StrPairVector(),
                         const StrPairVector & queryParams = StrPairVector()) const;

    /* get */
    Response getEscaped(const std::string & bucket,
                        const std::string & resource,
                        const Range & downloadRange,
                        const std::string & subResource = "",
                        const StrPairVector & headers = StrPairVector(),
                        const StrPairVector & queryParams = StrPairVector()) const;

    /* post */
    Response postEscaped(const std::string & bucket,
                         const std::string & resource,
                         const std::string & subResource = "",
                         const StrPairVector & headers = StrPairVector(),
                         const StrPairVector & queryParams = StrPairVector(),
                         const Content & content = Content()) const;

    /* put */
    Response putEscaped(const std::string & bucket,
                        const std::string & resource,
                        const std::string & subResource = "",
                        const StrPairVector & headers = StrPairVector(),
                        const StrPairVector & queryParams = StrPairVector(),
                        const Content & content = Content()) const;

    /* erase */
    Response eraseEscaped(const std::string & bucket,
                          const std::string & resource,
                          const std::string & subResource,
                          const StrPairVector & headers = StrPairVector(),
                          const StrPairVector & queryParams = StrPairVector(),
                          const Content & content = Content()) const;

    //easy handle for v8 wrapping
    void setDefaultBandwidthToServiceMbps(double mpbs);

private:
    ObjectInfo tryGetObjectInfoShort(const std::string & bucket,
                                     const std::string & object) const;
    ObjectInfo tryGetObjectInfoFull(const std::string & bucket,
                                    const std::string & object) const;
    ObjectInfo getObjectInfoShort(const std::string & bucket,
                                  const std::string & object) const;
    ObjectInfo getObjectInfoFull(const std::string & bucket,
                                 const std::string & object) const;

    // Used to pool connections to the S3 service
    static HttpRestProxy proxy;

    /// Static variable to hold the default redundancy to be used
    static Redundancy defaultRedundancy;

};

struct S3Handle{
    S3Api s3;
    std::string s3UriPrefix;

    void initS3(const std::string & accessKeyId,
                const std::string & accessKey,
                const std::string & uriPrefix);

    size_t getS3Buffer(const std::string & filename, char** outBuffer);
};

/** S3 support for filter_ostream opens.  Register the bucket name here, and
    you can open it directly from s3.
*/

void registerS3Bucket(const std::string & bucketName,
                      const std::string & accessKeyId,
                      const std::string & accessKey,
                      double bandwidthToServiceMbps = S3Api::defaultBandwidthToServiceMbps,
                      const std::string & protocol = "http",
                      const std::string & serviceUri = "s3.amazonaws.com");

/** S3 support for filter_ostream opens.  Register the bucket name here, and
    you can open it directly from s3.  Queries and iterates over all
    buckets within the account.
*/

void registerS3Buckets(const std::string & accessKeyId,
                       const std::string & accessKey,
                       double bandwidthToServiceMbps = S3Api::defaultBandwidthToServiceMbps,
                       const std::string & protocol = "http",
                       const std::string & serviceUri = "s3.amazonaws.com");

std::shared_ptr<S3Api> getS3ApiForBucket(const std::string & bucketName);

std::shared_ptr<S3Api> getS3ApiForUri(const std::string & uri);

std::tuple<std::string, std::string, std::string, std::string, std::string> 
    getCloudCredentials();

/** Returns the keyId, key and list of buckets to register (can be empty,
    which means all) from the environment variables

    S3_KEY_ID, S3_KEY and S3_BUCKETS
*/
std::tuple<std::string, std::string, std::vector<std::string> >
getS3CredentialsFromEnvVar();

// std::pair<std::string, std::string> getDefaultCredentials();

} // namespace Datacratic
