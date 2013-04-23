/** s3.cc
    Jeremy Barnes, 3 July 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Code to talk to s3.
*/

#include "soa/service/s3.h"
#include "jml/utils/string_functions.h"
#include "soa/types/date.h"
#include "soa/types/url.h"
#include "jml/arch/futex.h"
#include "jml/utils/exc_assert.h"
#include "jml/utils/pair_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/filter_streams.h"
#include "jml/arch/timers.h"
#include "jml/utils/ring_buffer.h"
#include "jml/utils/hash.h"
#include "jml/utils/file_functions.h"

#define CRYPTOPP_ENABLE_NAMESPACE_WEAK 1
#include "crypto++/sha.h"
#include "crypto++/md5.h"
#include "crypto++/hmac.h"
#include "crypto++/base64.h"

#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Info.hpp>
#include <curlpp/Infos.hpp>

#include <boost/iostreams/stream_buffer.hpp>
#include <exception>
#include <thread>
#include <unordered_map>

#include <boost/filesystem.hpp>


using namespace std;
using namespace ML;

namespace Datacratic {

double
S3Api::
defaultBandwidthToServiceMbps = 20.0;

S3Api::
S3Api()
{
    bandwidthToServiceMbps = defaultBandwidthToServiceMbps;
}

S3Api::
S3Api(const std::string & accessKeyId,
      const std::string & accessKey,
      double bandwidthToServiceMbps,
      const std::string & defaultProtocol,
      const std::string & serviceUri)
    : accessKeyId(accessKeyId),
      accessKey(accessKey),
      defaultProtocol(defaultProtocol),
      serviceUri(serviceUri),
      bandwidthToServiceMbps(bandwidthToServiceMbps)
{
}

void
S3Api::
init(const std::string & accessKeyId,
     const std::string & accessKey,
     double bandwidthToServiceMbps,
     const std::string & defaultProtocol,
     const std::string & serviceUri)
{
    this->accessKeyId = accessKeyId;
    this->accessKey = accessKey;
    this->defaultProtocol = defaultProtocol;
    this->serviceUri = serviceUri;
    this->bandwidthToServiceMbps = bandwidthToServiceMbps;
}

S3Api::Content::
Content(const tinyxml2::XMLDocument & xml)
{
    tinyxml2::XMLPrinter printer;
    const_cast<tinyxml2::XMLDocument &>(xml).Print(&printer);
    this->contentType = "application/xml";
    this->str = printer.CStr();
    this->hasContent = true;
    this->data = str.c_str();
    this->size = str.length();
}

std::string
S3Api::
getDigestMulti(const std::string & verb,
               const std::string & bucket,
               const std::string & resource,
               const std::string & subResource,
               const std::string & contentType,
               const std::string & contentMd5,
               const std::string & date,
               const std::vector<std::pair<std::string, std::string> > & headers)
{
    map<string, string> canonHeaders;
    for (auto it = headers.begin(), end = headers.end();
         it != end;  ++it) {
        string key = lowercase(it->first);
        if (key.find("x-amz") != 0) continue;

        string value = it->second;
        if (canonHeaders.count(key))
            canonHeaders[key] += ",";
        canonHeaders[key] += value;
    }

    return getDigest(verb, bucket, resource, subResource,
                     contentType, contentMd5, date, canonHeaders);

}

/*
Authorization = "AWS" + " " + AWSAccessKeyId + ":" + Signature;

Signature = Base64( HMAC-SHA1( YourSecretAccessKeyID, UTF-8-Encoding-Of( StringToSign ) ) );

StringToSign = HTTP-Verb + "\n" +
    Content-MD5 + "\n" +
    Content-Type + "\n" +
    Date + "\n" +
    CanonicalizedAmzHeaders +
    CanonicalizedResource;

CanonicalizedResource = [ "/" + Bucket ] +
    <HTTP-Request-URI, from the protocol name up to the query string> +
    [ sub-resource, if present. For example "?acl", "?location", "?logging", or "?torrent"];

CanonicalizedAmzHeaders = <described below>
To construct the CanonicalizedAmzHeaders part of StringToSign, select all HTTP request headers that start with 'x-amz-' (using a case-insensitive comparison) and use the following process.

CanonicalizedAmzHeaders Process
1	Convert each HTTP header name to lower-case. For example, 'X-Amz-Date' becomes 'x-amz-date'.
2	Sort the collection of headers lexicographically by header name.
3	Combine header fields with the same name into one "header-name:comma-separated-value-list" pair as prescribed by RFC 2616, section 4.2, without any white-space between values. For example, the two metadata headers 'x-amz-meta-username: fred' and 'x-amz-meta-username: barney' would be combined into the single header 'x-amz-meta-username: fred,barney'.
4	"Unfold" long headers that span multiple lines (as allowed by RFC 2616, section 4.2) by replacing the folding white-space (including new-line) by a single space.
5	Trim any white-space around the colon in the header. For example, the header 'x-amz-meta-username: fred,barney' would become 'x-amz-meta-username:fred,barney'
6	Finally, append a new-line (U+000A) to each canonicalized header in the resulting list. Construct the CanonicalizedResource element by concatenating all headers in this list into a single string.


*/
std::string
S3Api::
getDigest(const std::string & verb,
          const std::string & bucket,
          const std::string & resource,
          const std::string & subResource,
          const std::string & contentType,
          const std::string & contentMd5,
          const std::string & date,
          const std::map<std::string, std::string> & headers)
{
    string canonHeaderString;

    for (auto it = headers.begin(), end = headers.end();
         it != end;  ++it) {
        string key = lowercase(it->first);
        if (key.find("x-amz") != 0) continue;

        string value = it->second;

        canonHeaderString += key + ":" + value + "\n";
    }

    //cerr << "bucket = " << bucket << " resource = " << resource << endl;

    string canonResource
        = (bucket == "" ? "" : "/" + bucket)
        + resource
        + (subResource.empty() ? "" : "?")
        + subResource;

    string stringToSign
        = verb + "\n"
        + contentMd5 + "\n"
        + contentType + "\n"
        + date + "\n"
        + canonHeaderString
        + canonResource;

    return stringToSign;
}

std::string
S3Api::
sign(const std::string & stringToSign,
     const std::string & accessKey)
{
    typedef CryptoPP::SHA1 Hash;

    size_t digestLen = Hash::DIGESTSIZE;
    byte digest[digestLen];
    CryptoPP::HMAC<Hash> hmac((byte *)accessKey.c_str(), accessKey.length());
    hmac.CalculateDigest(digest,
                         (byte *)stringToSign.c_str(),
                         stringToSign.length());

    // base64
    char outBuf[256];

    CryptoPP::Base64Encoder baseEncoder;
    baseEncoder.Put(digest, digestLen);
    baseEncoder.MessageEnd();
    size_t got = baseEncoder.Get((byte *)outBuf, 256);
    outBuf[got] = 0;

    //cerr << "got " << got << " characters" << endl;

    string base64digest(outBuf, outBuf + got - 1);

    //cerr << "base64digest.size() = " << base64digest.size() << endl;

    return base64digest;
}

S3Api::Response
S3Api::SignedRequest::
performSync() const
{
    int numRetries = 7;

    for (unsigned i = 0;  i < numRetries;  ++i) {
        string responseHeaders;
        string body;

        try {
            responseHeaders.clear();
            body.clear();

            curlpp::Easy myRequest;

            using namespace curlpp::options;
            using namespace curlpp::infos;

            list<string> curlHeaders;
            for (unsigned i = 0;  i < params.headers.size();  ++i) {
                curlHeaders.push_back(params.headers[i].first + ": "
                                      + params.headers[i].second);
            }

            curlHeaders.push_back("Date: " + params.date);
            curlHeaders.push_back("Authorization: " + auth);

            //cerr << "getting " << uri << " " << params.headers << endl;

            uint64_t totalBytesToTransfer = params.expectedBytesToDownload
                + params.content.size;
            double expectedTimeSeconds
                = totalBytesToTransfer
                / 1000000.0
                / bandwidthToServiceMbps;
            int timeout = 15 + std::max<int>(30, expectedTimeSeconds * 3);

#if 0
            cerr << "totalBytesToTransfer = " << totalBytesToTransfer << endl;
            cerr << "expectedTimeSeconds = " << expectedTimeSeconds << endl;
            cerr << "timeout = " << timeout << endl;
#endif

#if 0
            if (params.verb == "GET") ;
            else if (params.verb == "POST") {
                //myRequest.setOpt<Post>(true);
            }
            else if (params.verb == "PUT") {
                myRequest.setOpt<Post>(true);
            }
            else throw ML::Exception("unknown verb " + params.verb);
#endif
            //cerr << "!!!Setting params verb " << params.verb << endl;
            myRequest.setOpt<CustomRequest>(params.verb);

            myRequest.setOpt<curlpp::options::Url>(uri);
            //myRequest.setOpt<Verbose>(true);
            myRequest.setOpt<ErrorBuffer>((char *)0);
            myRequest.setOpt<Timeout>(timeout);
            myRequest.setOpt<NoSignal>(1);

            auto onData = [&] (char * data, size_t ofs1, size_t ofs2) -> size_t
                {
                    //cerr << "called onData for " << ofs1 << " " << ofs2 << endl;
                    return 0;
                };

            auto onWriteData = [&] (char * data, size_t ofs1, size_t ofs2) -> size_t
                {
                    body.append(data, ofs1 * ofs2);
                    return ofs1 * ofs2;
                    //cerr << "called onWrite for " << ofs1 << " " << ofs2 << endl;
                    return 0;
                };

            auto onProgress = [&] (double p1, double p2, double p3, double p4) -> int
                {
                    cerr << "progress " << p1 << " " << p2 << " " << p3 << " "
                         << p4 << endl;
                    return 0;
                };

            bool afterContinue = false;

            auto onHeader = [&] (char * data, size_t ofs1, size_t ofs2) -> size_t
                {
                    string headerLine(data, ofs1 * ofs2);
                    if (headerLine.find("HTTP/1.1 100 Continue") == 0) {
                        afterContinue = true;
                    }
                    else if (afterContinue) {
                        if (headerLine == "\r\n")
                            afterContinue = false;
                    }
                    else {
                        responseHeaders.append(headerLine);
                        //cerr << "got header data " << headerLine << endl;
                    }
                    return ofs1 * ofs2;
                };

            myRequest.setOpt<BoostHeaderFunction>(onHeader);
            myRequest.setOpt<BoostWriteFunction>(onWriteData);
            myRequest.setOpt<BoostProgressFunction>(onProgress);
            //myRequest.setOpt<Header>(true);
            if (params.content.data) {
                string s(params.content.data, params.content.size);
                myRequest.setOpt<PostFields>(s);
            }
            else
            {
                string s;
                myRequest.setOpt<PostFields>(s);
            }
            myRequest.setOpt<PostFieldSize>(params.content.size);
            curlHeaders.push_back(ML::format("Content-Length: %lld",
                                             params.content.size));
            curlHeaders.push_back("Transfer-Encoding:");
            curlHeaders.push_back("Content-Type:");
            myRequest.setOpt<curlpp::options::HttpHeader>(curlHeaders);

            myRequest.perform();

            Response response;
            response.body_ = body;

            curlpp::InfoGetter::get(myRequest, CURLINFO_RESPONSE_CODE,
                                    response.code_);

            if (response.code_ == 500) {
                // Internal server error
                // Wait 10 seconds and retry
                cerr << "Service returned 500: " << endl;
                cerr << "uri is " << uri << endl;
                cerr << "response headers " << responseHeaders << endl;
                cerr << "body is " << body << endl;

                ML::sleep(10);
                continue;  // retry
            }

            double bytesUploaded;

            curlpp::InfoGetter::get(myRequest, CURLINFO_SIZE_UPLOAD,
                                    bytesUploaded);

            //cerr << "uploaded " << bytesUploaded << " bytes" << endl;

            response.header_.parse(responseHeaders);

            return response;
        } catch (const curlpp::LibcurlRuntimeError & exc) {
            cerr << "libCurl returned an error with code " << exc.whatCode()
                 << endl;
            cerr << "error is " << curl_easy_strerror(exc.whatCode())
                 << endl;
            cerr << "uri is " << uri << endl;
            cerr << "headers are " << responseHeaders << endl;
            cerr << "body contains " << body.size() << " bytes" << endl;

            if (i < 2)
                cerr << "retrying" << endl;
            else throw;
        }
    }

    throw ML::Exception("logic error");
}

std::string
S3Api::
signature(const RequestParams & request) const
{
    string digest
        = S3Api::getDigestMulti(request.verb,
                                request.bucket,
                                request.resource, request.subResource,
                                request.contentType, request.contentMd5,
                                request.date, request.headers);

    //cerr << "digest = " << digest << endl;

    return S3Api::sign(digest, accessKey);
}

inline std::string uriEncode(const std::string & str)
{
    return str;
}

S3Api::SignedRequest
S3Api::
prepare(const RequestParams & request) const
{
    string protocol = defaultProtocol;
    if(protocol.length() == 0){
        throw ML::Exception("attempt to perform s3 request without a "
            "default protocol. (Could be caused by S3Api initialisation with "
            "the empty constructor.)");
    }

    SignedRequest result;
    result.params = request;
    result.bandwidthToServiceMbps = bandwidthToServiceMbps;

    if (request.resource.find("//") != string::npos)
        throw ML::Exception("attempt to perform s3 request with double slash: "
                            + request.resource);

    if (request.bucket.empty()) {
        result.uri = protocol + "://" + serviceUri
            + request.resource
            + (request.subResource != "" ? "?" + request.subResource : "");
    }
    else {
        result.uri = protocol + "://" + request.bucket + "." + serviceUri
            + request.resource
            + (request.subResource != "" ? "?" + request.subResource : "");
    }

    for (unsigned i = 0;  i < request.queryParams.size();  ++i) {
        if (i == 0 && request.subResource == "")
            result.uri += "?";
        else result.uri += "&";
        result.uri += uriEncode(request.queryParams[i].first)
            + "=" + uriEncode(request.queryParams[i].second);
    }

    string sig = signature(request);
    result.auth = "AWS " + accessKeyId + ":" + sig;

    //cerr << "result.uri = " << result.uri << endl;
    //cerr << "result.auth = " << result.auth << endl;

    return result;
}

S3Api::Response
S3Api::
get(const std::string & bucket,
    const std::string & resource,
    uint64_t expectedBytesToDownload,
    const std::string & subResource,
    const StrPairVector & headers,
    const StrPairVector & queryParams) const
{
    RequestParams request;
    request.verb = "GET";
    request.bucket = bucket;
    request.resource = resource;
    request.subResource = subResource;
    request.headers = headers;
    request.queryParams = queryParams;
    request.date = Date::now().printRfc2616();
    request.expectedBytesToDownload = expectedBytesToDownload;

    return prepare(request).performSync();
}

    /** Perform a POST request from end to end. */
S3Api::Response
S3Api::
post(const std::string & bucket,
     const std::string & resource,
     const std::string & subResource,
     const StrPairVector & headers,
     const StrPairVector & queryParams,
     const Content & content) const
{
    RequestParams request;
    request.verb = "POST";
    request.bucket = bucket;
    request.resource = resource;
    request.subResource = subResource;
    request.headers = headers;
    request.queryParams = queryParams;
    request.date = Date::now().printRfc2616();
    request.content = content;

    return prepare(request).performSync();
}

S3Api::Response
S3Api::
put(const std::string & bucket,
    const std::string & resource,
    const std::string & subResource,
    const StrPairVector & headers,
    const StrPairVector & queryParams,
    const Content & content) const
{
    RequestParams request;
    request.verb = "PUT";
    request.bucket = bucket;
    request.resource = resource;
    request.subResource = subResource;
    request.headers = headers;
    request.queryParams = queryParams;
    request.date = Date::now().printRfc2616();
    request.content = content;

    return prepare(request).performSync();
}
S3Api::Response
S3Api::
erase(const std::string & bucket,
    const std::string & resource,
    const std::string & subResource,
    const StrPairVector & headers,
    const StrPairVector & queryParams,
    const Content & content) const
{
    RequestParams request;
    request.verb = "DELETE";
    request.bucket = bucket;
    request.resource = resource;
    request.subResource = subResource;
    request.headers = headers;
    request.queryParams = queryParams;
    request.date = Date::now().printRfc2616();
    request.content = content;

    return prepare(request).performSync();
}

template<typename T>
T extract(tinyxml2::XMLNode * element, const std::string & path)
{
    if (!element)
        throw ML::Exception("can't extract from missing element");
    //tinyxml2::XMLHandle handle(element);

    vector<string> splitPath = ML::split(path, '/');
    auto p = element;
    for (unsigned i = 0;  i < splitPath.size();  ++i) {
        p = p->FirstChildElement(splitPath[i].c_str());
        if (!p) {
            element->GetDocument()->Print();
            throw ML::Exception("required key " + splitPath[i]
                                + " not found on path " + path);
        }
    }

    auto text = tinyxml2::XMLHandle(p).FirstChild().ToText();

    if (!text) {
        element->GetDocument()->Print();
        throw ML::Exception("no text at node "  + path);
    }
    return boost::lexical_cast<T>(text->Value());
}

template<typename T>
T extractDef(tinyxml2::XMLNode * element, const std::string & path,
             const T & ifMissing)
{
    if (!element) return ifMissing;

    vector<string> splitPath = ML::split(path, '/');
    auto p = element;
    for (unsigned i = 0;  i < splitPath.size();  ++i) {
        p = p->FirstChildElement(splitPath[i].c_str());
        if (!p)
            return ifMissing;
    }

    auto text = tinyxml2::XMLHandle(p).FirstChild().ToText();

    if (!text) return ifMissing;

    return boost::lexical_cast<T>(text->Value());
}

template<typename T>
T extract(const std::unique_ptr<tinyxml2::XMLDocument> & doc,
          const std::string & path)
{
    return extract<T>(doc.get(), path);
}

template<typename T>
T extractDef(const std::unique_ptr<tinyxml2::XMLDocument> & doc,
             const std::string & path, const T & def)
{
    return extractDef<T>(doc.get(), path, def);
}

namespace {

} // file scope

std::vector<std::pair<std::string, std::string> >
S3Api::ObjectMetadata::
getRequestHeaders() const
{
    std::vector<std::pair<std::string, std::string> > result;
    if (redundancy == REDUNDANCY_REDUCED)
        result.push_back({"x-amz-storage-class", "REDUCED_REDUNDANCY"});
    else if(redundancy == REDUNDANCY_GLACIER)
        result.push_back({"x-amz-storage-class", "GLACIER"});
    if (serverSideEncryption == SSE_AES256)
        result.push_back({"x-amz-server-side-encryption", "AES256"});
    if (contentType != "")
        result.push_back({"Content-Type", contentType});
    if (contentEncoding != "")
        result.push_back({"Content-Encoding", contentEncoding});
    for (auto md: metadata) {
        result.push_back({"x-amz-meta-" + md.first, md.second});
    }
    return result;
}

S3Api::MultiPartUpload
S3Api::
obtainMultiPartUpload(const std::string & bucket,
                      const std::string & resource,
                      const ObjectMetadata & metadata) const
{
    // Contains the resource without the leading slash
    string outputPrefix(resource, 1);

    // Check if there is already a multipart upload in progress
    auto inProgress = get(bucket, "/", 8192, "uploads", {},
                          { { "prefix", outputPrefix } })
        .bodyXml();

    using namespace tinyxml2;

    XMLHandle handle(*inProgress);

    auto upload
        = handle
        .FirstChildElement("ListMultipartUploadsResult")
        .FirstChildElement("Upload")
        .ToElement();

    string uploadId;
    vector<MultiPartUploadPart> parts;

    uint64_t partSize = 0;
    uint64_t currentOffset = 0;

    for (; upload; upload = upload->NextSiblingElement("Upload")) {
        XMLHandle uploadHandle(upload);

        auto foundNode
            = uploadHandle
            .FirstChildElement("UploadId")
            .FirstChild()
            .ToText();

        if (!foundNode)
            throw ML::Exception("found node has no ID");

        // TODO: check metadata, etc

        // Already an upload in progress
        uploadId = foundNode->Value();

        auto inProgressInfo = get(bucket, resource, 8192,
                                  "uploadId=" + uploadId)
            .bodyXml();

        //inProgressInfo->Print();

        XMLHandle handle(*inProgressInfo);

        auto foundPart
            = handle
            .FirstChildElement("ListPartsResult")
            .FirstChildElement("Part")
            .ToElement();

        int numPartsDone = 0;
        uint64_t biggestPartSize = 0;
        for (; foundPart;
             foundPart = foundPart->NextSiblingElement("Part"),
                 ++numPartsDone) {
            MultiPartUploadPart currentPart;
            currentPart.fromXml(foundPart);
            if (currentPart.partNumber != numPartsDone + 1) {
                //cerr << "missing part " << numPartsDone + 1 << endl;
                // from here we continue alone
                break;
            }
            currentPart.startOffset = currentOffset;
            currentOffset += currentPart.size;
            biggestPartSize = std::max(biggestPartSize, currentPart.size);
            parts.push_back(currentPart);
        }

        partSize = biggestPartSize;

        //cerr << "numPartsDone = " << numPartsDone << endl;
        //cerr << "currentOffset = " << currentOffset
        //     << "dataSize = " << dataSize << endl;
    }

    if (uploadId.empty()) {
        //cerr << "getting new ID" << endl;

        vector<pair<string, string> > headers = metadata.getRequestHeaders();
        auto result = post(bucket, resource, "uploads", headers).bodyXml();
        //result->Print();
        //cerr << "result = " << result << endl;

        uploadId
            = extract<string>(result, "InitiateMultipartUploadResult/UploadId");

        //cerr << "new upload = " << uploadId << endl;
    }
        //return;

    MultiPartUpload result;
    result.parts.swap(parts);
    result.id = uploadId;
    return result;
}

std::string
S3Api::
finishMultiPartUpload(const std::string & bucket,
                      const std::string & resource,
                      const std::string & uploadId,
                      const std::vector<std::string> & etags) const
{
    using namespace tinyxml2;
    // Finally, send back a response to join the parts together
    XMLDocument joinRequest;
    auto r = joinRequest.InsertFirstChild(joinRequest.NewElement("CompleteMultipartUpload"));
    for (unsigned i = 0;  i < etags.size();  ++i) {
        auto n = r->InsertEndChild(joinRequest.NewElement("Part"));
        n->InsertEndChild(joinRequest.NewElement("PartNumber"))
            ->InsertEndChild(joinRequest.NewText(ML::format("%d", i + 1).c_str()));
        n->InsertEndChild(joinRequest.NewElement("ETag"))
            ->InsertEndChild(joinRequest.NewText(etags[i].c_str()));
    }
    auto joinResponse
        = post(bucket, resource, "uploadId=" + uploadId,
                  {}, {}, joinRequest);

    //cerr << joinResponse.bodyXmlStr() << endl;

    auto joinResponseXml = joinResponse.bodyXml();

    try {

        string etag = extract<string>(joinResponseXml,
                                      "CompleteMultipartUploadResult/ETag");
        return etag;
    } catch (const std::exception & exc) {
        cerr << "--- request is " << endl;
        joinRequest.Print();
        cerr << "error completing multipart upload: " << exc.what() << endl;
        throw;
    }
}

void
S3Api::MultiPartUploadPart::
fromXml(tinyxml2::XMLElement * element)
{
    partNumber = extract<int>(element, "PartNumber");
    lastModified = extract<string>(element, "LastModified");
    etag = extract<string>(element, "ETag");
    size = extract<uint64_t>(element, "Size");
    done = true;
}

std::string
S3Api::
upload(const char * data,
       size_t dataSize,
       const std::string & bucket,
       const std::string & resource,
       CheckMethod check,
       const ObjectMetadata & metadata,
       int numInParallel)
{
    if (resource == "" || resource[0] != '/')
        throw ML::Exception("resource should start with a /");
    // Contains the resource without the leading slash
    string outputPrefix(resource, 1);

    //cerr << "need to upload " << dataSize << " bytes" << endl;

    // Check if it's already there

    if (check == CM_SIZE || check == CM_MD5_ETAG) {
        auto existingResource
            = get(bucket, "/", 8192, "", {},
                  { { "prefix", outputPrefix } })
            .bodyXml();

        //cerr << "existing" << endl;
        //existingResource->Print();

        auto foundContent
            = tinyxml2::XMLHandle(*existingResource)
            .FirstChildElement("ListBucketResult")
            .FirstChildElement("Contents")
            .ToElement();

        if (foundContent) {
            uint64_t size = extract<uint64_t>(foundContent, "Size");
            std::string etag = extract<string>(foundContent, "ETag");
            std::string lastModified = extract<string>(foundContent, "LastModified");

            if (size == dataSize) {
                //cerr << "already uploaded" << endl;
                return etag;
            }
        }
    }

    auto upload = obtainMultiPartUpload(bucket, resource, metadata);

    uint64_t partSize = 0;
    uint64_t currentOffset = 0;

    for (auto & part: upload.parts) {
        partSize = std::max(partSize, part.size);
        currentOffset = std::max(currentOffset, part.startOffset + part.size);
    }

    if (partSize == 0) {
        if (dataSize < 5 * 1024 * 1024) {
            partSize = dataSize;
        }
        else {
            partSize = 8 * 1024 * 1024;
            while (dataSize / partSize > 150) {
                partSize *= 2;
            }
        }
    }

    string uploadId = upload.id;
    vector<MultiPartUploadPart> & parts = upload.parts;

    uint64_t offset = currentOffset;
    for (int i = 0;  offset < dataSize;  offset += partSize, ++i) {
        MultiPartUploadPart part;
        part.partNumber = parts.size() + 1;
        part.startOffset = offset;
        part.size = min<uint64_t>(partSize, dataSize - offset);
        parts.push_back(part);
    }
    // we are dealing with an empty file
    if(parts.empty())
    {
        MultiPartUploadPart part;
        part.partNumber = parts.size() + 1;
        part.startOffset = offset;
        part.size = min<uint64_t>(partSize, dataSize - offset);
        parts.push_back(part);
    }
    //cerr << "total parts = " << parts.size() << endl;

    //if (!foundId)

    uint64_t bytesDone = 0;
    Date start;

    auto touchByte = [] (const char * c)
        {

            __asm__
            (" # [in]"
             :
             : [in] "r" (*c)
             :
             );
        };

    auto touch = [&] (const char * start, size_t size)
        {
            for (size_t i = 0;  i < size;  i += 4096) {
                touchByte(start + i);
            }
        };

    int readyPart = 0;

    auto doPart = [&] (int i)
        {
            MultiPartUploadPart & part = parts[i];
            //cerr << "part " << i << " with " << part.size << " bytes" << endl;

            // Wait until we're allowed to go
            for (;;) {
                int isReadyPart = readyPart;
                if (isReadyPart >= i) {
                    break;
                }
                futex_wait(readyPart, isReadyPart);
            }

            // First touch the input range
            touch(data + part.startOffset,
                  part.size);

            //cerr << "done touching " << i << endl;

            // Now let the next one go
            ExcAssertEqual(readyPart, i);
            ++readyPart;

            futex_wake(readyPart);

            string md5 = md5HashToHex(data + part.startOffset,
                                      part.size);

            if (part.done) {
                //cerr << "etag is " << part.etag << endl;
                if ('"' + md5 + '"' == part.etag) {
                    //cerr << "part " << i << " verified done" << endl;
                    return;
                }
            }

            auto putResult = put(bucket, resource,
                                    ML::format("partNumber=%d&uploadId=%s",
                                               part.partNumber, uploadId),
                                    {}, {},
                                    S3Api::Content(data
                                                   + part.startOffset,
                                                   part.size,
                                                   md5));

            //cerr << "result of part " << i << " is "
            //<< putResult.bodyXmlStr() << endl;

            if (putResult.code_ != 200) {
                part.etag = "ERROR";
                cerr << putResult.bodyXmlStr() << endl;
                throw ML::Exception("put didn't work: %d", (int)putResult.code_);
            }



            ML::atomic_add(bytesDone, part.size);

#if 0
            double seconds = Date::now().secondsSince(start);
            cerr << "done " << bytesDone / 1024 / 1024 << " MB in "
            << seconds << " s at "
            << bytesDone / 1024.0 / 1024 / seconds
            << " MB/second" << endl;
#endif
            //cerr << putResult.header_ << endl;

            string etag = putResult.getHeader("etag");

            //cerr << "etag = " << etag << endl;

            part.etag = etag;
        };

    int currentPart = 0;

    start = Date::now();

    auto doPartThread = [&] ()
        {
            for (;;) {
                if (currentPart >= parts.size()) break;
                int partToDo = __sync_fetch_and_add(&currentPart, 1);
                if (partToDo >= parts.size()) break;
                doPart(partToDo);
            }
        };

    if (numInParallel == -1)
        numInParallel = 16;

    boost::thread_group tg;
    for (unsigned i = 0;  i < numInParallel;  ++i)
        tg.create_thread(doPartThread);

    tg.join_all();

    vector<string> etags;
    for (unsigned i = 0;  i < parts.size();  ++i) {
        etags.push_back(parts[i].etag);
    }
    string finalEtag = finishMultiPartUpload(bucket, resource, uploadId, etags);
    return finalEtag;
}

std::string
S3Api::
upload(const char * data,
       size_t bytes,
       const std::string & uri,
       CheckMethod check,
       const ObjectMetadata & metadata,
       int numInParallel)
{
    string bucket, object;
    std::tie(bucket, object) = parseUri(uri);
    return upload(data, bytes, bucket, "/" + object, check, metadata,
                  numInParallel);
}

S3Api::ObjectInfo::
ObjectInfo()
    : size(0), exists(false)
{
}

S3Api::ObjectInfo::
ObjectInfo(tinyxml2::XMLNode * element)
{
    size = extract<uint64_t>(element, "Size");
    key  = extract<string>(element, "Key");
    string lastModifiedStr = extract<string>(element, "LastModified");
    lastModified = Date::parseIso8601(lastModifiedStr);
    etag = extract<string>(element, "ETag");
    ownerId = extract<string>(element, "Owner/ID");
    ownerName = extractDef<string>(element, "Owner/DisplayName", "");
    storageClass = extract<string>(element, "StorageClass");
    exists = true;
}

void
S3Api::
forEachObject(const std::string & bucket,
              const std::string & prefix,
              const OnObject & onObject,
              const OnSubdir & onSubdir,
              const std::string & delimiter,
              int depth) const
{
    using namespace tinyxml2;

    //cerr << "forEachObject under " << prefix << endl;

    string marker;
    do {
        StrPairVector queryParams;
        if (prefix != "")
            queryParams.push_back({"prefix", prefix});
        if (delimiter != "")
            queryParams.push_back({"delimiter", delimiter});
        if (marker != "")
            queryParams.push_back({"marker", marker});

        auto listingResult = get(bucket, "/", 8192, "",
                                 {}, queryParams);
        auto listingResultXml = listingResult.bodyXml();

        //listingResultXml->Print();

        string foundPrefix
            = extractDef<string>(listingResult, "ListBucketResult/Prefix", "");
        string truncated
            = extract<string>(listingResult, "ListBucketResult/IsTruncated");
        if (truncated == "true") {
            marker = extract<string>(listingResult, "ListBucketResult/Marker");
        }
        else marker = "";

        auto foundObject
            = XMLHandle(*listingResultXml)
            .FirstChildElement("ListBucketResult")
            .FirstChildElement("Contents")
            .ToElement();

        for (; onObject && foundObject;
             foundObject = foundObject->NextSiblingElement("Contents")) {
            ObjectInfo info(foundObject);

            ExcAssertEqual(info.key.find(foundPrefix), 0);
            string basename(info.key, foundPrefix.length());

            if (!onObject(foundPrefix, basename, info, depth))
                break;
        }

        auto foundDir
            = XMLHandle(*listingResultXml)
            .FirstChildElement("ListBucketResult")
            .FirstChildElement("CommonPrefixes")
            .ToElement();

        for (; onSubdir && foundDir;
             foundDir = foundDir->NextSiblingElement("CommonPrefixes")) {
            string dirName = extract<string>(foundDir, "Prefix");

            // Strip off the delimiter
            if (dirName.rfind(delimiter) == dirName.size() - delimiter.size()) {
                dirName = string(dirName, 0, dirName.size() - delimiter.size());
                ExcAssertEqual(dirName.find(prefix), 0);
                dirName = string(dirName, prefix.size());
            }
            if (onSubdir(foundPrefix, dirName, depth)) {
                string newPrefix = foundPrefix + dirName + "/";
                //cerr << "newPrefix = " << newPrefix << endl;
                forEachObject(bucket, newPrefix, onObject, onSubdir, delimiter,
                              depth + 1);
            }
        }
    } while (marker != "");

    //cerr << "done scanning" << endl;
}

S3Api::ObjectInfo
S3Api::
getObjectInfo(const std::string & bucket,
              const std::string & object) const
{
    StrPairVector queryParams;
    queryParams.push_back({"prefix", object});

    auto listingResult = get(bucket, "/", 8192, "", {}, queryParams);

    if (listingResult.code_ != 200) {
        cerr << listingResult.bodyXmlStr() << endl;
        throw ML::Exception("error getting object");
    }

    auto listingResultXml = listingResult.bodyXml();

    auto foundObject
        = tinyxml2::XMLHandle(*listingResultXml)
        .FirstChildElement("ListBucketResult")
        .FirstChildElement("Contents")
        .ToElement();

    if (!foundObject)
        throw ML::Exception("object " + object + " not found in bucket "
                            + bucket);

    ObjectInfo info(foundObject);

    if(info.key != object){
        throw ML::Exception("object " + object + " not found in bucket "
                            + bucket);
    }


    return info;
}

S3Api::ObjectInfo
S3Api::
tryGetObjectInfo(const std::string & bucket,
                 const std::string & object) const
{
    StrPairVector queryParams;
    queryParams.push_back({"prefix", object});

    auto listingResult = get(bucket, "/", 8192, "", {}, queryParams);
    if (listingResult.code_ != 200) {
        cerr << listingResult.bodyXmlStr() << endl;
        throw ML::Exception("error getting object request: %d",
                            listingResult.code_);
    }
    auto listingResultXml = listingResult.bodyXml();

    auto foundObject
        = tinyxml2::XMLHandle(*listingResultXml)
        .FirstChildElement("ListBucketResult")
        .FirstChildElement("Contents")
        .ToElement();

    if (!foundObject)
        return ObjectInfo();

    ObjectInfo info(foundObject);

    if(info.key != object){
        return ObjectInfo();
    }

    return info;
}

S3Api::ObjectInfo
S3Api::
getObjectInfo(const std::string & uri) const
{
    string bucket, object;
    std::tie(bucket, object) = parseUri(uri);
    return getObjectInfo(bucket, object);
}

S3Api::ObjectInfo
S3Api::
tryGetObjectInfo(const std::string & uri) const
{
    string bucket, object;
    std::tie(bucket, object) = parseUri(uri);
    return tryGetObjectInfo(bucket, object);
}

void
S3Api::
download(const std::string & uri,
         const OnChunk & onChunk,
         ssize_t startOffset,
         ssize_t endOffset) const
{
    string bucket, object;
    std::tie(bucket, object) = parseUri(uri);
    return download(bucket, object, onChunk, startOffset, endOffset);
}

void
S3Api::
download(const std::string & bucket,
         const std::string & object,
         const OnChunk & onChunk,
         ssize_t startOffset,
         ssize_t endOffset) const
{

    ObjectInfo info = getObjectInfo(bucket, object);
    if(info.storageClass == "GLACIER"){
        throw ML::Exception("Cannot download [" + info.key + "] because its "
            "storage class is [GLACIER]");
    }

    size_t chunkSize = 128 * 1024 * 1024;  // 128MB probably good

    struct Part {
        uint64_t offset;
        uint64_t size;
    };

    if (endOffset == -1)
        endOffset = info.size;

    //cerr << "getting " << endOffset << " bytes" << endl;

    vector<Part> parts;

    for (uint64_t offset = 0;  offset < endOffset;  offset += chunkSize) {
        Part part;
        part.offset = offset;
        part.size = std::min<ssize_t>(endOffset - offset, chunkSize);
        parts.push_back(part);
    }

    //cerr << "getting in " << parts.size() << " parts" << endl;

    uint64_t bytesDone = 0;
    Date start;
    bool failed = false;

    auto doPart = [&] (int i)
        {
            if (failed) return;

            Part & part = parts[i];
            //cerr << "part " << i << " with " << part.size << " bytes" << endl;

            StrPairVector headerParams;
            headerParams.push_back({"range",
                        ML::format("bytes=%zd-%zd",
                                   part.offset,
                                   part.offset + part.size - 1)});

            auto partResult = get(bucket, "/" + object, part.size, "", headerParams, {});
            if (partResult.code_ != 206) {
                cerr << "error getting part " << i << ": "
                     << partResult.bodyXmlStr() << endl;
                failed = true;
                return;
            }

            ExcAssertEqual(partResult.body_.size(), part.size);

            onChunk(partResult.body_.c_str(),
                    part.size,
                    i,
                    part.offset,
                    info.size);

            ML::atomic_add(bytesDone, part.size);
            double seconds = Date::now().secondsSince(start);
            cerr << "done " << bytesDone / 1024 / 1024 << " MB in "
            << seconds << " s at "
            << bytesDone / 1024.0 / 1024 / seconds
            << " MB/second" << endl;
        };

    int currentPart = 0;

    start = Date::now();

    auto doPartThread = [&] ()
        {
            for (;;) {
                if (currentPart >= parts.size()) break;
                int partToDo = __sync_fetch_and_add(&currentPart, 1);
                if (partToDo >= parts.size()) break;
                doPart(partToDo);
            }
        };

    boost::thread_group tg;
    for (unsigned i = 0;  i < 16;  ++i)
        tg.create_thread(doPartThread);

    tg.join_all();

    if (failed)
        throw ML::Exception("Failed to get part");
}

/**
 * Downloads a file from s3 to a local file. If the maxSize is specified, only
 * the first maxSize bytes will be downloaded.
 */
void
S3Api::
downloadToFile(const std::string & uri, const std::string & outfile,
        ssize_t endOffset) const
{

    auto info = getObjectInfo(uri);
    if (!info){
        throw ML::Exception("unknown s3 object");
    }
    if(endOffset == -1 || endOffset > info.size){
        endOffset = info.size;
    }

    ofstream myFile;
    myFile.open(outfile.c_str());

    uint64_t done = 0;

    auto onChunk = [&] (const char * data,
                            size_t size,
                            int chunkIndex,
                            uint64_t offset,
                            uint64_t totalSize){
        ExcAssertEqual(info.size, totalSize);
        ExcAssertLessEqual(offset + size, totalSize);
        myFile.seekp(offset);
        myFile.write(data, size);
        ML::atomic_add(done, size);
    };
    download(uri, onChunk, 0, endOffset);
}

struct StreamingDownloadSource {

    StreamingDownloadSource(const S3Api * owner,
                            const std::string & bucket,
                            const std::string & object)
    {
        impl.reset(new Impl());
        impl->owner = owner;
        impl->bucket = bucket;
        impl->object = object;
        impl->info = owner->getObjectInfo(bucket, object);
        impl->chunkSize = 1024 * 1024;  // start with 1MB and ramp up

        int numThreads = 1;
        if (impl->info.size > 1024 * 1024)
            numThreads = 2;
        if (impl->info.size > 16 * 1024 * 1024)
            numThreads = 3;
        if (impl->info.size > 256 * 1024 * 1024)
            numThreads = 5;
        
        impl->start(numThreads);
    }

    typedef char char_type;
    struct category
        : //input_seekable,
        boost::iostreams::input,
        boost::iostreams::device_tag,
        boost::iostreams::closable_tag
    { };

    struct Impl {
        Impl()
            : owner(0), offset(0), shutdown(false),
              readPartOffset(0), readPartDone(1)
        {
        }

        ~Impl()
        {
            stop();
        }

        const S3Api * owner;
        S3Api::ObjectInfo info;
        std::string bucket;
        std::string object;
        size_t offset;
        size_t chunkSize;
        size_t bytesDone;
        bool shutdown;
        boost::thread_group tg;

        string readPart;
        size_t readPartOffset;

        Date startDate;

        size_t writeOffset, readOffset;
        int readPartReady, readPartDone, writePartNumber, allocPartNumber;


        void start(int numThreads)
        {
            readPartOffset = offset = bytesDone = writeOffset
                = writePartNumber = allocPartNumber = readOffset = 0;
            readPartReady = 0;
            readPartDone = 0;
            startDate = Date::now();
            for (unsigned i = 0;  i < numThreads;  ++i)
                tg.create_thread(boost::bind<void>(&Impl::runThread, this));
        }

        void stop()
        {
            shutdown = true;
            ML::memory_barrier();
            futex_wake(writePartNumber);
            futex_wake(readPartReady);
            futex_wake(readPartDone);
            tg.join_all();
        }

        std::streamsize read(char_type* s, std::streamsize n)
        {
            if (readOffset == info.size)
                return -1;

            //Date start = Date::now();

#if 0
            cerr << "read: readPartReady = " << readPartReady
                 << " readPartDone = " << readPartDone
                 << " writePartNumber = " << writePartNumber
                 << " allocPartNumber = " << allocPartNumber
                 << " readPartOffset = " << readPartOffset
                 << endl;
#endif

            //cerr << "trying to read " << n << " characters at offset "
            //     << readPartOffset << " of "
            //     << readPart.size() << endl;

            while (readPartDone == readPartReady) {
                //cerr << "waiting for part " << readPartDone << endl;
                ML::futex_wait(readPartReady, readPartDone);
            }

            ExcAssertGreaterEqual(readPartReady, readPartDone);

            //cerr << "ready to start reading" << endl;

            //cerr << "trying to read " << n << " characters at offset "
            //     << readPartOffset << " of "
            //     << readPart.size() << endl;

            ExcAssertGreaterEqual(readPart.size(), readPartOffset);

            size_t toDo = std::min<size_t>(readPart.size() - readPartOffset,
                                           n);

            //cerr << "toDo = " << toDo << endl;

            std::copy(readPart.c_str() + readPartOffset,
                      readPart.c_str() + readPartOffset + toDo,
                      s);

            readPartOffset += toDo;
            readOffset += toDo;
            if (readPartOffset == readPart.size()) {
                //cerr << "finished part " << readPartDone << endl;
                ++readPartDone;
                ML::futex_wake(readPartDone);
            }

            //Date end = Date::now();
            //double elapsed = end.secondsSince(start);
            //if (elapsed > 0.0001)
            //    cerr << "read elapsed " << elapsed << endl;

            return toDo;
        }

        void runThread()
        {
            // Maximum chunk size is what we can do in 30 seconds
            size_t maxChunkSize
                = owner->bandwidthToServiceMbps
                * 15.0 * 1000000;

            while (!shutdown) {
                // Go in the lottery to see which part I need to download
                int partToDo = __sync_fetch_and_add(&allocPartNumber, 1);

                //cerr << "partToDo = " << partToDo << endl;

                // Wait until it's my turn to increment the offset
                while (!shutdown) {
                    int currentWritePart = writePartNumber;
                    if (currentWritePart >= partToDo) break;
                    ML::futex_wait(writePartNumber, currentWritePart, 0.1);
                }
                if (shutdown) return;

                //cerr << "ready" << endl;

                ExcAssertEqual(writePartNumber, partToDo);

                // If we're done then get out
                if (writeOffset >= info.size || shutdown) return;  // done

                if (partToDo && partToDo % 2 == 0 && chunkSize < maxChunkSize)
                    chunkSize *= 2;

                size_t start = writeOffset;
                size_t end = std::min<size_t>(writeOffset + chunkSize,
                                              info.size);

                writeOffset = end;

                // Finished my turn to increment.  Wake up the next thread
                ++writePartNumber;
                futex_wake(writePartNumber);

                // Download my part
                S3Api::StrPairVector headerParams;
                headerParams.push_back({"range",
                            ML::format("bytes=%zd-%zd",
                                       start, end - 1)});

                //cerr << "downloading" << endl;

                auto partResult
                    = owner->get(bucket, "/" + object, end - start,
                                 "", headerParams, {});
                if (partResult.code_ != 206) {
                    cerr << "error getting part "
                         << partResult.bodyXmlStr() << endl;
                    return;
                }

                //cerr << "done downloading" << endl;

                // Wait until the reader needs my part
                while (!shutdown) {
                    int currentReadPart = readPartDone;
                    if (currentReadPart == partToDo) break;
                    ML::futex_wait(readPartDone, currentReadPart, 0.1);
                }
                if (shutdown) return;

                //cerr << "ready for part " << partToDo << endl;

                bytesDone += partResult.body_.size();

                //double elapsed = Date::now().secondsSince(startDate);

                // Give my part to the reader
                readPart = partResult.body();
                readPartOffset = 0;

                // Wake up the reader
                ++readPartReady;
                ML::memory_barrier();
                ML::futex_wake(readPartReady);

#if 0
                cerr << "done " << bytesDone << " at "
                     << bytesDone / elapsed / 1024 / 1024
                     << "MB/second" << endl;
#endif
            }
            //cerr << "finished thread" << endl;
        }
    };

    std::shared_ptr<Impl> impl;

    std::streamsize read(char_type* s, std::streamsize n)
    {
        return impl->read(s, n);
    }

    bool is_open() const
    {
        return !!impl;
    }

    void close()
    {
        impl.reset();
    }
};

std::auto_ptr<std::streambuf>
S3Api::
streamingDownload(const std::string & bucket,
                  const std::string & object,
                  ssize_t startOffset,
                  ssize_t endOffset,
                  const OnChunk & onChunk) const
{
    std::auto_ptr<std::streambuf> result;
    result.reset(new boost::iostreams::stream_buffer<StreamingDownloadSource>
                 (StreamingDownloadSource(this, bucket, object),
                  131072));
    return result;
}

struct StreamingUploadSource {

    StreamingUploadSource(const S3Api * owner,
                          const std::string & bucket,
                          const std::string & object,
                          const S3Api::ObjectMetadata & metadata)
    {
        impl.reset(new Impl());
        impl->owner = owner;
        impl->bucket = bucket;
        impl->object = object;
        impl->metadata = metadata;
        impl->chunkSize = 8 * 1024 * 1024;  // start with 8MB and ramp up

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
            : owner(0), offset(0), chunkIndex(0), shutdown(false),
              chunks(16)
        {
        }

        ~Impl()
        {
            //cerr << "destroying streaming upload at " << object << endl;
            stop();
        }

        const S3Api * owner;
        std::string bucket;
        std::string object;
        S3Api::ObjectMetadata metadata;
        std::string uploadId;
        size_t offset;
        size_t chunkSize;
        size_t chunkIndex;
        bool shutdown;
        boost::thread_group tg;

        Date startDate;

        struct Chunk {

            void init(uint64_t offset, size_t capacity, int index)
            {
                this->offset = offset;
                this->size = 0;
                this->capacity = capacity;
                this->index = index;
                this->data = new char[capacity];
            }

            size_t append(const char * input, size_t n)
            {
                size_t todo = std::min(n, capacity - size);
                std::copy(input, input + todo, data + size);
                size += todo;
                return todo;
            }

            char * data;
            size_t size;
            size_t capacity;
            int index;
            uint64_t offset;
        };

        Chunk current;

        RingBufferSWMR<Chunk> chunks;

        boost::mutex etagsLock;
        std::vector<std::string> etags;
        std::exception_ptr exc;

        void start()
        {
            auto upload = owner->obtainMultiPartUpload(bucket, "/" + object, metadata);

            uploadId = upload.id;
            //cerr << "uploadId = " << uploadId << endl;

            startDate = Date::now();
            for (unsigned i = 0;  i < 8;  ++i)
                tg.create_thread(boost::bind<void>(&Impl::runThread, this));
            current.init(0, chunkSize, 0);
        }

        void stop()
        {
            shutdown = true;
            tg.join_all();
        }

        std::streamsize write(const char_type* s, std::streamsize n)
        {
            if (exc)
                std::rethrow_exception(exc);

            size_t done = current.append(s, n);
            offset += done;
            if (done < n) {
                flush();
                done += current.append(s + done, n - done);
            }

            //cerr << "writing " << n << " characters returned "
            //     << done << endl;

            if (exc)
                std::rethrow_exception(exc);

            return done;
        }

        void flush()
        {
            if (current.size == 0) return;
            chunks.push(current);
            ++chunkIndex;

            // Get bigger for bigger files
            if (chunkIndex % 5 == 0 && chunkSize < 64 * 1024 * 1024)
                chunkSize *= 2;

            current.init(offset, chunkSize, chunkIndex);
        }

        void finish()
        {
            if (exc)
                std::rethrow_exception(exc);
            //cerr << "pushing last chunk " << chunkIndex << endl;
            flush();
            //cerr << "waiting for everything to stop" << endl;
            chunks.waitUntilEmpty();
            //cerr << "empty" << endl;
            stop();
            //cerr << "stopped" << endl;
            string etag = owner->finishMultiPartUpload(bucket, "/" + object,
                                                       uploadId,
                                                       etags);
            //cerr << "final etag is " << etag << endl;

            if (exc)
                std::rethrow_exception(exc);

            double elapsed = Date::now().secondsSince(startDate);

            cerr << "uploaded " << offset / 1024.0 / 1024.0
                 << "MB in " << elapsed << "s at "
                 << offset / 1024.0 / 1024.0 / elapsed
                 << "MB/s" << " to " << etag << endl;
        }

        void runThread()
        {
            while (!shutdown) {
                Chunk chunk;
                if (chunks.tryPop(chunk, 0.01)) {
                    if (exc)
                        return;
                    try {
                        //cerr << "got chunk " << chunk.index
                        //     << " with " << chunk.size << " bytes at index "
                        //     << chunk.index << endl;

                        // Upload the data
                        string md5 = md5HashToHex(chunk.data, chunk.size);

                        auto putResult = owner->put(bucket, "/" + object,
                                                    ML::format("partNumber=%d&uploadId=%s",
                                                               chunk.index + 1, uploadId),
                                                    {}, {},
                                                    S3Api::Content(chunk.data,
                                                                   chunk.size,
                                                                   md5));
                        if (putResult.code_ != 200) {
                            cerr << putResult.bodyXmlStr() << endl;

                            throw ML::Exception("put didn't work: %d", (int)putResult.code_);
                        }
                        string etag = putResult.getHeader("etag");
                        //cerr << "successfully uploaded part " << chunk.index
                        //     << " with etag " << etag << endl;

                        boost::unique_lock<boost::mutex> guard(etagsLock);
                        while (etags.size() <= chunk.index)
                            etags.push_back("");
                        etags[chunk.index] = etag;
                    } catch (...) {
                        // Capture exception to be thrown later
                        exc = std::current_exception();
                    }
                }
            }
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

std::auto_ptr<std::streambuf>
S3Api::
streamingUpload(const std::string & uri,
                const ObjectMetadata & metadata) const
{
    string bucket, object;
    std::tie(bucket, object) = parseUri(uri);
    return streamingUpload(bucket, object, metadata);
}

std::auto_ptr<std::streambuf>
S3Api::
streamingUpload(const std::string & bucket,
                const std::string & object,
                const ObjectMetadata & metadata) const
{
    std::auto_ptr<std::streambuf> result;
    result.reset(new boost::iostreams::stream_buffer<StreamingUploadSource>
                 (StreamingUploadSource(this, bucket, object, metadata),
                  131072));
    return result;
}

std::pair<std::string, std::string>
S3Api::
parseUri(const std::string & uri)
{
    if (uri.find("s3://") != 0)
        throw ML::Exception("wrong scheme (should start with s3://)");
    string pathPart(uri, 5);
    string::size_type pos = pathPart.find('/');
    if (pos == string::npos)
        throw ML::Exception("couldn't find bucket name");
    string bucket(pathPart, 0, pos);
    string object(pathPart, pos + 1);

    return make_pair(bucket, object);
}

std::auto_ptr<std::streambuf>
S3Api::
streamingDownload(const std::string & uri,
                  ssize_t startOffset,
                  ssize_t endOffset,
                  const OnChunk & onChunk) const
{
    string bucket, object;
    std::tie(bucket, object) = parseUri(uri);

    //cerr << "bucket = " << bucket << " object = " << object << endl;

    return streamingDownload(bucket, object, startOffset, endOffset, onChunk);
}

void
S3Handle::
initS3(const std::string & accessKeyId,
       const std::string & accessKey,
       const std::string & uriPrefix)
{
    s3.init(accessKeyId, accessKey);
    this->s3UriPrefix = uriPrefix;
}

size_t
S3Handle::
getS3Buffer(const std::string & filename, char** outBuffer){
    auto stats = s3.getObjectInfo(filename);
    if (!stats)
        throw ML::Exception("unknown s3 object");

    *outBuffer = new char[stats.size];

    uint64_t done = 0;

    auto onChunk = [&] (const char * data,
                        size_t size,
                        int chunkIndex,
                        uint64_t offset,
                        uint64_t totalSize)
        {
            ExcAssertEqual(stats.size, totalSize);
            ExcAssertLessEqual(offset + size, totalSize);
            std::copy(data, data + size, *outBuffer + offset);
            ML::atomic_add(done, size);
        };

    s3.download(filename, onChunk);

    ExcAssertEqual(done, stats.size);

    cerr << "done downloading " << stats.size << " bytes from "
         << filename << endl;

    return stats.size;

}

bool
S3Api::
forEachBucket(const OnBucket & onBucket) const
{
    using namespace tinyxml2;

    //cerr << "forEachObject under " << prefix << endl;

    auto listingResult = get("", "/", 8192, "");
    auto listingResultXml = listingResult.bodyXml();

    //listingResultXml->Print();

    auto foundBucket
        = XMLHandle(*listingResultXml)
        .FirstChildElement("ListAllMyBucketsResult")
        .FirstChildElement("Buckets")
        .FirstChildElement("Bucket")
        .ToElement();

    for (; onBucket && foundBucket;
         foundBucket = foundBucket->NextSiblingElement("Bucket")) {

        string foundName
            = extract<string>(foundBucket, "Name");
        if (!onBucket(foundName))
            return false;
    }

    return true;
}

void
S3Api::
uploadRecursive(string dirSrc, string bucketDest, bool includeDir){
    using namespace boost::filesystem;
    path targetDir(dirSrc);
    if(!is_directory(targetDir)){
        throw ML::Exception("%s is not a directory", dirSrc.c_str());
    }
    recursive_directory_iterator it(targetDir), itEnd;
    int toTrim = includeDir ? 0 : dirSrc.length() + 1;
    for(; it != itEnd; it ++){
        if(!is_directory(*it)){
            string path = it->path().string();
            ML::File_Read_Buffer frb(path);
            size_t size = file_size(path);
            if(toTrim){
                path = path.substr(toTrim);
            }
            upload(frb.start(), size, "s3://" + bucketDest + "/" + path);
        }
    }
}

void S3Api::setDefaultBandwidthToServiceMbps(double mbps){
    S3Api::defaultBandwidthToServiceMbps = mbps;
}

namespace {

struct S3BucketInfo {
    std::string s3Bucket;
    std::shared_ptr<S3Api> api;  //< Used to access this uri
};

std::mutex s3BucketsLock;
std::unordered_map<std::string, S3BucketInfo> s3Buckets;

} // file scope

/** S3 support for filter_ostream opens.  Register the bucket name here, and
    you can open it directly from s3.
*/

void registerS3Bucket(const std::string & bucketName,
                      const std::string & accessKeyId,
                      const std::string & accessKey,
                      double bandwidthToServiceMbps,
                      const std::string & protocol,
                      const std::string & serviceUri)
{
    std::unique_lock<std::mutex> guard(s3BucketsLock);

    auto it = s3Buckets.find(bucketName);
    if(it != s3Buckets.end()){
        shared_ptr<S3Api> api = it->second.api;
        //if the info is different, raise an exception, otherwise return
        if (api->accessKeyId != accessKeyId
            || api->accessKey != accessKey
            || api->bandwidthToServiceMbps != bandwidthToServiceMbps
            || api->defaultProtocol != protocol
            || api->serviceUri != serviceUri)
        {
            throw ML::Exception("Trying to re-register a bucket with different "
                "parameters");
        }
        return;
    }

    S3BucketInfo info;
    info.s3Bucket = bucketName;
    info.api = std::make_shared<S3Api>(accessKeyId, accessKey,
                                       bandwidthToServiceMbps,
                                       protocol, serviceUri);

    s3Buckets[bucketName] = info;
}

struct RegisterS3Handler {
    static std::pair<std::streambuf *, bool>
    getS3Handler(const std::string & scheme,
                 const std::string & resource,
                 std::ios_base::open_mode mode)
    {
        string::size_type pos = resource.find('/');
        if (pos == string::npos)
            throw ML::Exception("unable to find s3 bucket name in resource "
                                + resource);
        string bucket(resource, 0, pos);

        std::shared_ptr<S3Api> api;

        {
            std::unique_lock<std::mutex> guard(s3BucketsLock);
            auto it = s3Buckets.find(bucket);
            if (it == s3Buckets.end())
                throw ML::Exception("unregistered s3 bucket " + bucket);
            api = it->second.api;
        }

        ExcAssert(api);

        if (mode == ios::in) {
            return make_pair(api->streamingDownload("s3://" + resource)
                             .release(),
                             true);
        }
        else if (mode == ios::out) {
            return make_pair(api->streamingUpload("s3://" + resource)
                             .release(),
                             true);
        }
        else throw ML::Exception("no way to create s3 handler for non in/out");
    }

    RegisterS3Handler()
    {
        ML::registerUriHandler("s3", getS3Handler);
    }

} registerS3Handler;

void registerS3Buckets(const std::string & accessKeyId,
                       const std::string & accessKey,
                       double bandwidthToServiceMbps,
                       const std::string & protocol,
                       const std::string & serviceUri)
{
    std::unique_lock<std::mutex> guard(s3BucketsLock);

    auto api = std::make_shared<S3Api>(accessKeyId, accessKey,
                                       bandwidthToServiceMbps,
                                       protocol, serviceUri);

    auto onBucket = [&] (const std::string & bucketName)
        {
            //cerr << "got bucket " << bucketName << endl;

            S3BucketInfo info;
            info.s3Bucket = bucketName;
            info.api = api;
            s3Buckets[bucketName] = info;

            return true;
        };

    api->forEachBucket(onBucket);
}

std::shared_ptr<S3Api> getS3ApiForBucket(const std::string & bucketName)
{
    std::unique_lock<std::mutex> guard(s3BucketsLock);
    auto it = s3Buckets.find(bucketName);
    if (it == s3Buckets.end())
        throw ML::Exception("unregistered s3 bucket " + bucketName);
    return it->second.api;
}

// Return an URI for either a file or an s3 object
size_t getUriSize(const std::string & filename)
{
    if (filename.find("s3://") == 0) {
        string bucket = S3Api::parseUri(filename).first;
        auto api = getS3ApiForBucket(bucket);
        return api->getObjectInfo(filename).size;
    }
    else {
        struct stat stats;
        int res = stat(filename.c_str(), &stats);
        if (res == -1)
            throw ML::Exception("error getting stats file");
        return stats.st_size;
    }
}

} // namespace Datacratic
