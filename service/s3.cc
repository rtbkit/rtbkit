/** s3.cc
    Jeremy Barnes, 3 July 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Code to talk to s3.
*/

/* last known compatible commit in master branch:
 * dda48cb65cf6a1689efa9d6c7949fd0d43c964d7 */

#include <atomic>
#include <mutex>

#include "soa/service/s3.h"
#include "jml/utils/string_functions.h"
#include "soa/types/date.h"
#include "soa/types/url.h"
#include "soa/utils/print_utils.h"
#include "jml/arch/futex.h"
#include "jml/arch/threads.h"
#include "jml/utils/exc_assert.h"
#include "jml/utils/pair_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/ring_buffer.h"
#include "jml/utils/hash.h"
#include "jml/utils/file_functions.h"
#include "jml/utils/info.h"
#include "xml_helpers.h"

#define CRYPTOPP_ENABLE_NAMESPACE_WEAK 1
#include "crypto++/sha.h"
#include "crypto++/md5.h"
#include "crypto++/hmac.h"
#include "crypto++/base64.h"

#include <boost/iostreams/stream_buffer.hpp>
#include <exception>
#include <unordered_map>

#include <boost/filesystem.hpp>

#include "message_loop.h"
#include "http_client.h"
#include "fs_utils.h"


using namespace std;
using namespace ML;
using namespace Datacratic;

namespace {

/****************************************************************************/
/* S3 GLOBALS                                                               */
/****************************************************************************/

struct S3Globals {
    S3Globals()
        : baseRetryDelay(3), numRetries(-1)
    {
        if (numRetries == -1) {
            char * numRetriesEnv = getenv("S3_RETRIES");
            if (numRetriesEnv) {
                numRetries = atoi(numRetriesEnv);
            }
            else {
                numRetries = 45;
            }
        }

        loop.start();
    }

    shared_ptr<HttpClient> &
    getClient(const string & bucket,
              const string & baseHostname = "s3.amazonaws.com")
    {
        string hostname = bucket;
        if (hostname.size() > 0) {
            hostname += ".";
        }
        hostname += baseHostname;

        unique_lock<mutex> guard(clientsLock);
        auto & client = clients[hostname];
        if (!client) {
            client.reset(new HttpClient(hostname, 30));
            client->sendExpect100Continue(false);
            loop.addSource("s3-client-" + hostname, client);
        }

        return client;
    }

    int baseRetryDelay;
    int numRetries;
    MessageLoop loop;

private:
    mutex clientsLock;
    map<string, shared_ptr<HttpClient> > clients;
};

static S3Globals &
getS3Globals()
{
    static S3Globals s3Config;
    return s3Config;
}


/****************************************************************************/
/* SYNC RESPONSE                                                            */
/****************************************************************************/

/* This class provides a standard way of propagating the response to an async
 * requests from the worker thread to the caller. */

struct SyncResponse {
    SyncResponse()
        : data_(new Data())
    {
    }

    S3Api::Response response()
    {
        return std::move(data_->response());
    }

    void operator () (S3Api::Response && response)
    {
        data_->setResponse(std::move(response));
    }

private:
    struct Data {
        Data()
            : done_(false)
        {
        }

        S3Api::Response response()
        {
            while (!done_) {
                ML::futex_wait(done_, false);
            }
            if (response_.excPtr_) {
                rethrow_exception(response_.excPtr_);
            }

            return std::move(response_);
        }

        void setResponse(S3Api::Response && response)
        {
            response_ = std::move(response);
            done_ = true;
            ML::futex_wake(done_);
        }

        int done_;
        S3Api::Response response_;
    };

    shared_ptr<Data> data_;
};


/****************************************************************************/
/* S3 URL FS HANDLER                                                        */
/****************************************************************************/

struct S3UrlFsHandler : public UrlFsHandler {
    virtual FsObjectInfo getInfo(const Url & url) const
    {
        string bucket = url.host();
        auto api = getS3ApiForBucket(bucket);
        return api->getObjectInfo(bucket, url.path().substr(1));
    }

    virtual FsObjectInfo tryGetInfo(const Url & url) const
    {
        string bucket = url.host();
        auto api = getS3ApiForBucket(bucket);
        return api->tryGetObjectInfo(bucket, url.path().substr(1));
    }

    virtual void makeDirectory(const Url & url) const
    {
    }

    virtual bool erase(const Url & url, bool throwException) const
    {
        string bucket = url.host();
        auto api = getS3ApiForBucket(bucket);
        if (throwException) {
            api->eraseObject(bucket, url.path());
            return true;
        }
        else { 
            return api->tryEraseObject(bucket, url.path());
        }
    }

    virtual bool forEach(const Url & prefix,
                         const OnUriObject & onObject,
                         const OnUriSubdir & onSubdir,
                         const std::string & delimiter,
                         const std::string & startAt) const
    {
        string bucket = prefix.host();
        auto api = getS3ApiForBucket(bucket);

        bool result = true;

        auto onObject2 = [&] (const std::string & prefix,
                              const std::string & objectName,
                              const S3Api::ObjectInfo & info,
                              int depth)
            {
                return onObject("s3://" + bucket + "/" + prefix + objectName,
                                info, depth);
            };

        auto onSubdir2 = [&] (const std::string & prefix,
                              const std::string & dirName,
                              int depth)
            {
                return onSubdir("s3://" + bucket + "/" + prefix + dirName,
                                depth);
            };

        // Get rid of leading / on prefix
        string prefix2 = string(prefix.path(), 1);

        api->forEachObject(bucket, prefix2, onObject2,
                           onSubdir ? onSubdir2 : S3Api::OnSubdir(),
                           delimiter, 1, startAt);

        return result;
    }
};


/****************************************************************************/
/* S3 DOWNLOADER                                                            */
/****************************************************************************/

size_t getTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}

struct S3Downloader {
    S3Downloader(const string & urlStr,
                 ssize_t startOffset = 0, ssize_t endOffset = -1)
        : owner(getS3ApiForUri(urlStr)),
          info(owner->getObjectInfo(urlStr)),
          offset(startOffset),
          baseChunkSize(1024*1024), // start with 1MB and ramp up
          closed(false),
          readOffset(0),
          readPartOffset(-1),
          currentChunk(0),
          requestedBytes(0),
          currentRq(0),
          activeRqs(0)
    {
        if (!info) {
            throw ML::Exception("missing object: " + urlStr);
        }

        std::tie(bucket, object) = S3Api::parseUri(urlStr);
        if (endOffset == -1 || endOffset > info.size) {
            endOffset = info.size;
        }
        downloadSize = endOffset - startOffset;

        /* Maximum chunk size is what we can do in 3 seconds, up to 1% of
           system memory. */
        maxChunkSize = owner->bandwidthToServiceMbps * 3.0 * 1000000;
        size_t sysMemory = getTotalSystemMemory();
        maxChunkSize = std::min(maxChunkSize, sysMemory / 100);

        /* The maximum number of concurrent requests is set depending on
           the total size of the stream. */
        maxRqs = 1;
        if (info.size > 1024 * 1024)
            maxRqs = 2;
        if (info.size > 16 * 1024 * 1024)
            maxRqs = 3;
        if (info.size > 256 * 1024 * 1024)
            maxRqs = 5;
        chunks.resize(maxRqs);

        /* Kick start the requests */
        ensureRequests();
    }

    ~S3Downloader()
    {
        /* We ensure at runtime that "close" is called because it is mandatory
           for the proper cleanup of active requests. Because "close" can
           throw, we cannot however call it from the destructor. */
        if (!closed) {
            cerr << "destroying S3Downloader without invoking close()\n";
            abort();
        }
    }

    std::streamsize read(char * s, std::streamsize n)
    {
        if (closed) {
            throw ML::Exception("invoking read() on a closed download");
        }

        if (endOfDownload()) {
            return -1;
        }

        if (readPartOffset == -1) {
            waitNextPart();
        }
        ensureRequests();

        size_t toDo = min<size_t>(readPart.size() - readPartOffset,
                                  n);
        const char * start = readPart.c_str() + readPartOffset;
        std::copy(start, start + toDo, s);

        readPartOffset += toDo;
        if (readPartOffset == readPart.size()) {
            readPartOffset = -1;
        }

        readOffset += toDo;

        return toDo;
    }

    uint64_t getDownloadSize()
        const
    {
        return downloadSize;
    }

    bool endOfDownload()
        const
    {
        return (readOffset == downloadSize);
    }

    void close()
    {
        closed = true;
        while (activeRqs > 0) {
            ML::futex_wait(activeRqs, activeRqs);
        }
        excPtrHandler.rethrowIfSet();
    }

private:
    /* download Chunk */
    struct Chunk {
        enum State {
            IDLE,
            QUERY,
            RESPONSE
        };

        Chunk() noexcept
            : state(IDLE)
        {
        }

        Chunk(Chunk && other) noexcept
            : state(other.state.load()),
              data(std::move(other.data))
        {
        }

        void setQuerying()
        {
            ExcAssertEqual(state, IDLE);
            setState(QUERY);
        }

        void assign(string newData)
        {
            ExcAssertEqual(state, QUERY);
            data = move(newData);
            setState(RESPONSE);
            ML::futex_wake(state);
        }

        std::string retrieve()
        {
            ExcAssertEqual(state, RESPONSE);
            string chunkData = std::move(data);
            setState(IDLE);
            return std::move(chunkData);
        }

        void setState(int newState)
        {
            state = newState;
            ML::futex_wake(state);
        }

        bool isIdle()
            const
        {
            return (state == IDLE);
        }

        bool waitResponse(double timeout)
            const
        {
            if (timeout > 0.0) {
                int old = state;
                if (state != RESPONSE) {
                    ML::futex_wait(state, old, timeout);
                }
            }

            return (state == RESPONSE);
        }

    private:
        std::atomic<int> state;
        string data;
    };

    void waitNextPart()
    {
        unsigned int chunkNr(currentChunk % maxRqs);
        Chunk & chunk = chunks[chunkNr];
        while (!excPtrHandler.hasException() && !chunk.waitResponse(1.0));
        excPtrHandler.rethrowIfSet();
        readPart = chunk.retrieve();
        readPartOffset = 0;
        currentChunk++;
    }

    void ensureRequests()
    {
        while (true) {
            if (excPtrHandler.hasException()) {
                break;
            }
            if (activeRqs == maxRqs) {
                break;
            }
            ExcAssert(activeRqs < maxRqs);
            if (requestedBytes == downloadSize) {
                break;
            }
            ExcAssert(requestedBytes < downloadSize);

            Chunk & chunk = chunks[currentRq % maxRqs];
            if (!chunk.isIdle()) {
                break;
            }

            ensureRequest();
        }
    }

    void ensureRequest()
    {
        size_t chunkSize = getChunkSize(currentRq);
        uint64_t end = requestedBytes + chunkSize;
        if (end > info.size) {
            end = info.size;
            chunkSize = end - requestedBytes;
        }

        unsigned int chunkNr = currentRq % maxRqs;
        Chunk & chunk = chunks[chunkNr];
        activeRqs++;
        chunk.setQuerying();

        auto onResponse = [&, chunkNr, chunkSize] (S3Api::Response && response) {
            this->handleResponse(chunkNr, chunkSize, std::move(response));
        };
        S3Api::Range range(offset + requestedBytes, chunkSize);
        owner->getAsync(onResponse, bucket, "/" + object, range);
        ExcAssertLess(currentRq, UINT_MAX);
        currentRq++;
        requestedBytes += chunkSize;
    }

    void handleResponse(unsigned int chunkNr, size_t chunkSize,
                        S3Api::Response && response)
    {
        try {
            if (response.excPtr_) {
                rethrow_exception(response.excPtr_);
            }

            if (response.code_ != 206) {
                throw ML::Exception("http error "
                                    + to_string(response.code_)
                                    + " while getting chunk "
                                    + response.bodyXmlStr());
            }

            /* It can sometimes happen that a file changes during download i.e
               it is being overwritten. Make sure we check for this condition
               and throw an appropriate exception. */
            string chunkEtag = response.getHeader("etag");
            if (chunkEtag != info.etag) {
                throw ML::Exception("chunk etag %s not equal to file etag"
                                    " %s: file <%s> has changed during"
                                    " download",
                                    chunkEtag.c_str(), info.etag.c_str(),
                                    object.c_str());
            }
            ExcAssertEqual(response.body().size(), chunkSize);
            Chunk & chunk = chunks[chunkNr];
            chunk.assign(std::move(response.body_));
        }
        catch (const std::exception & exc) {
            excPtrHandler.takeCurrentException();
        }
        activeRqs--;
        ML::futex_wake(activeRqs);
    }

    size_t getChunkSize(unsigned int chunkNbr)
        const
    {
        size_t chunkSize = std::min(baseChunkSize * (1 << (chunkNbr / 2)),
                                    maxChunkSize);
        return chunkSize;
    }

    /* static variables, set during or right after construction */
    shared_ptr<S3Api> owner;
    std::string bucket;
    std::string object;
    S3Api::ObjectInfo info;
    uint64_t offset; /* the lower position in the file from which the download
                      * is started */
    uint64_t downloadSize; /* total number of bytes to download */
    size_t baseChunkSize;
    size_t maxChunkSize;

    bool closed; /* whether close() was invoked */
    ExceptionPtrHandler excPtrHandler;

    /* read thread */
    uint64_t readOffset; /* number of bytes from the entire stream that
                          * have been returned to the caller */
    string readPart; /* data buffer for the part of the stream being
                      * transferred to the caller */
    ssize_t readPartOffset; /* number of bytes from "readPart" that have
                             * been returned to the caller, or -1 when
                             * awaiting a new part */
    unsigned int currentChunk; /* chunk being read */

    /* http requests */
    unsigned int maxRqs; /* maximum number of concurrent http requests */
    uint64_t requestedBytes; /* total number of bytes that have been
                              * requested, including the non-received ones */
    vector<Chunk> chunks; /* chunks */
    unsigned int currentRq;  /* number of done requests */
    atomic<unsigned int> activeRqs; /* number of pending http requests */
};


/****************************************************************************/
/* S3 UPLOADER                                                              */
/****************************************************************************/

inline void touchByte(const char * c)
{
    __asm__(" # [in]":: [in] "r" (*c):);
}

inline void touch(const char * start, size_t size)
{
    const char * current = start - (intptr_t) start % 4096;
    if (current < start) {
        current += 4096;
    }
    const char * end = start + size;
    for (; current < end; current += 4096) {
        touchByte(current);
    }
}

struct S3Uploader {
    S3Uploader(const std::string & urlStr,
               const ML::OnUriHandlerException & excCallback,
               const S3Api::ObjectMetadata & objectMetadata)
        : owner(getS3ApiForUri(urlStr)),
          metadata(objectMetadata),
          onException(excCallback),
          closed(false),
          chunkSize(8 * 1024 * 1024), // start with 8MB and ramp up
          currentRq(0),
          activeRqs(0)
    {
        std::tie(bucket, object) = S3Api::parseUri(urlStr);

        /* Maximum chunk size is what we can do in 3 seconds, up to 1% of
           system memory. */
#if 0
        maxChunkSize = (owner->bandwidthToServiceMbps
                        * 3.0 * 1000000);
        size_t sysMemory = getTotalSystemMemory();
        maxChunkSize = std::min(maxChunkSize, sysMemory / 100);
#else
        maxChunkSize = 64 * 1024 * 1024;
#endif

        try {
            S3Api::MultiPartUpload upload = owner->obtainMultiPartUpload(bucket, "/" + object,
                                                                         metadata,
                                                                         S3Api::UR_EXCLUSIVE);
            uploadId = upload.id;
        }
        catch (...) {
            if (onException) {
                onException();
            }
            throw;
        }
    }

    ~S3Uploader()
    {
        /* We ensure at runtime that "close" is called because it is mandatory
           for the proper cleanup of active requests. Because "close" can
           throw, we cannot however call it from the destructor. */
        if (!closed) {
            cerr << "destroying S3Uploader without invoking close()\n";
            abort();
        }
    }

    std::streamsize write(const char * s, std::streamsize n)
    {
        std::streamsize done(0);

        touch(s, n);

        size_t remaining = chunkSize - current.size();
        while (n > 0) {
            if (excPtrHandler.hasException() && onException) {
                onException();
            }
            excPtrHandler.rethrowIfSet();
            size_t toDo = min(remaining, (size_t) n);
            if (toDo < n) {
                flush();
                remaining = chunkSize - current.size();
            }
            current.append(s, toDo);
            s += toDo;
            n -= toDo;
            done += toDo;
            remaining -= toDo;
        }

        return done;
    }

    void flush(bool force = false)
    {
        if (!force) {
            ExcAssert(current.size() > 0);
        }
        while (activeRqs == metadata.numRequests) {
            ML::futex_wait(activeRqs, activeRqs);
        }
        if (excPtrHandler.hasException() && onException) {
            onException();
        }
        excPtrHandler.rethrowIfSet();

        unsigned int rqNbr(currentRq);
        auto onResponse = [&, rqNbr] (S3Api::Response && response) {
            this->handleResponse(rqNbr, std::move(response));
        };

        unsigned int partNumber = currentRq + 1;
        if (etags.size() < partNumber) {
            etags.resize(partNumber);
        }

        activeRqs++;
        owner->putAsync(onResponse, bucket, "/" + object,
                        ML::format("partNumber=%d&uploadId=%s",
                                   partNumber, uploadId),
                        {}, {}, current);

        if (currentRq % 5 == 0 && chunkSize < maxChunkSize)
            chunkSize *= 2;

        current.clear();
        currentRq = partNumber;
    }

    void handleResponse(unsigned int rqNbr, S3Api::Response && response)
    {
        try {
            if (response.excPtr_) {
                rethrow_exception(response.excPtr_);
            }

            if (response.code_ != 200) {
                cerr << response.bodyXmlStr() << endl;
                throw ML::Exception("put didn't work: %d", (int)response.code_);
            }

            string etag = response.getHeader("etag");
            ExcAssert(etag.size() > 0);
            etags[rqNbr] = etag;
        }
        catch (const std::exception & exc) {
            excPtrHandler.takeCurrentException();
        }
        activeRqs--;
        ML::futex_wake(activeRqs);
    }

    string close()
    {
        closed = true;
        if (current.size() > 0) {
            flush();
        }
        else if (currentRq == 0) {
            /* for empty files, force the creation of a single empty part */
            flush(true);
        }
        while (activeRqs > 0) {
            ML::futex_wait(activeRqs, activeRqs);
        }
        if (excPtrHandler.hasException() && onException) {
            onException();
        }
        excPtrHandler.rethrowIfSet();

        string finalEtag;
        try {
            finalEtag = owner->finishMultiPartUpload(bucket, "/" + object,
                                                     uploadId, etags);
        }
        catch (...) {
            if (onException) {
                onException();
            }
            throw;
        }

        return finalEtag;
    }

private:
    shared_ptr<S3Api> owner;
    std::string bucket;
    std::string object;
    S3Api::ObjectMetadata metadata;
    ML::OnUriHandlerException onException;

    size_t maxChunkSize;
    std::string uploadId;

    /* state variables, used between "start" and "stop" */
    bool closed; /* whether close() was invoked */
    ExceptionPtrHandler excPtrHandler;

    string current; /* current chunk data */
    size_t chunkSize; /* current chunk size */
    std::vector<std::string> etags; /* etags of individual chunks */
    unsigned int currentRq;  /* number of done requests */
    atomic<unsigned int> activeRqs; /* number of pending http requests */
};


struct AtInit {
    AtInit() {
        registerUrlFsHandler("s3", new S3UrlFsHandler());
    }
} atInit;

HttpRequest::Content
makeXmlContent(const tinyxml2::XMLDocument & xmlDocument)
{
    tinyxml2::XMLPrinter printer;
    const_cast<tinyxml2::XMLDocument &>(xmlDocument).Print(&printer);

    return HttpRequest::Content(string(printer.CStr()), "application/xml");
}


/****************************************************************************/
/* S3 REQUEST STATE                                                         */
/****************************************************************************/

struct S3RequestState {
    S3RequestState(const shared_ptr<S3Api::SignedRequest> & rq,
                   const S3Api::OnResponse & onResponse)
        : rq(rq), onResponse(onResponse),
          range(rq->params.downloadRange), retries(0)
    {
    }

    RestParams makeHeaders()
        const
    {
        RestParams headers = rq->params.headers;
        headers.push_back({"Date", rq->params.date});
        headers.push_back({"Authorization", rq->auth});
        if (rq->params.useRange()) {
            headers.push_back({"Range", range.headerValue()});
        }

        return headers;
    }

    int makeTimeout()
        const
    {
        double expectedTimeSeconds
            = (range.size / 1000000.0) / rq->bandwidthToServiceMbps;
        return 15 + std::max<int>(30, expectedTimeSeconds * 6);
    }

    shared_ptr<S3Api::SignedRequest> rq;

    S3Api::OnResponse onResponse;

    string body;
    string requestBody;
    S3Api::Range range;
    int retries;
};

/****************************************************************************/
/* S3 REQUEST CALLBACKS                                                     */
/****************************************************************************/

struct S3RequestCallbacks : public HttpClientCallbacks {
    S3RequestCallbacks(const shared_ptr<S3RequestState> & state)
        : state_(state)
    {
    }

    virtual void onResponseStart(const HttpRequest & rq,
                                 const std::string & httpVersion,
                                 int code);
    virtual void onHeader(const HttpRequest & rq,
                          const char * data, size_t size);
    virtual void onData(const HttpRequest & rq,
                        const char * data, size_t size);
    virtual void onDone(const HttpRequest & rq,
                        HttpClientError errorCode);

    void appendErrorContext(string & message) const;
    void scheduleRestart() const;

    shared_ptr<S3RequestState> state_;

    S3Api::Response response_;
    string header_;
};

void
performStateRequest(const shared_ptr<S3RequestState> & state)
{
    auto & client = getS3Globals().getClient(state->rq->params.bucket);

    const S3Api::RequestParams & params = state->rq->params;
    auto callbacks = make_shared<S3RequestCallbacks>(state);
    RestParams headers = state->makeHeaders();
    int timeout = state->makeTimeout();

    while (!client->enqueueRequest(params.verb, state->rq->resource,
                                   callbacks,
                                   state->rq->params.content,
                                   /* query params already encoded in
                                      resource */
                                   {},
                                   headers,
                                   timeout)) {
        /* TODO: should invoke onResponse with "too many requests" */
        throw ML::Exception("the http client could not enqueue the request");
    }
}

void
S3RequestCallbacks::
onResponseStart(const HttpRequest & rq, const std::string & httpVersion,
                int code)
{
    response_.code_ = code;
}

void
S3RequestCallbacks::
onHeader(const HttpRequest & rq, const char * data, size_t size)
{
    header_.append(data, size);
}

void
S3RequestCallbacks::
onData(const HttpRequest & rq, const char * data, size_t size)
{
    state_->requestBody.append(data, size);
}

void
S3RequestCallbacks::
onDone(const HttpRequest & rq, HttpClientError errorCode)
{
    bool restart(false);
    bool errorCondition(false);
    string message;

    if (errorCode == HttpClientError::None) {
        if (response_.code_ >= 300 && response_.code_ != 404) {
            errorCondition = true;
            message = ("S3 operation failed with HTTP code "
                       + to_string(response_.code_) + "\n");

            /* retry on 50X range errors (recoverable) */
            if (response_.code_ >= 500 and response_.code_ < 505) {
                restart = true;
                message += "Error is recoverable.\n";
            }
            else {
                message += "Error is unrecoverable.\n";
            }
        }
    }
    else {
        restart = true;
        if (state_->rq->params.useRange()) {
            state_->range.adjust(state_->requestBody.size());
        }
        state_->body.append(state_->requestBody);
        state_->requestBody.clear();
        message = ("S3 operation failed with internal error: "
                   + errorMessage(errorCode) + "\n");
    }

    if (restart) {
        if (state_->retries < getS3Globals().numRetries) {
            message += "Will retry operation.\n";
        }
        else {
            errorCondition = true;
            message += "Too many retries.\n";
            restart = false;
        }
    }

    if (message.size() > 0) {
        appendErrorContext(message);
        ::fprintf(stderr, "%s\n", message.c_str());
    }

    if (restart) {
        state_->retries++;
        scheduleRestart();
    }
    else {
        if (errorCondition) {
            response_.excPtr_ = make_exception_ptr(ML::Exception(message));
        }
        else {
            response_.header_.parse(header_, false);
            state_->body.append(state_->requestBody);
            response_.body_ = std::move(state_->body);
        }
        header_.clear();
        state_->requestBody.clear();
        state_->onResponse(std::move(response_));
    }
}

void
S3RequestCallbacks::
appendErrorContext(string & message)
    const
{
    const S3Api::RequestParams & params = state_->rq->params;

    message += params.verb + " " + state_->rq->resource + "\n";
    if (header_.size() > 0) {
        message += "Response headers:\n" + header_;
    }
    if (response_.body_.size() > 0) {
        message += (string("Response body (") + to_string(response_.body_.size())
                    + " bytes):\n" + response_.body_ + "\n");

        /* log so-called "REST error"
           (http://docs.aws.amazon.com/AmazonS3/latest/API/ErrorResponses.html)
        */
        if (header_.find("Content-Type: application/xml")
            != string::npos) {
            unique_ptr<tinyxml2::XMLDocument> localXml(
                new tinyxml2::XMLDocument()
                );
            localXml->Parse(response_.body_.c_str());
            auto element = tinyxml2::XMLHandle(*localXml)
                .FirstChildElement("Error")
                .ToElement();
            if (element) {
                message += ("S3 REST error: ["
                            + extract<string>(element, "Code")
                            + "] message ["
                            + extract<string>(element, "Message")
                            +"]\n");
            }
        }
    }
}

void
S3RequestCallbacks::
scheduleRestart()
    const
{
    S3Globals & globals = getS3Globals();

    // allow a maximum of 384 seconds for retry delays (1 << 7 * 3) 
    int multiplier = (state_->retries < 8
                      ? (1 << state_->retries)
                      : state_->retries << 7);
    double numSeconds = ::random() % (globals.baseRetryDelay
                                      * multiplier);
    if (numSeconds == 0) {
        numSeconds = globals.baseRetryDelay * multiplier;
    }

    numSeconds = 0.05;

    const S3Api::RequestParams & params = state_->rq->params;

    ::fprintf(stderr,
              "S3 operation retry in %f seconds: %s %s\n",
              numSeconds, params.verb.c_str(), params.resource.c_str());

    auto timer = make_shared<PeriodicEventSource>();

    auto state = state_;
    auto onTimeout = [&, timer, state] (uint64_t ticks) {
        S3Globals & globals = getS3Globals();
        performStateRequest(state);
        globals.loop.removeSource(timer.get());
    };
    timer->init(numSeconds, std::move(onTimeout));
    globals.loop.addSource("retry-timer-" + randomString(8), timer);
}

}


namespace Datacratic {

/****************************************************************************/
/* S3 CONFIG DESCRIPTION                                                    */
/****************************************************************************/

S3ConfigDescription::
S3ConfigDescription()
{
    addField("accessKeyId", &S3Config::accessKeyId, "");
    addField("accessKey", &S3Config::accessKey, "");
}

/****************************************************************************/
/* S3 API                                                                   */
/****************************************************************************/

double
S3Api::
defaultBandwidthToServiceMbps = 20.0;

S3Api::Range S3Api::Range::Full(0);

std::string
S3Api::
s3EscapeResource(const std::string & str)
{
    if (str.size() == 0) {
        throw ML::Exception("empty str name");
    }

    if (str[0] != '/') {
        throw ML::Exception("resource name must start with a '/'");
    }

    std::string result;
    for (auto c: str) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/')
            result += c;
        else result += ML::format("%%%02X", c);
    }
    
    return result;
}

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

void
S3Api::
init()
{
    string keyId, key;
    std::tie(keyId, key, std::ignore)
        = getS3CredentialsFromEnvVar();

    if (keyId == "" || key == "") {
        tie(keyId, key, std::ignore, std::ignore, std::ignore)
            = getCloudCredentials();
    }
    if (keyId == "" || key == "")
        throw ML::Exception("Cannot init S3 API with no keys, environment or creedentials file");
    
    this->init(keyId, key);
}

void
S3Api::
perform(const OnResponse & onResponse, const shared_ptr<SignedRequest> & rq)
    const
{
    size_t spacePos = rq->resource.find(" ");
    if (spacePos != string::npos) {
        throw ML::Exception("url '" + rq->resource + "' contains an unescaped"
                            " space at position " + to_string(spacePos));
    }

    auto state = make_shared<S3RequestState>(rq, onResponse);
    performStateRequest(state);
}

S3Api::Response
S3Api::
performSync(const shared_ptr<SignedRequest> & rq) const
{
    SyncResponse syncResponse;
    perform(syncResponse, rq);
    return syncResponse.response();
}

std::string
S3Api::
signature(const RequestParams & request) const
{
    string digest
        = S3Api::getStringToSignV2Multi(request.verb,
                                        request.bucket,
                                        request.resource, request.subResource,
                                        request.content.contentType, request.contentMd5,
                                        request.date, request.headers);
    
    //cerr << "digest = " << digest << endl;
    
    return signV2(digest, accessKey);
}

shared_ptr<S3Api::SignedRequest>
S3Api::
prepare(const RequestParams & request) const
{
    string protocol = defaultProtocol;
    if(protocol.length() == 0){
        throw ML::Exception("attempt to perform s3 request without a "
            "default protocol. (Could be caused by S3Api initialisation with "
            "the empty constructor.)");
    }

    auto sharedResult = make_shared<SignedRequest>();
    SignedRequest & result = *sharedResult;
    result.params = request;
    result.bandwidthToServiceMbps = bandwidthToServiceMbps;

    if (request.resource.find("//") != string::npos)
        throw ML::Exception("attempt to perform s3 request with double slash: "
                            + request.resource);

    result.resource += request.resource;
    if (request.subResource.size() > 0) {
        result.resource += "?" + request.subResource;
    }

    for (unsigned i = 0;  i < request.queryParams.size();  ++i) {
        if (i == 0 && request.subResource == "")
            result.resource += "?";
        else
            result.resource += "&";
        result.resource += (uriEncode(request.queryParams[i].first)
                            + "=" + uriEncode(request.queryParams[i].second));
    }

    string sig = signature(request);
    result.auth = "AWS " + accessKeyId + ":" + sig;

    //cerr << "result.resource = " << result.resource << endl;
    //cerr << "result.auth = " << result.auth << endl;

    return sharedResult;
}

S3Api::Response
S3Api::
headEscaped(const std::string & bucket,
            const std::string & resource,
            const std::string & subResource,
            const RestParams & headers,
            const RestParams & queryParams) const
{
    RequestParams request;
    request.verb = "HEAD";
    request.bucket = bucket;
    request.resource = resource;
    request.subResource = subResource;
    request.headers = headers;
    request.queryParams = queryParams;
    request.date = Date::now().printRfc2616();

    return performSync(prepare(request));
}

S3Api::Response
S3Api::
getEscaped(const std::string & bucket,
           const std::string & resource,
           const Range & downloadRange,
           const std::string & subResource,
           const RestParams & headers,
           const RestParams & queryParams) const
{
    SyncResponse syncResponse;
    getEscapedAsync(syncResponse, bucket, resource, downloadRange,
                    subResource, headers, queryParams);
    return syncResponse.response();
}

void
S3Api::
getEscapedAsync(const S3Api::OnResponse & onResponse,
                const std::string & bucket,
                const std::string & resource,
                const Range & downloadRange,
                const std::string & subResource,
                const RestParams & headers,
                const RestParams & queryParams) const
{
    RequestParams request;
    request.verb = "GET";
    request.bucket = bucket;
    request.resource = resource;
    request.subResource = subResource;
    request.headers = headers;
    request.queryParams = queryParams;
    request.date = Date::now().printRfc2616();
    request.downloadRange = downloadRange;

    perform(onResponse, prepare(request));
}

/** Perform a POST request from end to end. */
S3Api::Response
S3Api::
postEscaped(const std::string & bucket,
            const std::string & resource,
            const std::string & subResource,
            const RestParams & headers,
            const RestParams & queryParams,
            const HttpRequest::Content & content) const
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

    return performSync(prepare(request));
}

S3Api::Response
S3Api::
putEscaped(const std::string & bucket,
           const std::string & resource,
           const std::string & subResource,
           const RestParams & headers,
           const RestParams & queryParams,
           const HttpRequest::Content & content) const
{
    SyncResponse syncResponse;
    putEscapedAsync(syncResponse, bucket, resource, subResource, headers,
                    queryParams, content);
    return syncResponse.response();
}

void
S3Api::
putEscapedAsync(const OnResponse & onResponse,
                const std::string & bucket,
                const std::string & resource,
                const std::string & subResource,
                const RestParams & headers,
                const RestParams & queryParams,
                const HttpRequest::Content & content)
    const
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

    perform(onResponse, prepare(request));
}

S3Api::Response
S3Api::
eraseEscaped(const std::string & bucket,
             const std::string & resource,
             const std::string & subResource,
             const RestParams & headers,
             const RestParams & queryParams) const
{
    RequestParams request;
    request.verb = "DELETE";
    request.bucket = bucket;
    request.resource = resource;
    request.subResource = subResource;
    request.headers = headers;
    request.queryParams = queryParams;
    request.date = Date::now().printRfc2616();

    return performSync(prepare(request));
}

RestParams
S3Api::ObjectMetadata::
getRequestHeaders() const
{
    RestParams result;
    Redundancy redundancy = this->redundancy;

    if (redundancy == REDUNDANCY_DEFAULT)
        redundancy = defaultRedundancy;

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
    if (acl != "")
        result.push_back({"x-amz-acl", acl});
    for (auto md: metadata) {
        result.push_back({"x-amz-meta-" + md.first, md.second});
    }
    return result;
}

pair<bool,string>
S3Api::isMultiPartUploadInProgress(
    const std::string & bucket,
    const std::string & resource) const
{
    // Contains the resource without the leading slash
    string outputPrefix(resource, 1);

    // Check if there is already a multipart upload in progress
    auto inProgressReq = get(bucket, "/", Range::Full, "uploads", {},
                             { { "prefix", outputPrefix } });

    //cerr << inProgressReq.bodyXmlStr() << endl;

    auto inProgress = inProgressReq.bodyXml();

    using namespace tinyxml2;

    XMLHandle handle(*inProgress);

    auto upload
        = handle
        .FirstChildElement("ListMultipartUploadsResult")
        .FirstChildElement("Upload")
        .ToElement();

    string uploadId;
    vector<MultiPartUploadPart> parts;


    for (; upload; upload = upload->NextSiblingElement("Upload")) 
    {
        XMLHandle uploadHandle(upload);

        auto key = extract<string>(upload, "Key");

        if (key != outputPrefix)
            continue;

        // Already an upload in progress
        string uploadId = extract<string>(upload, "UploadId");

        return make_pair(true,uploadId);
    }
    return make_pair(false,"");
}

S3Api::MultiPartUpload
S3Api::
obtainMultiPartUpload(const std::string & bucket,
                      const std::string & resource,
                      const ObjectMetadata & metadata,
                      UploadRequirements requirements) const
{
    string escapedResource = s3EscapeResource(resource);
    // Contains the resource without the leading slash
    string outputPrefix(resource, 1);

    string uploadId;
    vector<MultiPartUploadPart> parts;

    if (requirements != UR_FRESH) {

        // Check if there is already a multipart upload in progress
        auto inProgressReq = get(bucket, "/", Range::Full, "uploads", {},
                                 { { "prefix", outputPrefix } });

        //cerr << "in progress requests:" << endl;
        //cerr << inProgressReq.bodyXmlStr() << endl;

        auto inProgress = inProgressReq.bodyXml();

        using namespace tinyxml2;

        XMLHandle handle(*inProgress);

        auto upload
            = handle
            .FirstChildElement("ListMultipartUploadsResult")
            .FirstChildElement("Upload")
            .ToElement();

        // uint64_t partSize = 0;
        uint64_t currentOffset = 0;

        for (; upload; upload = upload->NextSiblingElement("Upload")) {
            XMLHandle uploadHandle(upload);

            auto key = extract<string>(upload, "Key");

            if (key != outputPrefix)
                continue;
        
            // Already an upload in progress
            string uploadId = extract<string>(upload, "UploadId");

            // From here onwards is only useful if we want to continue a half-finished
            // upload.  Instead, we will delete it to avoid problems with creating
            // half-finished files when we don't know what we're doing.

            auto deletedInfo = eraseEscaped(bucket, escapedResource,
                                            "uploadId=" + uploadId);

            continue;

            // TODO: check metadata, etc
            auto inProgressInfo = getEscaped(bucket, escapedResource, Range::Full,
                                             "uploadId=" + uploadId)
                .bodyXml();

            inProgressInfo->Print();

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

            // partSize = biggestPartSize;

            //cerr << "numPartsDone = " << numPartsDone << endl;
            //cerr << "currentOffset = " << currentOffset
            //     << "dataSize = " << dataSize << endl;
        }
    }

    if (uploadId.empty()) {
        //cerr << "getting new ID" << endl;

        RestParams headers = metadata.getRequestHeaders();
        auto result = postEscaped(bucket, escapedResource,
                                  "uploads", headers).bodyXml();
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
    ExcAssert(etags.size() > 0);

    XMLDocument joinRequest;
    auto r = joinRequest.InsertFirstChild(joinRequest.NewElement("CompleteMultipartUpload"));
    for (unsigned i = 0;  i < etags.size();  ++i) {
        auto n = r->InsertEndChild(joinRequest.NewElement("Part"));
        n->InsertEndChild(joinRequest.NewElement("PartNumber"))
            ->InsertEndChild(joinRequest.NewText(ML::format("%d", i + 1).c_str()));
        n->InsertEndChild(joinRequest.NewElement("ETag"))
            ->InsertEndChild(joinRequest.NewText(etags[i].c_str()));
    }

    //joinRequest.Print();

    string escapedResource = s3EscapeResource(resource);

    auto joinResponse
        = postEscaped(bucket, escapedResource, "uploadId=" + uploadId,
                      {}, {}, makeXmlContent(joinRequest));

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
       const std::string & uri,
       CheckMethod check,
       ObjectMetadata metadata,
       int numInParallel)
{
    //cerr << "need to upload " << dataSize << " bytes" << endl;

    // Check if it's already there

    if (check == CM_SIZE || check == CM_MD5_ETAG) {
        string bucket, resource;
        std::tie(bucket, resource) = parseUri(uri);

        auto info = tryGetObjectInfo(bucket, resource);
        if (info.size == dataSize) {
            //cerr << "already uploaded" << endl;
            return info.etag;
        }
    }

    if (numInParallel != -1) {
        metadata.numRequests = numInParallel;
    }
    S3Uploader uploader(uri, nullptr, metadata);

    /* The size of the slices we pass as argument to S3Uploader::write.
       Internally, S3Uploader will uses its own chunk size when performing the
       actual upload requests. */
    size_t partSize = 5 * 1024 * 1024;
    for (size_t i = 0; i < dataSize;) {
        size_t remaining = dataSize - i;
        size_t currentSize = std::min(partSize, remaining);
        uploader.write(data + i, currentSize);
        i += currentSize;
    }

    return uploader.close();
}

std::string
S3Api::
upload(const char * data,
       size_t dataSize,
       const std::string & bucket,
       const std::string & resource,
       CheckMethod check,
       ObjectMetadata metadata,
       int numInParallel)
{
    string urlStr("s3://" + bucket + resource);
    return upload(data, dataSize, urlStr, check, move(metadata),
                  numInParallel);
}

S3Api::ObjectInfo::
ObjectInfo(tinyxml2::XMLNode * element)
{
    size = extract<uint64_t>(element, "Size");
    key  = extract<string>(element, "Key");
    string lastModifiedStr = extract<string>(element, "LastModified");
    lastModified = Date::parseIso8601DateTime(lastModifiedStr);
    etag = extract<string>(element, "ETag");
    ownerId = extract<string>(element, "Owner/ID");
    ownerName = extractDef<string>(element, "Owner/DisplayName", "");
    storageClass = extract<string>(element, "StorageClass");
    exists = true;
}

S3Api::ObjectInfo::
ObjectInfo(const S3Api::Response & response)
{
    exists = true;
    lastModified = Date::parse(response.getHeader("last-modified"),
            "%a, %e %b %Y %H:%M:%S %Z");
    size = response.header_.contentLength;
    etag = response.getHeader("etag");
    storageClass = ""; // Not available in headers
    ownerId = "";      // Not available in headers
    ownerName = "";    // Not available in headers
}


void
S3Api::
forEachObject(const std::string & bucket,
              const std::string & prefix,
              const OnObject & onObject,
              const OnSubdir & onSubdir,
              const std::string & delimiter,
              int depth,
              const std::string & startAt) const
{
    using namespace tinyxml2;

    string marker = startAt;
    // bool firstIter = true;
    do {
        //cerr << "Starting at " << marker << endl;
        
        RestParams queryParams;
        if (prefix != "")
            queryParams.push_back({"prefix", prefix});
        if (delimiter != "")
            queryParams.push_back({"delimiter", delimiter});
        if (marker != "")
            queryParams.push_back({"marker", marker});

        auto listingResult = get(bucket, "/", Range::Full, "",
                                 {}, queryParams);
        auto listingResultXml = listingResult.bodyXml();

        //listingResultXml->Print();

        string foundPrefix
            = extractDef<string>(listingResult, "ListBucketResult/Prefix", "");
        string truncated
            = extract<string>(listingResult, "ListBucketResult/IsTruncated");
        bool isTruncated = truncated == "true";
        marker = "";

        auto foundObject
            = XMLHandle(*listingResultXml)
            .FirstChildElement("ListBucketResult")
            .FirstChildElement("Contents")
            .ToElement();

        bool stop = false;

        for (int i = 0; onObject && foundObject;
             foundObject = foundObject->NextSiblingElement("Contents"), ++i) {
            ObjectInfo info(foundObject);

            string key = info.key;
            ExcAssertNotEqual(key, marker);
            marker = key;

            ExcAssertEqual(info.key.find(foundPrefix), 0);
            // cerr << "info.key: " + info.key + "; foundPrefix: " +foundPrefix + "\n";
            string basename(info.key, foundPrefix.length());

            if (!onObject(foundPrefix, basename, info, depth)) {
                stop = true;
                break;
            }
        }

        if (stop) return;

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

        // firstIter = false;
        if (!isTruncated)
            break;
    } while (marker != "");

    //cerr << "done scanning" << endl;
}

void
S3Api::
forEachObject(const std::string & uriPrefix,
              const OnObjectUri & onObject,
              const OnSubdir & onSubdir,
              const std::string & delimiter,
              int depth,
              const std::string & startAt) const
{
    string bucket, objectPrefix;
    std::tie(bucket, objectPrefix) = parseUri(uriPrefix);

    auto onObject2 = [&] (const std::string & prefix,
                          const std::string & objectName,
                          const ObjectInfo & info,
                          int depth)
        {
            string uri = "s3://" + bucket + "/" + prefix;
            if (objectName.size() > 0) {
                uri += objectName;
            }
            return onObject(uri, info, depth);
        };

    forEachObject(bucket, objectPrefix, onObject2, onSubdir, delimiter, depth, startAt);
}

S3Api::ObjectInfo
S3Api::
getObjectInfo(const std::string & bucket, const std::string & object,
              S3ObjectInfoTypes infos)
    const
{
    return ((infos & int(S3ObjectInfoTypes::FULL_EXTRAS)) != 0
            ? getObjectInfoFull(bucket, object)
            : getObjectInfoShort(bucket, object));
}

S3Api::ObjectInfo
S3Api::
getObjectInfoFull(const std::string & bucket, const std::string & object)
    const
{
    RestParams queryParams;
    queryParams.push_back({"prefix", object});

    auto listingResult = getEscaped(bucket, "/", Range::Full, "", {}, queryParams);

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
getObjectInfoShort(const std::string & bucket, const std::string & object)
    const
{
    auto res = head(bucket, "/" + object);
    if (res.code_ == 404) {
        throw ML::Exception("object " + object + " not found in bucket "
                            + bucket);
    }
    if (res.code_ != 200) {
        throw ML::Exception("error getting object");
    }
    return ObjectInfo(res);
}

S3Api::ObjectInfo
S3Api::
tryGetObjectInfo(const std::string & bucket,
                 const std::string & object,
                 S3ObjectInfoTypes infos)
    const
{
    return ((infos & int(S3ObjectInfoTypes::FULL_EXTRAS)) != 0
            ? tryGetObjectInfoFull(bucket, object)
            : tryGetObjectInfoShort(bucket, object));
}

S3Api::ObjectInfo
S3Api::
tryGetObjectInfoFull(const std::string & bucket, const std::string & object)
    const
{
    RestParams queryParams;
    queryParams.push_back({"prefix", object});

    auto listingResult = get(bucket, "/", Range::Full, "", {}, queryParams);
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

    if (info.key != object) {
        return ObjectInfo();
    }

    return info;
}

S3Api::ObjectInfo
S3Api::
tryGetObjectInfoShort(const std::string & bucket, const std::string & object)
    const
{
    auto res = head(bucket, "/" + object);
    if (res.code_ == 404) {
        return ObjectInfo();
    }
    if (res.code_ != 200) {
        throw ML::Exception("error getting object");
    }

    return ObjectInfo(res);
}

S3Api::ObjectInfo
S3Api::
getObjectInfo(const std::string & uri, S3ObjectInfoTypes infos)
    const
{
    string bucket, object;
    std::tie(bucket, object) = parseUri(uri);
    return getObjectInfo(bucket, object, infos);
}

S3Api::ObjectInfo
S3Api::
tryGetObjectInfo(const std::string & uri, S3ObjectInfoTypes infos)
    const
{
    string bucket, object;
    std::tie(bucket, object) = parseUri(uri);
    return tryGetObjectInfo(bucket, object, infos);
}

void
S3Api::
eraseObject(const std::string & bucket,
            const std::string & object)
{
    Response response = erase(bucket, object);

    if (response.code_ != 204) {
        cerr << response.bodyXmlStr() << endl;
        throw ML::Exception("error erasing object request: %d",
                            response.code_);
    }
}

bool
S3Api::
tryEraseObject(const std::string & bucket,
               const std::string & object)
{
    Response response = erase(bucket, object);
    
    if (response.code_ != 200) {
        return false;
    }

    return true;
}

void
S3Api::
eraseObject(const std::string & uri)
{
    string bucket, object;
    std::tie(bucket, object) = parseUri(uri);
    eraseObject(bucket, object);
}

bool
S3Api::
tryEraseObject(const std::string & uri)
{
    string bucket, object;
    std::tie(bucket, object) = parseUri(uri);
    return tryEraseObject(bucket, object);
}

std::string
S3Api::
getPublicUri(const std::string & uri,
             const std::string & protocol)
{
    string bucket, object;
    std::tie(bucket, object) = parseUri(uri);
    return getPublicUri(bucket, object, protocol);
}

std::string
S3Api::
getPublicUri(const std::string & bucket,
             const std::string & object,
             const std::string & protocol)
{
    return protocol + "://" + bucket + ".s3.amazonaws.com/" + object;
}

void
S3Api::
download(const std::string & uri,
         const OnChunk & onChunk,
         ssize_t startOffset,
         ssize_t endOffset) const
{
    S3Downloader downloader(uri, startOffset, endOffset);
    uint64_t downloadSize = downloader.getDownloadSize();
    uint64_t downloaded(0);
    int chunkIndex(0);
    while (!downloader.endOfDownload()) {
        char buffer[128 * 1024 * 1024];
        streamsize chunkSize = downloader.read(buffer, sizeof(buffer));
        onChunk(buffer, chunkSize, chunkIndex, downloaded, downloadSize);
        downloaded += chunkSize;
        chunkIndex++;
    }
    downloader.close();
}

void
S3Api::
download(const std::string & bucket, const std::string & object,
         const OnChunk & onChunk,
         ssize_t startOffset,
         ssize_t endOffset) const
{
    string urlStr("s3://" + bucket + "/" + object);
    download(urlStr, onChunk, startOffset, endOffset);
}

/**
 * Downloads a file from s3 to a local file. If the maxSize is specified, only
 * the first maxSize bytes will be downloaded.
 */
void
S3Api::
downloadToFile(const std::string & uri, const std::string & outfile,
               ssize_t endOffset)
    const
{
    ofstream myFile;
    myFile.open(outfile.c_str());

    auto onChunk = [&] (const char * data, size_t size,
                        int chunkIndex,
                        uint64_t offset, uint64_t totalSize) {
        ExcAssertLessEqual(offset + size, totalSize);
        myFile.seekp(offset);
        myFile.write(data, size);
    };
    download(uri, onChunk, 0, endOffset);
    myFile.close();
}


/****************************************************************************/
/* EXCEPTIONPTR HANDLER                                                     */
/****************************************************************************/

bool
ExceptionPtrHandler::
hasException()
{
    std::unique_lock<mutex> guard(excLock);
    return bool(excPtr);
}

void
ExceptionPtrHandler::
takeException(std::exception_ptr newPtr)
{
    std::unique_lock<mutex> guard(excLock);
    excPtr = newPtr;
}

void
ExceptionPtrHandler::
takeCurrentException()
{
    takeException(std::current_exception());
}

void
ExceptionPtrHandler::
rethrowIfSet()
{
    std::unique_lock<mutex> guard(excLock);
    if (excPtr) {
        std::exception_ptr ptr = excPtr;
        excPtr = nullptr;
        std::rethrow_exception(ptr);
    }
}


/****************************************************************************/
/* STREAMING DOWNLOAD SOURCE                                                */
/****************************************************************************/

struct StreamingDownloadSource {
    StreamingDownloadSource(const std::string & urlStr)
    {
        downloader.reset(new S3Downloader(urlStr));
    }

    typedef char char_type;
    struct category
        : //input_seekable,
        boost::iostreams::input,
        boost::iostreams::device_tag,
        boost::iostreams::closable_tag
    { };

    std::streamsize read(char_type * s, std::streamsize n)
    {
        return downloader->read(s, n);
    }

    bool is_open() const
    {
        return !!downloader;
    }

    void close()
    {
        downloader->close();
        downloader.reset();
    }

private:
    std::shared_ptr<S3Downloader> downloader;
};

std::unique_ptr<std::streambuf>
makeStreamingDownload(const std::string & uri)
{
    std::unique_ptr<std::streambuf> result;
    result.reset(new boost::iostreams::stream_buffer<StreamingDownloadSource>
                 (StreamingDownloadSource(uri),
                  131072));
    return result;
}

std::unique_ptr<std::streambuf>
makeStreamingDownload(const std::string & bucket,
                      const std::string & object)
{
    return makeStreamingDownload("s3://" + bucket + "/" + object);
}


/****************************************************************************/
/* STREAMING UPLOAD SOURCE                                                  */
/****************************************************************************/

struct StreamingUploadSource {

    StreamingUploadSource(const std::string & urlStr,
                          const ML::OnUriHandlerException & excCallback,
                          const S3Api::ObjectMetadata & metadata)
    {
        uploader.reset(new S3Uploader(urlStr, excCallback, metadata));
    }

    typedef char char_type;
    struct category
        : public boost::iostreams::output,
          public boost::iostreams::device_tag,
          public boost::iostreams::closable_tag
    {
    };

    std::streamsize write(const char_type* s, std::streamsize n)
    {
        return uploader->write(s, n);
    }

    bool is_open() const
    {
        return !!uploader;
    }

    void close()
    {
        uploader->close();
        uploader.reset();
    }

private:
    std::shared_ptr<S3Uploader> uploader;
};

std::unique_ptr<std::streambuf>
makeStreamingUpload(const std::string & uri,
                    const ML::OnUriHandlerException & onException,
                    const S3Api::ObjectMetadata & metadata)
{
    std::unique_ptr<std::streambuf> result;
    result.reset(new boost::iostreams::stream_buffer<StreamingUploadSource>
                 (StreamingUploadSource(uri, onException, metadata),
                  131072));
    return result;
}

std::unique_ptr<std::streambuf>
makeStreamingUpload(const std::string & bucket,
                    const std::string & object,
                    const ML::OnUriHandlerException & onException,
                    const S3Api::ObjectMetadata & metadata)
{
    return makeStreamingUpload("s3://" + bucket + "/" + object,
                               onException, metadata);
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

bool
S3Api::
forEachBucket(const OnBucket & onBucket) const
{
    using namespace tinyxml2;

    //cerr << "forEachObject under " << prefix << endl;

    auto listingResult = get("", "/", Range::Full, "");
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

S3Api::Redundancy S3Api::defaultRedundancy = S3Api::REDUNDANCY_STANDARD;

void
S3Api::
setDefaultRedundancy(Redundancy redundancy)
{
    if (redundancy == REDUNDANCY_DEFAULT)
        throw ML::Exception("Can't set default redundancy as default");
    defaultRedundancy = redundancy;
}

S3Api::Redundancy
S3Api::
getDefaultRedundancy()
{
    return defaultRedundancy;
}

namespace {

struct S3BucketInfo {
    std::string s3Bucket;
    std::shared_ptr<S3Api> api;  //< Used to access this uri
};

std::recursive_mutex s3BucketsLock;
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
    std::unique_lock<std::recursive_mutex> guard(s3BucketsLock);

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
            return;
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
    info.api->getEscaped("", "/" + bucketName + "/",
                         S3Api::Range::Full); //throws if !accessible
    s3Buckets[bucketName] = info;

    if (accessKeyId.size() > 0 && accessKey.size() > 0) {
        registerAwsCredentials(accessKeyId, accessKey);
    }
}

/** Register S3 with the filter streams API so that a filter_stream can be used to
    treat an S3 object as a simple stream.
*/
struct RegisterS3Handler {
    static std::pair<std::streambuf *, bool>
    getS3Handler(const std::string & scheme,
                 const std::string & resource,
                 std::ios_base::open_mode mode,
                 const std::map<std::string, std::string> & options,
                 const ML::OnUriHandlerException & onException)
    {
        string::size_type pos = resource.find('/');
        if (pos == string::npos)
            throw ML::Exception("unable to find s3 bucket name in resource "
                                + resource);
        string bucket(resource, 0, pos);

        if (mode == ios::in) {
            return make_pair(makeStreamingDownload("s3://" + resource)
                             .release(),
                             true);
        }
        else if (mode == ios::out) {

            S3Api::ObjectMetadata md;
            for (auto & opt: options) {
                string name = opt.first;
                string value = opt.second;
                if (name == "redundancy" || name == "aws-redundancy") {
                    if (value == "STANDARD")
                        md.redundancy = S3Api::REDUNDANCY_STANDARD;
                    else if (value == "REDUCED")
                        md.redundancy = S3Api::REDUNDANCY_REDUCED;
                    else throw ML::Exception("unknown redundancy value " + value
                                             + " writing S3 object " + resource);
                }
                else if (name == "contentType" || name == "aws-contentType") {
                    md.contentType = value;
                }
                else if (name == "contentEncoding" || name == "aws-contentEncoding") {
                    md.contentEncoding = value;
                }
                else if (name == "acl" || name == "aws-acl") {
                    md.acl = value;
                }
                else if (name == "mode" || name == "compression"
                         || name == "compressionLevel") {
                    // do nothing
                }
                else if (name.find("aws-") == 0) {
                    throw ML::Exception("unknown aws option " + name + "=" + value
                                        + " opening S3 object " + resource);
                }
                else if(name == "num-threads")
                {
                    cerr << ("warning: use of obsolete 'num-threads' option"
                             " key\n");
                    md.numRequests = std::stoi(value);
                }
                else if(name == "num-requests")
                {
                    md.numRequests = std::stoi(value);
                }
                else {
                    cerr << "warning: skipping unknown S3 option "
                         << name << "=" << value << endl;
                }
            }

            return make_pair(makeStreamingUpload("s3://" + resource,
                                                 onException, md)
                             .release(),
                             true);
        }
        else throw ML::Exception("no way to create s3 handler for non in/out");
    }

    void registerBuckets()
    {
    }

    RegisterS3Handler()
    {
        ML::registerUriHandler("s3", getS3Handler);
    }

} registerS3Handler;

bool defaultBucketsRegistered = false;
std::mutex registerBucketsMutex;

tuple<string, string, string, string, string> getCloudCredentials()
{
    string filename = "";
    char* home;
    home = getenv("HOME");
    if (home != NULL)
        filename = home + string("/.cloud_credentials");
    if (filename != "" && ML::fileExists(filename)) {
        std::ifstream stream(filename.c_str());
        while (stream) {
            string line;

            getline(stream, line);
            if (line.empty() || line[0] == '#')
                continue;
            if (line.find("s3") != 0)
                continue;

            vector<string> fields = ML::split(line, '\t');

            if (fields[0] != "s3")
                continue;

            if (fields.size() < 4) {
                cerr << "warning: skipping invalid line in ~/.cloud_credentials: "
                     << line << endl;
                continue;
            }
                
            fields.resize(7);

            string version = fields[1];
            if (version != "1") {
                cerr << "warning: ignoring unknown version "
                     << version <<  " in ~/.cloud_credentials: "
                     << line << endl;
                continue;
            }
                
            string keyId = fields[2];
            string key = fields[3];
            string bandwidth = fields[4];
            string protocol = fields[5];
            string serviceUri = fields[6];

            return make_tuple(keyId, key, bandwidth, protocol, serviceUri);
        }
    }
    return make_tuple("", "", "", "", "");
}

std::string getEnv(const char * varName)
{
    const char * val = getenv(varName);
    return val ? val : "";
}

tuple<string, string, std::vector<std::string> >
getS3CredentialsFromEnvVar()
{
    return make_tuple(getEnv("S3_KEY_ID"), getEnv("S3_KEY"),
                      ML::split(getEnv("S3_BUCKETS"), ','));
}

/** Parse the ~/.cloud_credentials file and add those buckets in.

    The format of that file is as follows:
    1.  One entry per line
    2.  Tab separated
    3.  Comments are '#' in the first position
    4.  First entry is the name of the URI scheme (here, s3)
    5.  Second entry is the "version" of the configuration (here, 1)
        for forward compatibility
    6.  The rest of the entries depend upon the scheme; for s3 they are
        tab-separated and include the following:
        - Access key ID
        - Access key
        - Bandwidth from this machine to the server (MBPS)
        - Protocol (http)
        - S3 machine host name (s3.amazonaws.com)

    If S3_KEY_ID and S3_KEY environment variables are specified,
    they will be used first.
*/
void registerDefaultBuckets()
{
    if (defaultBucketsRegistered)
        return;

    std::unique_lock<std::mutex> guard(registerBucketsMutex);
    defaultBucketsRegistered = true;

    tuple<string, string, string, string, string> cloudCredentials = 
        getCloudCredentials();
    if (get<0>(cloudCredentials) != "") {
        string keyId      = get<0>(cloudCredentials);
        string key        = get<1>(cloudCredentials);
        string bandwidth  = get<2>(cloudCredentials);
        string protocol   = get<3>(cloudCredentials);
        string serviceUri = get<4>(cloudCredentials);

        if (protocol == "")
            protocol = "http";
        if (bandwidth == "")
            bandwidth = "20.0";
        if (serviceUri == "")
            serviceUri = "s3.amazonaws.com";

        registerS3Buckets(keyId, key, boost::lexical_cast<double>(bandwidth),
                          protocol, serviceUri);
        return;
    }
    string keyId;
    string key;
    vector<string> buckets;

    std::tie(keyId, key, buckets) = getS3CredentialsFromEnvVar();
    if (keyId != "" && key != "") {
        if (buckets.empty()) {
            registerS3Buckets(keyId, key);
        }
        else {
            for (string bucket: buckets)
                registerS3Bucket(bucket, keyId, key);
        }
    }
    else
        cerr << "WARNING: registerDefaultBuckets needs either a "
            ".cloud_credentials or S3_KEY_ID and S3_KEY environment "
            " variables" << endl;

#if 0
    char* configFilenameCStr = getenv("CONFIG");
    string configFilename = (configFilenameCStr == NULL ?
                                string() :
                                string(configFilenameCStr));

    if(configFilename != "")
    {
        ML::File_Read_Buffer buf(configFilename);
        Json::Value config = Json::parse(string(buf.start(), buf.end()));
        if(config.isMember("s3"))
        {
            registerS3Buckets(
                config["s3"]["accessKeyId"].asString(),
                config["s3"]["accessKey"].asString(),
                20.,
                "http",
                "s3.amazonaws.com");
            return;
        }
    }
    cerr << "WARNING: registerDefaultBuckets needs either a .cloud_credentials"
            " file or an environment variable CONFIG pointing toward a file "
            "having keys s3.accessKey and s3.accessKeyId" << endl;
#endif
}

void registerS3Buckets(const std::string & accessKeyId,
                       const std::string & accessKey,
                       double bandwidthToServiceMbps,
                       const std::string & protocol,
                       const std::string & serviceUri)
{
    std::unique_lock<std::recursive_mutex> guard(s3BucketsLock);
    int bucketCount(0);
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
            bucketCount++;

            return true;
        };

    api->forEachBucket(onBucket);

    if (bucketCount == 0) {
        cerr << "registerS3Buckets: no bucket registered\n";
    }
}

std::shared_ptr<S3Api> getS3ApiForBucket(const std::string & bucketName)
{
    std::unique_lock<std::recursive_mutex> guard(s3BucketsLock);
    auto it = s3Buckets.find(bucketName);
    if (it == s3Buckets.end()) {
        // On demand, load up the configuration file before we fail
        registerDefaultBuckets();
        it = s3Buckets.find(bucketName);
    }
    if (it == s3Buckets.end()) {
        throw ML::Exception("unregistered s3 bucket " + bucketName);
    }
    return it->second.api;
}

std::shared_ptr<S3Api> getS3ApiForUri(const std::string & uri)
{
    Url url(uri);

    string bucketName = url.host();
    string accessKeyId = url.username();
    if (accessKeyId.empty()) {
        return getS3ApiForBucket(bucketName);
    }

    string accessKey = url.password();
    if (accessKey.empty()) {
        accessKey = getAwsAccessKey(accessKeyId);
    }

    auto api = make_shared<S3Api>(accessKeyId, accessKey);
    api->getEscaped("", "/" + bucketName + "/",
                    S3Api::Range::Full); //throws if !accessible

    return api;
}


} // namespace Datacratic
