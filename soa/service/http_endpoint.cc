/* http_endpoint.cc
   Jeremy Barnes, 18 February 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

*/

#include "soa/service/http_endpoint.h"
#include "jml/arch/cmp_xchg.h"
#include "jml/arch/atomic_ops.h"
#include "jml/utils/parse_context.h"
#include "jml/utils/string_functions.h"
#include "jml/utils/exc_assert.h"
#include <fstream>
#include <boost/make_shared.hpp>


using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* HTTP CONNECTION HANDLER                                                   */
/*****************************************************************************/

HttpConnectionHandler::
HttpConnectionHandler()
    : readState(INVALID), httpEndpoint(0)
{
}

void
HttpConnectionHandler::
onGotTransport()
{
    this->httpEndpoint = dynamic_cast<HttpEndpoint *>(get_endpoint());
    
    readState = HEADER;
    startReading();
}

std::shared_ptr<ConnectionHandler>
HttpConnectionHandler::
makeNewHandlerShared()
{
    if (!this->httpEndpoint)
        throw Exception("HttpConnectionHandler needs to be owned by an "
                        "HttpEndpoint for makeNewHandlerShared() to work");
    return httpEndpoint->makeNewHandler();
}

void
HttpConnectionHandler::
handleData(const std::string & data)
{
   //cerr << "HttpConnectionHandler::handleData: got data <" << data << ">" << endl;
    //httpData.write(data.c_str(), data.length());

    if (headerText == "" && readState == HEADER)
        firstData = Date::now();

    addActivity("handleData with state %d", readState);

#if 0
    string dataSample;
    dataSample.reserve(data.size() * 3 / 2);
    for (unsigned i = 0;  i < 300 && i < data.size();  ++i) {
        if (data[i] == '\n') dataSample += "\\n";
        else if (data[i] == '\r') dataSample += "\\r";
        else if (data[i] == '\0') dataSample += "\\0";
        else if (data[i] == '\t') dataSample += "\\t";
        else if (data[i] < ' ' || data[i] >= 127) dataSample += '.';
        else dataSample += data[i];
    }

    addActivity("got %d bytes of data: %s", (int)data.size(),
                dataSample.c_str());
#endif

    if (readState == PAYLOAD
        || readState == CHUNK_HEADER || readState == CHUNK_BODY) {
        handleHttpData(data);
        return;
    }
    
    if (readState != HEADER) {
        throw Exception("invalid read state %d handling data '%s' for %08xp",
                        readState, data.c_str(), this);
    }
    
    headerText += data;
    // The first time that we see that character sequence, it's the break
    // between the header and the data.
    string::size_type breakPos = headerText.find("\r\n\r\n");
    

    if (breakPos == string::npos)
    {
        if (headerText.size() > 16384) {
            throw ML::Exception("HTTP header exceeds 16kb");
        }
        //cerr << "did not find break pos " << endl;
        return;
    }
    // We got a header
    try {
        header.parse(headerText);
    } catch (...) {
        cerr << "problem parsing in state: " << status() << endl;
        throw;
    }

    //cerr << "content length = " << header.contentLength << endl;

    if (header.contentLength == -1 && !header.isChunked)
        header.contentLength = 0;
    //doError("we need a Content-Length");
    
    addActivityS("header parsing OK");

    //cerr << "done header" << endl;

    handleHttpHeader(header);

    payload = "";

    if (header.isChunked)
        readState = CHUNK_HEADER;
    else readState = PAYLOAD;

    handleHttpData(header.knownData);
}

void
HttpConnectionHandler::
handleHttpHeader(const HttpHeader & header)
{
    // If the client expects a 100 continue, then oblige
    std::string expect = header.tryGetHeader("expect");
    if (!expect.empty()) {
        expect = lowercase(expect);

        if (expect == "100-continue") {

            send("HTTP/1.1 100 Continue\r\n\r\n",
                 NEXT_CONTINUE);
        }
        else {
            HttpResponse response(417, "", "");
            putResponseOnWire(response);
            doError("unknown expectation");
        }
    }

    //cerr << "GOT HTTP HEADER[" << header << "]" << endl;
}

void
HttpConnectionHandler::
handleHttpData(const std::string & data)
{
    //static const char *fName = "HttpConnectionHandler::handleHttpData:";
    //cerr << "got HTTP data in state " << readState << " with "
    //     << data.length() << " characters" << endl;
    //cerr << "data = [" << data << "]" << endl;
    //cerr << endl << endl << "---------------------------" << endl;
    
    if (readState == PAYLOAD) {
        //cerr << "handleHttpData" << endl;
        if (readState != PAYLOAD)
            throw Exception("invalid state: expected payload");

        payload += data;
#if 0
        cerr << "payload = " << payload << endl;
        cerr << "payload.length() = " << payload.length() << endl;
        cerr << "header.contentLength = " << header.contentLength << endl;
#endif
        if (payload.length() > header.contentLength) {
            doError("extra data");
        }
    

        if (payload.length() == header.contentLength) {
            addActivityS("got HTTP payload");
            handleHttpPayload(header, payload);

            //cerr << this << " switching to DONE" << endl;

            readState = DONE;
        }
    }
    if (readState == CHUNK_HEADER || readState == CHUNK_BODY) {
        const char * current = data.c_str();
        const char * end = current + data.length();

        //cerr << "processing " << data.length() << " characters" << endl;

        while (current != end) {
            if (current >= end)
                throw ML::Exception("current >= end");

            //cerr << (end - current) << " chars left; state "
            //     << readState << " chunkHeader = " << chunkHeader << endl;

            if (readState == CHUNK_HEADER) {
                while (current != end) {
                    char c = *current++;
                    chunkHeader += c;
                    if (c == '\n') break;
                }

                //cerr << "chunkHeader now '" << chunkHeader << "'" << endl;

                // Remove an extra cr/lf if there is one
                if (chunkHeader == "\r\n") {
                    chunkHeader = "";
                    continue;
                }

                //if (current == end) break;

                //cerr << "chunkHeader now '" << chunkHeader << "'" << endl;

                string::size_type pos = chunkHeader.find("\r\n");
                if (pos == string::npos) break;
                current += pos + 2 - chunkHeader.length();  // skip crlf
                
                chunkHeader.resize(pos);

                // We got to the end of the chunk header... parse it
                string::size_type lengthPos = chunkHeader.find(';');

                //cerr << "chunkHeader = " << chunkHeader << endl;
                //cerr << "lengthPos = " << lengthPos << endl;

                string lengthStr(chunkHeader, 0, lengthPos);
                //(lengthPos == string::npos
                //                  ? chunkHeader.length() : lengthPos));
                //cerr << "lengthStr = " << lengthStr << endl;

                char * endPtr = 0;
                chunkSize = strtol(lengthStr.c_str(), &endPtr, 16);

                //cerr << "chunkSize = " << chunkSize << endl;

                if (chunkSize == 0) {
                    //readState = DONE;
                    //return;
                }

                if (*endPtr != 0)
                    throw ML::Exception("invalid chunk length " + lengthStr);

                readState = CHUNK_BODY;
                chunkBody = "";
            }
            if (readState == CHUNK_BODY) {
                size_t chunkDataLeft = chunkSize - chunkBody.size();
                size_t dataAvail = end - current;
                size_t toRead = std::min(chunkDataLeft, dataAvail);

                chunkBody.append(current, current + toRead);
                current += toRead;

                if (chunkBody.length() == chunkSize) {

                    //cerr << "got chunk " << "-------------" << endl
                    //     << chunkBody << "--------------" << endl << endl;

                    handleHttpChunk(header, chunkHeader, chunkBody);
                    chunkBody = "";
                    chunkHeader = "";
                    readState = CHUNK_HEADER;
                }
            }
        }
    }
}

void
HttpConnectionHandler::
handleHttpChunk(const HttpHeader & header,
                const std::string & chunkHeader,
                const std::string & chunk)
{
    //cerr << "got chunk " << chunk << endl;
    handleHttpPayload(header, chunk);
}

void
HttpConnectionHandler::
handleHttpPayload(const HttpHeader & header,
                  const std::string & payload)
{
    throw Exception("no payload handler defined");
}

void
HttpConnectionHandler::
sendHttpChunk(const std::string & chunk,
              NextAction next,
              OnWriteFinished onWriteFinished)
{
    // Add the chunk header
    string fullChunk = ML::format("%zx\r\n%s\r\n", chunk.length(), chunk.c_str());
    send(fullChunk, next, onWriteFinished);
}

void
HttpConnectionHandler::
handleError(const std::string & message)
{
    cerr << "error: " << message << endl;
}

void
HttpConnectionHandler::
onCleanup()
{
}

void
HttpConnectionHandler::
putResponseOnWire(HttpResponse response,
                  std::function<void ()> onSendFinished,
                  NextAction next)
{
    onSendFinished = [=] ()
        {
#if 0
            Date finished = Date::now();
            double elapsedMs = finished.secondsSince(this->firstData) * 1000;
            cerr << "send finished in "
            << elapsedMs << "ms" << endl;


            if (elapsedMs > 100) {
                transport().dumpActivities();
                abort();
            }
#endif

            if (!onSendFinished)
                this->transport().associateWhenHandlerFinished
                    (this->makeNewHandlerShared(), "sendFinished");
            else onSendFinished();
        };

    std::string responseStr;
    responseStr.reserve(1024 + response.body.length());

    responseStr.append("HTTP/1.1 ");
    responseStr.append(to_string(response.responseCode));
    responseStr.append(" ");
    responseStr.append(response.responseStatus);
    responseStr.append("\r\n");

    if (response.contentType != "") {
        responseStr.append("Content-Type: ");
        responseStr.append(response.contentType);
        responseStr.append("\r\n");
    }

    if (response.sendBody) {
        responseStr.append("Content-Length: ");
        responseStr.append(to_string(response.body.length()));
        responseStr.append("\r\n");
        responseStr.append("Connection: Keep-Alive\r\n");
    }

    for (auto & h: response.extraHeaders) {
        responseStr.append(h.first);
        responseStr.append(": ");
        responseStr.append(h.second);
        responseStr.append("\r\n");
    }

    responseStr.append("\r\n");
    responseStr.append(response.body);

    //cerr << "sending " << responseStr << endl;
    
    send(responseStr,
         next,
         onSendFinished);
}


/*****************************************************************************/
/* HTTP ENDPOINT                                                             */
/*****************************************************************************/

HttpEndpoint::
HttpEndpoint(const std::string & name)
    : PassiveEndpointT<SocketTransport>(name)
{
    handlerFactory = [] ()
        {
            return std::make_shared<HttpConnectionHandler>();
        };
}

HttpEndpoint::
~HttpEndpoint()
{
}

template struct PassiveEndpointT<SocketTransport>;

} // namespace Datacratic
