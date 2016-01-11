/* json_endpoint.cc
   Jeremy Barnes, 22 February 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "soa/service//json_endpoint.h"
#include "jml/arch/exception.h"
#include "soa/jsoncpp/json.h"


using namespace ML;
using namespace std;


namespace Datacratic {


/*****************************************************************************/
/* JSON CONNECTION HANDLER                                                   */
/*****************************************************************************/

JsonConnectionHandler::
JsonConnectionHandler()
{
}

void
JsonConnectionHandler::
handleHttpHeader(const HttpHeader & header)
{
    if (!header.contentType.empty()
        && header.contentType.find("json") == string::npos
        && header.contentType.find("text") == string::npos)
        throw Exception("invalid content type '%s' for JSON",
                        header.contentType.c_str());
    if (header.verb != "POST")
        handleUnknownHeader(header);
}

void
JsonConnectionHandler::
handleUnknownHeader(const HttpHeader& header) {
    throw Exception("invalid verb");
}


void
JsonConnectionHandler::
handleHttpPayload(const HttpHeader & header,
                  const std::string & payload)
{
    Json::Value request;

    const char * start = payload.c_str();
    const char * end = start + payload.length();
    
    while (end > start && end[-1] == '\n') --end;
    
    try {
        Json::Reader reader;
        if (!reader.parse(start, end, request, false)) {
            //cerr << "JSON parsing error" << endl;
            doError("parsing JSON payload: "
                    + reader.getFormattedErrorMessages());
            return;
        }
    } catch (const std::exception & exc) {
        doError("parsing JSON request: " + string(exc.what()));
        return;
    }

    addActivity("finishedJsonParsing");

    handleJson(header, request, string(start, end));
}

void
JsonConnectionHandler::
handleHttpChunk(const HttpHeader & header,
                const std::string & chunkHeader,
                const std::string & chunk)
{
    Json::Value request;

    const char * start = chunk.c_str();
    const char * end = start + chunk.length();
    
    while (end > start && end[-1] == '\n') --end;
    
    try {
        Json::Reader reader;
        if (!reader.parse(start, end, request, false)) {
            //cerr << "JSON parsing error" << endl;
            doError("parsing JSON chunk: "
                    + reader.getFormattedErrorMessages());
            return;
        }
    } catch (const std::exception & exc) {
        doError("parsing JSON request: " + string(exc.what()));
        return;
    }

    addActivity("finishedJsonParsing");

    handleJson(header, request, string(start, end));
}

} // namespace Datacratic

